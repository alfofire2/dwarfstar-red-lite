#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include "redlite_native_deltanet_prestate.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define GGUF_DEFAULT_ALIGNMENT 32u
#define GGUF_MAX_STRING (1ull << 30)
#define GGUF_MAX_ARRAY  (1ull << 30)

enum {
    GGUF_U8 = 0, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32,
    GGUF_F32, GGUF_BOOL, GGUF_STRING, GGUF_ARRAY, GGUF_U64, GGUF_I64, GGUF_F64
};

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t shape[RL_DN_PRESTATE_MAX_DIMS];
    uint32_t ggml_type;
    uint64_t relative_offset;
    uint64_t absolute_offset;
    uint64_t span_bytes;
} raw_tensor;

typedef struct {
    uint32_t alignment;
    float rms_eps;
    uint32_t d_conv;
    uint32_t d_inner;
    uint32_t d_state;
    uint32_t dt_rank;
    uint32_t n_group;
    int have_eps, have_conv, have_inner, have_state, have_rank, have_group;
} dn_meta;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static int read_exact(FILE *f, void *dst, size_t n) { return n == 0 || fread(dst, 1, n, f) == n; }

static int read_u32(FILE *f, uint32_t *v) {
    unsigned char b[4];
    if (!read_exact(f, b, 4)) return 0;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 1;
}

static int read_u64(FILE *f, uint64_t *v) {
    unsigned char b[8];
    if (!read_exact(f, b, 8)) return 0;
    *v = (uint64_t)b[0] | ((uint64_t)b[1] << 8) | ((uint64_t)b[2] << 16) | ((uint64_t)b[3] << 24) |
         ((uint64_t)b[4] << 32) | ((uint64_t)b[5] << 40) | ((uint64_t)b[6] << 48) | ((uint64_t)b[7] << 56);
    return 1;
}

static int read_f32(FILE *f, float *v) {
    uint32_t bits = 0;
    if (!read_u32(f, &bits)) return 0;
    memcpy(v, &bits, sizeof(*v));
    return 1;
}

static int skip_bytes(FILE *f, uint64_t n) {
    while (n) {
        const uint64_t chunk = n > 0x3fffffffULL ? 0x3fffffffULL : n;
        if (fseeko(f, (off_t)chunk, SEEK_CUR) != 0) return 0;
        n -= chunk;
    }
    return 1;
}

static int read_string(FILE *f, char **out) {
    uint64_t n = 0;
    if (!read_u64(f, &n) || n > GGUF_MAX_STRING || n > SIZE_MAX - 1u) return 0;
    char *s = (char *)malloc((size_t)n + 1u);
    if (!s) return 0;
    if (!read_exact(f, s, (size_t)n)) { free(s); return 0; }
    s[n] = '\0'; *out = s; return 1;
}

static int skip_string(FILE *f) { uint64_t n = 0; return read_u64(f, &n) && n <= GGUF_MAX_STRING && skip_bytes(f, n); }

static size_t scalar_size(uint32_t type) {
    switch (type) {
        case GGUF_U8: case GGUF_I8: case GGUF_BOOL: return 1;
        case GGUF_U16: case GGUF_I16: return 2;
        case GGUF_U32: case GGUF_I32: case GGUF_F32: return 4;
        case GGUF_U64: case GGUF_I64: case GGUF_F64: return 8;
        default: return 0;
    }
}

static int skip_value(FILE *f, uint32_t type) {
    const size_t fixed = scalar_size(type);
    if (fixed) return skip_bytes(f, fixed);
    if (type == GGUF_STRING) return skip_string(f);
    if (type == GGUF_ARRAY) {
        uint32_t subtype = 0; uint64_t count = 0;
        if (!read_u32(f, &subtype) || !read_u64(f, &count) || count > GGUF_MAX_ARRAY) return 0;
        const size_t sf = scalar_size(subtype);
        if (sf) { if (count && count > UINT64_MAX / sf) return 0; return skip_bytes(f, count * sf); }
        for (uint64_t i = 0; i < count; ++i) if (!skip_value(f, subtype)) return 0;
        return 1;
    }
    return 0;
}

static int read_metadata(FILE *f, uint64_t count, dn_meta *m) {
    m->alignment = GGUF_DEFAULT_ALIGNMENT;
    for (uint64_t i = 0; i < count; ++i) {
        char *key = NULL; uint32_t type = 0;
        if (!read_string(f, &key) || !read_u32(f, &type)) { free(key); return 0; }
        uint32_t *u32dst = NULL; int *flag = NULL;
        if (strcmp(key, "general.alignment") == 0 && type == GGUF_U32) u32dst = &m->alignment;
        else if (strcmp(key, "qwen3next.ssm.conv_kernel") == 0 && type == GGUF_U32) { u32dst = &m->d_conv; flag = &m->have_conv; }
        else if (strcmp(key, "qwen3next.ssm.inner_size") == 0 && type == GGUF_U32) { u32dst = &m->d_inner; flag = &m->have_inner; }
        else if (strcmp(key, "qwen3next.ssm.state_size") == 0 && type == GGUF_U32) { u32dst = &m->d_state; flag = &m->have_state; }
        else if (strcmp(key, "qwen3next.ssm.time_step_rank") == 0 && type == GGUF_U32) { u32dst = &m->dt_rank; flag = &m->have_rank; }
        else if (strcmp(key, "qwen3next.ssm.group_count") == 0 && type == GGUF_U32) { u32dst = &m->n_group; flag = &m->have_group; }
        if (u32dst) {
            uint32_t v = 0; free(key); if (!read_u32(f, &v)) return 0; *u32dst = v; if (flag) *flag = 1; continue;
        }
        if (strcmp(key, "qwen3next.attention.layer_norm_rms_epsilon") == 0 && type == GGUF_F32) {
            free(key); if (!read_f32(f, &m->rms_eps) || !isfinite(m->rms_eps) || m->rms_eps < 0.0f) return 0; m->have_eps = 1; continue;
        }
        free(key); if (!skip_value(f, type)) return 0;
    }
    return m->alignment && m->have_eps && m->have_conv && m->have_inner && m->have_state && m->have_rank && m->have_group;
}

static uint64_t round_up_u64(uint64_t v, uint64_t a) {
    if (!a) return v; const uint64_t r = v % a; if (!r) return v; return v > UINT64_MAX - (a-r) ? 0 : v + a-r;
}

static int tensor_cmp(const void *a, const void *b) {
    const raw_tensor *ta = (const raw_tensor *)a, *tb = (const raw_tensor *)b;
    return ta->relative_offset < tb->relative_offset ? -1 : ta->relative_offset > tb->relative_offset ? 1 : 0;
}

static void free_raw(raw_tensor *raw, uint64_t count) {
    if (!raw) return; for (uint64_t i = 0; i < count; ++i) free(raw[i].name); free(raw);
}

static const char *kind_name(rl_dn_prestate_kind k) {
    switch (k) { case RL_DN_CONV1D: return "ssm_conv1d"; case RL_DN_DT: return "ssm_dt"; case RL_DN_A: return "ssm_a"; default: return "unknown"; }
}

static int target_kind(const char *name, uint32_t layer, rl_dn_prestate_kind *kind) {
    char e[96];
    snprintf(e, sizeof(e), "blk.%u.ssm_conv1d.weight", layer); if (strcmp(name, e) == 0) { *kind = RL_DN_CONV1D; return 1; }
    snprintf(e, sizeof(e), "blk.%u.ssm_dt.bias", layer); if (strcmp(name, e) == 0) { *kind = RL_DN_DT; return 1; }
    snprintf(e, sizeof(e), "blk.%u.ssm_a", layer); if (strcmp(name, e) == 0) { *kind = RL_DN_A; return 1; }
    return 0;
}

static int audit_layer(const char *model, uint32_t layer, dn_meta *meta,
        rl_dn_prestate_tensor_info out[RL_DN_PRESTATE_TENSOR_COUNT], char *error, size_t cap) {
    memset(meta, 0, sizeof(*meta)); memset(out, 0, RL_DN_PRESTATE_TENSOR_COUNT*sizeof(*out));
    FILE *f = fopen(model, "rb");
    if (!f) { snprintf(error, cap, "open failed: %s", strerror(errno)); return 0; }
    if (fseeko(f, 0, SEEK_END) != 0) { fclose(f); snprintf(error, cap, "seek failed"); return 0; }
    const off_t end = ftello(f); if (end <= 0 || fseeko(f, 0, SEEK_SET) != 0) { fclose(f); snprintf(error, cap, "invalid GGUF size"); return 0; }
    const uint64_t file_size = (uint64_t)end;
    unsigned char magic[4]; uint32_t version = 0; uint64_t tensor_count = 0, kv_count = 0;
    if (!read_exact(f, magic, 4) || memcmp(magic, "GGUF", 4) != 0 || !read_u32(f, &version) ||
        (version != 2u && version != 3u) || !read_u64(f, &tensor_count) || !read_u64(f, &kv_count)) {
        fclose(f); snprintf(error, cap, "invalid GGUF header"); return 0;
    }
    if (tensor_count > SIZE_MAX/sizeof(raw_tensor) || !read_metadata(f, kv_count, meta)) {
        fclose(f); snprintf(error, cap, "missing/invalid Qwen3-Next SSM metadata"); return 0;
    }
    raw_tensor *raw = (raw_tensor *)calloc((size_t)tensor_count, sizeof(*raw));
    if (!raw) { fclose(f); snprintf(error, cap, "out of memory for tensor directory"); return 0; }
    for (uint64_t i = 0; i < tensor_count; ++i) {
        uint32_t dims = 0;
        if (!read_string(f, &raw[i].name) || !read_u32(f, &dims) || dims > RL_DN_PRESTATE_MAX_DIMS) { free_raw(raw,tensor_count); fclose(f); snprintf(error,cap,"tensor descriptor parse failed"); return 0; }
        raw[i].n_dims = dims;
        for (uint32_t d = 0; d < dims; ++d) if (!read_u64(f, &raw[i].shape[d])) { free_raw(raw,tensor_count); fclose(f); snprintf(error,cap,"tensor shape parse failed"); return 0; }
        if (!read_u32(f, &raw[i].ggml_type) || !read_u64(f, &raw[i].relative_offset)) { free_raw(raw,tensor_count); fclose(f); snprintf(error,cap,"tensor type/offset parse failed"); return 0; }
    }
    const off_t directory_end = ftello(f); fclose(f);
    if (directory_end < 0) { free_raw(raw,tensor_count); snprintf(error,cap,"failed to locate GGUF data"); return 0; }
    const uint64_t data_base = round_up_u64((uint64_t)directory_end, meta->alignment);
    if (!data_base || data_base > file_size) { free_raw(raw,tensor_count); snprintf(error,cap,"invalid GGUF data base"); return 0; }
    qsort(raw,(size_t)tensor_count,sizeof(*raw),tensor_cmp); uint8_t seen[RL_DN_PRESTATE_TENSOR_COUNT]={0};
    for (uint64_t i = 0; i < tensor_count; ++i) {
        const uint64_t rel=raw[i].relative_offset, next=i+1<tensor_count?raw[i+1].relative_offset:file_size-data_base;
        raw[i].absolute_offset=data_base+rel; raw[i].span_bytes=next>=rel?next-rel:0;
        rl_dn_prestate_kind kind; if (!target_kind(raw[i].name, layer, &kind)) continue;
        if (seen[(uint32_t)kind]) { free_raw(raw,tensor_count); snprintf(error,cap,"duplicate DeltaNet prestate tensor"); return 0; }
        seen[(uint32_t)kind]=1; rl_dn_prestate_tensor_info *t=&out[(uint32_t)kind]; t->layer=layer; t->kind=kind; t->ggml_type=raw[i].ggml_type;
        t->n_dims=raw[i].n_dims; memcpy(t->shape,raw[i].shape,sizeof(t->shape)); t->tensor_offset=raw[i].absolute_offset; t->tensor_span_bytes=raw[i].span_bytes;
    }
    free_raw(raw,tensor_count);
    for (uint32_t k=0;k<RL_DN_PRESTATE_TENSOR_COUNT;++k) if(!seen[k]) { snprintf(error,cap,"missing %s for requested recurrent layer",kind_name((rl_dn_prestate_kind)k)); return 0; }
    return 1;
}

static int pread_full(int fd, void *dst, size_t bytes, uint64_t offset, uint64_t *calls) {
    unsigned char *p=(unsigned char *)dst; size_t done=0;
    while(done<bytes){ ssize_t n=pread(fd,p+done,bytes-done,(off_t)(offset+done)); if(n<0&&errno==EINTR)continue; if(calls)(*calls)++; if(n<=0)return 0; done+=(size_t)n; }
    return 1;
}

static void make_qkv(float *x, uint32_t n) { for(uint32_t i=0;i<n;++i)x[i]=(float)(((int)((i*29u+11u)%101u)-50)/64.0); }
static void make_ba(float *x, uint32_t n) { for(uint32_t i=0;i<n;++i)x[i]=(float)(((int)((i*13u+5u)%37u)-18)/16.0); }
static void make_state(float *x, uint32_t n) { for(uint32_t i=0;i<n;++i)x[i]=(float)(((int)((i*7u+19u)%43u)-21)/48.0); }

static double sigmoid_stable(double x) { if(x>=0.0){double z=exp(-x);return 1.0/(1.0+z);} double z=exp(x);return z/(1.0+z); }
static double softplus_stable(double x) { if(x>20.0)return x; if(x<-20.0)return exp(x); return log1p(exp(x)); }

static void cpu_ba_params(const float *ba,const float *dt,const float *avec,uint32_t rank,uint32_t groups,double *beta,double *gate){
    const uint32_t width=rank/groups, stride=2u*width;
    for(uint32_t h=0;h<rank;++h){uint32_t g=h/width,l=h-g*width; double b=ba[g*stride+l]; double a=(double)ba[g*stride+width+l]+dt[h]; beta[h]=sigmoid_stable(b); gate[h]=softplus_stable(a)*(double)avec[h];}
}

static void cpu_conv_silu(const float *state,const float *qkv,const float *kernel,uint32_t channels,uint32_t dconv,double *out){
    const uint32_t ns=dconv-1u;
    for(uint32_t c=0;c<channels;++c){double acc=0.0; for(uint32_t j=0;j<ns;++j)acc+=(double)state[(size_t)c*ns+j]*kernel[(size_t)c*dconv+j]; acc+=(double)qkv[c]*kernel[(size_t)c*dconv+ns]; out[c]=acc/(1.0+exp(-acc));}
}

static void cpu_l2_heads(const double *src,uint32_t offset,uint32_t head_dim,uint32_t heads,float eps,double *dst){
    for(uint32_t h=0;h<heads;++h){const uint32_t b=offset+h*head_dim; double ss=0.0; for(uint32_t j=0;j<head_dim;++j)ss+=src[b+j]*src[b+j]; const double scale=1.0/fmax(sqrt(ss),(double)eps); for(uint32_t j=0;j<head_dim;++j)dst[(size_t)h*head_dim+j]=src[b+j]*scale;}
}

static void cpu_shift_state(const float *state,const float *qkv,uint32_t channels,uint32_t dconv,double *next){
    const uint32_t ns=dconv-1u; for(uint32_t c=0;c<channels;++c){size_t b=(size_t)c*ns; for(uint32_t j=0;j+1u<ns;++j)next[b+j]=state[b+j+1u]; next[b+ns-1u]=qkv[c];}
}

static int compare_fd(const float *gpu,const double *cpu,size_t n,double *ma,double *mr){*ma=0.0;*mr=0.0;int ok=1;for(size_t i=0;i<n;++i){double ae=fabs((double)gpu[i]-cpu[i]),re=ae/fmax(fabs(cpu[i]),1e-12);if(ae>*ma)*ma=ae;if(re>*mr)*mr=re;if(ae>1e-4+1e-4*fabs(cpu[i]))ok=0;}return ok;}

static int selftest(void){
    const float x[2]={3.0f,4.0f}; double y[2]; cpu_l2_heads((const double[]){3.0,4.0},0,2,1,1e-6f,y); if(fabs(y[0]-0.6)>1e-12||fabs(y[1]-0.8)>1e-12)return 0;
    const float st[3]={1,2,3},q[1]={4}; double nx[3]; cpu_shift_state(st,q,1,4,nx); if(nx[0]!=2||nx[1]!=3||nx[2]!=4)return 0;
    (void)x; return 1;
}

static int parse_u64(const char *s,uint64_t *out){if(!s||!*s)return 0;errno=0;char *e=NULL;unsigned long long v=strtoull(s,&e,10);if(errno||!e||*e)return 0;*out=(uint64_t)v;return 1;}

static void usage(FILE *o){fprintf(o,"redlite-deltanet-prestate 0.3.0.dev15\n\nUsage:\n  redlite-deltanet-prestate parity MODEL --layer N\n  redlite-deltanet-prestate --selftest\n\nValidates Qwen3-Next beta/alpha transforms, conv-state shift, real F32 SSM conv+SiLU and pre-delta Q/K/V normalization.\n");}

int main(int argc,char **argv){
    if(argc==2&&strcmp(argv[1],"--selftest")==0){if(!selftest()){fprintf(stderr,"DeltaNet prestate selftest failed\n");return 1;}printf("DeltaNet conv shift: OK\nDeltaNet L2 norm   : OK\n");return 0;}
    if(argc<3||strcmp(argv[1],"parity")!=0){usage(argc>1?stderr:stdout);return 2;} const char *model=argv[2]; uint64_t layer64=UINT64_MAX;
    for(int i=3;i<argc;++i){if(strcmp(argv[i],"--layer")!=0||i+1>=argc||!parse_u64(argv[++i],&layer64)){fprintf(stderr,"invalid option\n");return 2;}}
    if(layer64==UINT64_MAX||layer64>UINT32_MAX){fprintf(stderr,"layer out of range\n");return 2;}
    char error[512]={0}; dn_meta m; rl_dn_prestate_tensor_info t[RL_DN_PRESTATE_TENSOR_COUNT];
    if(!audit_layer(model,(uint32_t)layer64,&m,t,error,sizeof(error))){fprintf(stderr,"DeltaNet prestate audit failed: %s\n",error);return 1;}
    if(!m.d_conv||!m.d_inner||!m.d_state||!m.dt_rank||!m.n_group||m.dt_rank%m.n_group||m.d_inner%m.dt_rank){fprintf(stderr,"invalid SSM metadata dimensions\n");return 1;}
    const uint32_t head_v=m.d_inner/m.dt_rank,qk_each=m.d_state*m.n_group,channels=2u*qk_each+m.d_inner,state_count=(m.d_conv-1u)*channels,ba_count=2u*m.dt_rank;
    if(t[0].ggml_type!=0u||t[0].n_dims!=2u||t[0].shape[0]!=m.d_conv||t[0].shape[1]!=channels||t[1].ggml_type!=0u||t[1].n_dims!=1u||t[1].shape[0]!=m.dt_rank||t[2].ggml_type!=0u||t[2].n_dims!=1u||t[2].shape[0]!=m.dt_rank){fprintf(stderr,"real recurrent tensors do not match SSM metadata\n");return 1;}
    const size_t conv_bytes=(size_t)m.d_conv*channels*sizeof(float),vec_bytes=(size_t)m.dt_rank*sizeof(float);
    if(t[0].tensor_span_bytes<conv_bytes||t[1].tensor_span_bytes<vec_bytes||t[2].tensor_span_bytes<vec_bytes){fprintf(stderr,"real recurrent tensor span mismatch\n");return 1;}

    float *qkv=malloc((size_t)channels*sizeof(float)),*ba=malloc((size_t)ba_count*sizeof(float)),*state=malloc((size_t)state_count*sizeof(float));
    float *convw=malloc(conv_bytes),*dt=malloc(vec_bytes),*avec=malloc(vec_bytes);
    double *cbeta=calloc(m.dt_rank,sizeof(double)),*cgate=calloc(m.dt_rank,sizeof(double)),*cconv=calloc(channels,sizeof(double)),*cq=calloc(qk_each,sizeof(double)),*ck=calloc(qk_each,sizeof(double)),*cv=calloc(m.d_inner,sizeof(double)),*cnext=calloc(state_count,sizeof(double));
    float *gbeta=calloc(m.dt_rank,sizeof(float)),*ggate=calloc(m.dt_rank,sizeof(float)),*gconv=calloc(channels,sizeof(float)),*gq=calloc(qk_each,sizeof(float)),*gk=calloc(qk_each,sizeof(float)),*gv=calloc(m.d_inner,sizeof(float)),*gnext=calloc(state_count,sizeof(float));
    if(!qkv||!ba||!state||!convw||!dt||!avec||!cbeta||!cgate||!cconv||!cq||!ck||!cv||!cnext||!gbeta||!ggate||!gconv||!gq||!gk||!gv||!gnext){fprintf(stderr,"out of memory for DeltaNet prestate parity\n");goto fail;}
    make_qkv(qkv,channels);make_ba(ba,ba_count);make_state(state,state_count);
    int fd=open(model,O_RDONLY);if(fd<0){fprintf(stderr,"open model failed: %s\n",strerror(errno));goto fail;}uint64_t cpu_calls=0;double r0=now_ms();int rok=pread_full(fd,convw,conv_bytes,t[0].tensor_offset,&cpu_calls)&&pread_full(fd,dt,vec_bytes,t[1].tensor_offset,&cpu_calls)&&pread_full(fd,avec,vec_bytes,t[2].tensor_offset,&cpu_calls);double r1=now_ms();close(fd);if(!rok){fprintf(stderr,"CPU prestate pread failed\n");goto fail;}
    double c0=now_ms();cpu_ba_params(ba,dt,avec,m.dt_rank,m.n_group,cbeta,cgate);cpu_conv_silu(state,qkv,convw,channels,m.d_conv,cconv);cpu_l2_heads(cconv,0,m.d_state,m.n_group,m.rms_eps,cq);cpu_l2_heads(cconv,qk_each,m.d_state,m.n_group,m.rms_eps,ck);for(uint32_t i=0;i<m.d_inner;++i)cv[i]=cconv[2u*qk_each+i];cpu_shift_state(state,qkv,channels,m.d_conv,cnext);double c1=now_ms();
    rl_dn_prestate_telemetry gt={0};
#ifdef __APPLE__
    if(!rl_deltanet_prestate_gpu_execute(model,t,qkv,channels,ba,ba_count,state,state_count,m.d_conv,m.d_inner,m.d_state,m.dt_rank,m.n_group,m.rms_eps,gbeta,ggate,gconv,gq,gk,gv,gnext,&gt,error,sizeof(error))){fprintf(stderr,"Metal prestate failed: %s\n",error);goto fail;}
#else
    fprintf(stderr,"DeltaNet prestate Metal parity requires macOS\n");goto fail;
#endif
    double beta_a,beta_r,gate_a,gate_r,conv_a,conv_r,q_a,q_r,k_a,k_r,v_a,v_r,next_a,next_r;
    int p_beta=compare_fd(gbeta,cbeta,m.dt_rank,&beta_a,&beta_r),p_gate=compare_fd(ggate,cgate,m.dt_rank,&gate_a,&gate_r),p_conv=compare_fd(gconv,cconv,channels,&conv_a,&conv_r),p_q=compare_fd(gq,cq,qk_each,&q_a,&q_r),p_k=compare_fd(gk,ck,qk_each,&k_a,&k_r),p_v=compare_fd(gv,cv,m.d_inner,&v_a,&v_r),p_next=compare_fd(gnext,cnext,state_count,&next_a,&next_r);
    printf("runtime            : native C + Metal DeltaNet prestate parity (no Python/ctypes)\n");
    printf("layer              : %u\n",(uint32_t)layer64);printf("SSM metadata       : d_conv=%u d_inner=%u d_state=%u dt_rank=%u groups=%u\n",m.d_conv,m.d_inner,m.d_state,m.dt_rank,m.n_group);
    printf("derived heads      : key=%u x %u, value=%u x %u\n",m.n_group,m.d_state,m.dt_rank,head_v);printf("conv channels      : %u\n",channels);printf("RMS/L2 epsilon     : %.9g\n",m.rms_eps);
    printf("CPU read           : %.3f ms / %" PRIu64 " calls / %.3f KiB\n",r1-r0,cpu_calls,(double)(conv_bytes+2u*vec_bytes)/1024.0);printf("GPU read           : %.3f ms / %" PRIu64 " calls / %.3f KiB\n",gt.read_ms,gt.read_calls,(double)gt.bytes_read/1024.0);printf("SSD during compute : %" PRIu64 " bytes / %" PRIu64 " calls\n",gt.ssd_during_compute_bytes,gt.ssd_during_compute_calls);printf("CPU / GPU compute  : %.3f / %.3f ms\n",c1-c0,gt.compute_ms);
    printf("beta max abs/rel   : %.6g / %.6g parity=%s\n",beta_a,beta_r,p_beta?"YES":"NO");printf("gate max abs/rel   : %.6g / %.6g parity=%s\n",gate_a,gate_r,p_gate?"YES":"NO");printf("conv+SiLU abs/rel : %.6g / %.6g parity=%s\n",conv_a,conv_r,p_conv?"YES":"NO");printf("Q L2 max abs/rel   : %.6g / %.6g parity=%s\n",q_a,q_r,p_q?"YES":"NO");printf("K L2 max abs/rel   : %.6g / %.6g parity=%s\n",k_a,k_r,p_k?"YES":"NO");printf("V split abs/rel    : %.6g / %.6g parity=%s\n",v_a,v_r,p_v?"YES":"NO");printf("conv state shift   : %.6g / %.6g parity=%s\n",next_a,next_r,p_next?"YES":"NO");
    const int all=p_beta&&p_gate&&p_conv&&p_q&&p_k&&p_v&&p_next;printf("prestate parity    : %s\n",all?"YES":"NO");
    printf("sample h0          : beta gpu=%+.7f cpu=%+.7f gate gpu=%+.7f cpu=%+.7f\n",gbeta[0],cbeta[0],ggate[0],cgate[0]);printf("sample q/k/v       : gpu=%+.7f/%+.7f/%+.7f cpu=%+.7f/%+.7f/%+.7f\n",gq[0],gk[0],gv[0],cq[0],ck[0],cv[0]);
    free(qkv);free(ba);free(state);free(convw);free(dt);free(avec);free(cbeta);free(cgate);free(cconv);free(cq);free(ck);free(cv);free(cnext);free(gbeta);free(ggate);free(gconv);free(gq);free(gk);free(gv);free(gnext);return all?0:2;
fail:
    free(qkv);free(ba);free(state);free(convw);free(dt);free(avec);free(cbeta);free(cgate);free(cconv);free(cq);free(ck);free(cv);free(cnext);free(gbeta);free(ggate);free(gconv);free(gq);free(gk);free(gv);free(gnext);return 1;
}
