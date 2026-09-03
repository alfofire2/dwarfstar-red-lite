#define main rl_dev15e_layer_cli_unused
#include "redlite_native_deltanet_layer_cli.c"
#undef main

#include "redlite_native_block_ops.h"
#include "redlite_native_gguf.h"
#include "redlite_native_metal.h"
#include "redlite_native_reference.h"
#include "redlite_native_router.h"
#include "redlite_native_router_exec.h"
#include "redlite_native_shared.h"
#include "redlite_native_shared_exec.h"

#define RB_MIB (1024ull * 1024ull)
#define RB_MAX_TOPK 64u
#define RB_MAX_SHARED (RL_SHARED_MAX_LAYERS * 4u)

typedef struct {
    uint64_t offset;
    uint64_t span;
    uint32_t type;
    uint32_t dims;
    uint64_t shape[RL_LAYER_MAX_DIMS];
    int found;
} rb_tensor;

static int rb_find_tensor(const char *model, const char *wanted, rb_tensor *out, char *error, size_t cap) {
    memset(out,0,sizeof(*out)); FILE *f=fopen(model,"rb"); if(!f){snprintf(error,cap,"open failed: %s",strerror(errno));return 0;}
    if(fseeko(f,0,SEEK_END)!=0){fclose(f);return 0;} off_t end=ftello(f); if(end<=0||fseeko(f,0,SEEK_SET)!=0){fclose(f);return 0;}
    uint8_t magic[4];uint32_t ver=0;uint64_t tc=0,kc=0;model_meta meta;
    if(!read_exact(f,magic,4)||memcmp(magic,"GGUF",4)||!read_u32(f,&ver)||(ver!=2u&&ver!=3u)||!read_u64(f,&tc)||!read_u64(f,&kc)||!read_metadata(f,kc,&meta)){fclose(f);snprintf(error,cap,"GGUF parse failed");return 0;}
    raw_tensor *raw=calloc((size_t)tc,sizeof(*raw)); if(!raw){fclose(f);return 0;}
    for(uint64_t i=0;i<tc;++i){uint32_t d=0;if(!read_string(f,&raw[i].name)||!read_u32(f,&d)||d>RL_LAYER_MAX_DIMS){free_raw(raw,tc);fclose(f);return 0;}raw[i].n_dims=d;for(uint32_t j=0;j<d;++j)if(!read_u64(f,&raw[i].shape[j])){free_raw(raw,tc);fclose(f);return 0;}if(!read_u32(f,&raw[i].ggml_type)||!read_u64(f,&raw[i].relative_offset)){free_raw(raw,tc);fclose(f);return 0;}}
    off_t de=ftello(f);fclose(f);uint64_t base=round_up_u64((uint64_t)de,meta.alignment);qsort(raw,(size_t)tc,sizeof(*raw),tensor_cmp);
    for(uint64_t i=0;i<tc;++i){uint64_t next=i+1<tc?raw[i+1].relative_offset:(uint64_t)end-base; if(strcmp(raw[i].name,wanted)==0){out->offset=base+raw[i].relative_offset;out->span=next-raw[i].relative_offset;out->type=raw[i].ggml_type;out->dims=raw[i].n_dims;memcpy(out->shape,raw[i].shape,sizeof(out->shape));out->found=1;break;}}
    free_raw(raw,tc);if(!out->found){snprintf(error,cap,"tensor %s not found",wanted);return 0;}return 1;
}

static int rb_attention(const char *model,uint32_t layer,float *input,uint32_t hidden,float *cpu_out,float *gpu_out,float *cpu_conv_next,float *gpu_conv_next,float *cpu_state_next,float *gpu_state_next,size_t *conv_count,size_t *state_count,float *eps_out,char *error,size_t cap) {
    model_meta m;tensor_info t[LT_COUNT];if(!audit_layer(model,layer,&m,t,error,cap))return 0;uint32_t h=0,channels=0,qk=0,hv=0;if(!validate_layout(&m,t,&h,&channels,&qk,&hv,error,cap)||h!=hidden)return 0;
    uint32_t zc=m.d_inner,bac=2u*m.dt_rank;size_t csc=(size_t)(m.d_conv-1u)*channels,rsc=(size_t)m.dt_rank*hv*hv,corec=(size_t)m.dt_rank*hv;
    *conv_count=csc;*state_count=rsc;*eps_out=m.rms_eps;
    size_t iqr=iq2_row_bytes(hidden),q8r=q8_row_bytes(hidden),q4r=q4_row_bytes(m.d_inner);
    size_t nb=(size_t)hidden*4,qb=(size_t)channels*iqr,zb=(size_t)zc*iqr,bab=(size_t)bac*q8r,cb=(size_t)m.d_conv*channels*4,vb=(size_t)m.dt_rank*4,snb=(size_t)hv*4,sob=(size_t)hidden*q4r;
    float *nw=malloc(nb),*cw=malloc(cb),*dt=malloc(vb),*av=malloc(vb),*snw=malloc(snb);uint8_t *qw=malloc(qb),*zw=malloc(zb),*baw=malloc(bab),*sow=malloc(sob);
    float *conv_state=calloc(csc,4),*rec_state=calloc(rsc,4),*cn=calloc(hidden,4),*cqv=calloc(channels,4),*cz=calloc(zc,4),*cba=calloc(bac,4),*cbet=calloc(m.dt_rank,4),*cgat=calloc(m.dt_rank,4),*ccv=calloc(channels,4),*cq=calloc(qk,4),*ck=calloc(qk,4),*cv=calloc(m.d_inner,4),*cd=calloc(corec,4),*ccore=calloc(corec,4),*cng=calloc(m.d_inner,4);
    float *gn=calloc(hidden,4),*gqv=calloc(channels,4),*gz=calloc(zc,4),*gba=calloc(bac,4),*gbet=calloc(m.dt_rank,4),*ggat=calloc(m.dt_rank,4),*gcv=calloc(channels,4),*gq=calloc(qk,4),*gk=calloc(qk,4),*gv=calloc(m.d_inner,4),*gd=calloc(corec,4),*gcore=calloc(corec,4),*gng=calloc(m.d_inner,4);
    if(!nw||!cw||!dt||!av||!snw||!qw||!zw||!baw||!sow||!conv_state||!rec_state||!cn||!cqv||!cz||!cba||!cbet||!cgat||!ccv||!cq||!ck||!cv||!cd||!ccore||!cng||!gn||!gqv||!gz||!gba||!gbet||!ggat||!gcv||!gq||!gk||!gv||!gd||!gcore||!gng){snprintf(error,cap,"attention allocation failed");goto fail;}
    make_conv_state(conv_state,csc);make_recurrent_state(rec_state,m.dt_rank,hv);
    int fd=open(model,O_RDONLY);if(fd<0){snprintf(error,cap,"open weights failed");goto fail;}uint64_t calls=0;int rok=pread_full(fd,nw,nb,t[LT_ATTN_NORM].offset,&calls)&&pread_full(fd,qw,qb,t[LT_QKV].offset,&calls)&&pread_full(fd,zw,zb,t[LT_Z].offset,&calls)&&pread_full(fd,baw,bab,t[LT_BA].offset,&calls)&&pread_full(fd,cw,cb,t[LT_CONV].offset,&calls)&&pread_full(fd,dt,vb,t[LT_DT].offset,&calls)&&pread_full(fd,av,vb,t[LT_A].offset,&calls)&&pread_full(fd,snw,snb,t[LT_SSM_NORM].offset,&calls)&&pread_full(fd,sow,sob,t[LT_SSM_OUT].offset,&calls);close(fd);if(!rok){snprintf(error,cap,"attention pread failed");goto fail;}
    uint8_t grid[RL_IQ2_XXS_GRID_COUNT];if(!rl_native_iq2_xxs_build_grid(grid,error,cap))goto fail;rmsnorm_cpu(input,nw,hidden,m.rms_eps,cn);
    for(uint32_t r=0;r<channels;++r){double d=0;if(!rl_native_shared_quant_row_dot(qw+(size_t)r*iqr,iqr,16u,cn,hidden,grid,sizeof(grid),&d,error,cap))goto fail;cqv[r]=(float)d;}
    for(uint32_t r=0;r<zc;++r){double d=0;if(!rl_native_shared_quant_row_dot(zw+(size_t)r*iqr,iqr,16u,cn,hidden,grid,sizeof(grid),&d,error,cap))goto fail;cz[r]=(float)d;}
    for(uint32_t r=0;r<bac;++r){double d=0;if(!q8_row_dot(baw+(size_t)r*q8r,cn,hidden,&d))goto fail;cba[r]=(float)d;}
    cpu_ba_params(cba,dt,av,m.dt_rank,m.n_group,cbet,cgat);cpu_conv_silu(conv_state,cqv,cw,channels,m.d_conv,ccv);cpu_l2_heads(ccv,0,m.d_state,m.n_group,m.rms_eps,cq);cpu_l2_heads(ccv,qk,m.d_state,m.n_group,m.rms_eps,ck);memcpy(cv,ccv+2u*qk,(size_t)m.d_inner*4);cpu_shift_conv_state(conv_state,cqv,channels,m.d_conv,cpu_conv_next);cpu_state_update(cq,ck,cv,cgat,cbet,rec_state,hv,m.n_group,m.dt_rank,cd,cpu_state_next,ccore);cpu_gated_norm(ccore,cz,snw,m.rms_eps,hv,m.dt_rank,cng);for(uint32_t r=0;r<hidden;++r){double d=0;if(!q4_row_dot(sow+(size_t)r*q4r,cng,m.d_inner,&d))goto fail;cpu_out[r]=(float)d;}
#ifdef __APPLE__
    rl_dn_proj_tensor_info pi[RL_DN_PROJ_TENSOR_COUNT];rl_dn_prestate_tensor_info pri[RL_DN_PRESTATE_TENSOR_COUNT];fill_proj_info(t,layer,pi);fill_prestate_info(t,layer,pri);rl_dn_proj_telemetry pt={0};rl_dn_prestate_telemetry prt={0};rl_dn_state_telemetry st={0};rl_dn_tail_telemetry tt={0};
    if(!rl_deltanet_proj_gpu_execute_full(model,pi,input,hidden,m.rms_eps,gn,hidden,gqv,channels,gz,zc,gba,bac,&pt,error,cap)||!rl_deltanet_prestate_gpu_execute(model,pri,gqv,channels,gba,bac,conv_state,(uint32_t)csc,m.d_conv,m.d_inner,m.d_state,m.dt_rank,m.n_group,m.rms_eps,gbet,ggat,gcv,gq,gk,gv,gpu_conv_next,&prt,error,cap)||!rl_deltanet_state_gpu_execute(gq,gk,gv,ggat,gbet,rec_state,hv,m.n_group,m.dt_rank,gd,gpu_state_next,gcore,&st,error,cap)||!rl_deltanet_tail_gpu_execute(gcore,gz,snw,sow,sob,m.rms_eps,hv,m.dt_rank,hidden,gng,gpu_out,&tt,error,cap))goto fail;
#else
    snprintf(error,cap,"Metal required");goto fail;
#endif
    free(nw);free(cw);free(dt);free(av);free(snw);free(qw);free(zw);free(baw);free(sow);free(conv_state);free(rec_state);free(cn);free(cqv);free(cz);free(cba);free(cbet);free(cgat);free(ccv);free(cq);free(ck);free(cv);free(cd);free(ccore);free(cng);free(gn);free(gqv);free(gz);free(gba);free(gbet);free(ggat);free(gcv);free(gq);free(gk);free(gv);free(gd);free(gcore);free(gng);return 1;
fail:
    free(nw);free(cw);free(dt);free(av);free(snw);free(qw);free(zw);free(baw);free(sow);free(conv_state);free(rec_state);free(cn);free(cqv);free(cz);free(cba);free(cbet);free(cgat);free(ccv);free(cq);free(ck);free(cv);free(cd);free(ccore);free(cng);free(gn);free(gqv);free(gz);free(gba);free(gbet);free(ggat);free(gcv);free(gq);free(gk);free(gv);free(gd);free(gcore);free(gng);return 0;
}

static const rl_router_tensor_info *rb_router(const rl_router_tensor_info *r,uint32_t n,uint32_t l){for(uint32_t i=0;i<n;++i)if(r[i].layer==l)return &r[i];return NULL;}
static int rb_shared(const rl_shared_tensor_info *a,uint32_t n,uint32_t l,rl_shared_tensor_info out[4]){uint8_t s[4]={0};memset(out,0,4*sizeof(*out));for(uint32_t i=0;i<n;++i)if(a[i].layer==l&&a[i].kind<=RL_SHARED_DOWN){uint32_t k=(uint32_t)a[i].kind;if(s[k])return 0;out[k]=a[i];s[k]=1;}return s[0]&&s[1]&&s[2]&&s[3];}

static int rb_ffn(const char *model,uint32_t layer,uint32_t topk,uint64_t cache_mib,const float *cpu_in,const float *gpu_in,uint32_t hidden,float *cpu_out,float *gpu_out,char *error,size_t cap){
    rl_router_tensor_info rs[RL_ROUTER_MAX_LAYERS];uint32_t rc=0;if(!rl_native_router_audit(model,rs,RL_ROUTER_MAX_LAYERS,&rc,error,cap))return 0;const rl_router_tensor_info *rt=rb_router(rs,rc,layer);if(!rt||rt->shape[0]!=hidden)return 0;uint32_t experts=(uint32_t)rt->shape[1];
    rl_shared_tensor_info *all=calloc(RB_MAX_SHARED,sizeof(*all)),sh[4];uint32_t sc=0;if(!all||!rl_native_shared_audit(model,all,RB_MAX_SHARED,&sc,error,cap)||!rb_shared(all,sc,layer,sh)){free(all);return 0;}free(all);
    float *cl=malloc((size_t)experts*4),*gl=malloc((size_t)experts*4),*cp=malloc((size_t)experts*4),*gp=malloc((size_t)experts*4),*gr=calloc(hidden,4),*gs=calloc(hidden,4);double *cr=calloc(hidden,sizeof(double)),*cs=calloc(hidden,sizeof(double));if(!cl||!gl||!cp||!gp||!gr||!gs||!cr||!cs)goto fail;
    rl_native_router_telemetry ct={0},gt={0};if(!rl_native_router_cpu_f32(model,rt,cpu_in,hidden,cl,experts,&ct,error,cap)||!rl_native_router_gpu_f32(model,rt,gpu_in,hidden,gl,experts,&gt,error,cap))goto fail;
    uint32_t ci[RB_MAX_TOPK],gi[RB_MAX_TOPK];float cw[RB_MAX_TOPK],gw[RB_MAX_TOPK];if(!rl_native_router_select_softmax_topk(cl,experts,topk,ci,cw,cp,error,cap)||!rl_native_router_select_softmax_topk(gl,experts,topk,gi,gw,gp,error,cap))goto fail;
    for(uint32_t k=0;k<topk;++k)if(ci[k]!=gi[k]){snprintf(error,cap,"router IDs diverged in composed block");goto fail;}
    rl_expert_map map;if(!rl_native_build_expert_map(model,experts,&map,error,cap))goto fail;rl_native_metal_runtime *mr=rl_native_metal_create(model,&map,cache_mib*RB_MIB,64u,error,cap);if(!mr){rl_native_free_expert_map(&map);goto fail;}rl_native_metal_telemetry mt={0};double cms=0;
    int ok=rl_native_metal_execute_topk(mr,&map,layer,gi,gw,topk,0,hidden,gpu_in,hidden,gr,hidden,&mt,error,cap)&&rl_native_reference_topk(model,&map,layer,ci,cw,topk,0,hidden,cpu_in,hidden,cr,hidden,&cms,error,cap);rl_native_metal_destroy(mr);rl_native_free_expert_map(&map);if(!ok)goto fail;
    rl_native_shared_telemetry cst={0},gst={0};if(!rl_native_shared_cpu_execute(model,sh,cpu_in,hidden,0,hidden,cs,hidden,&cst,error,cap)||!rl_native_shared_gpu_execute(model,sh,gpu_in,hidden,0,hidden,gs,hidden,&gst,error,cap))goto fail;
    for(uint32_t i=0;i<hidden;++i){cpu_out[i]=(float)(cr[i]+cs[i]);gpu_out[i]=gr[i]+gs[i];}
    free(cl);free(gl);free(cp);free(gp);free(gr);free(gs);free(cr);free(cs);return 1;
fail: free(cl);free(gl);free(cp);free(gp);free(gr);free(gs);free(cr);free(cs);return 0;
}

static void rb_cpu_resnorm(const float *a,const float *b,const float *w,uint32_t n,float eps,float *sum,float *norm){double ss=0;for(uint32_t i=0;i<n;++i){sum[i]=a[i]+b[i];ss+=(double)sum[i]*sum[i];}double inv=1.0/sqrt(ss/n+eps);for(uint32_t i=0;i<n;++i)norm[i]=(float)((double)sum[i]*inv*w[i]);}

int main(int argc,char **argv){
    if(argc<5||strcmp(argv[1],"parity")||strcmp(argv[3],"--layer")){fprintf(stderr,"usage: redlite-recurrent-block parity MODEL --layer N [--top-k 10] [--cache-mib 256]\n");return 2;}uint32_t layer=0;if(!parse_u32(argv[4],&layer))return 2;uint32_t topk=10;uint64_t cache=256;for(int i=5;i<argc;++i){if(i+1>=argc)return 2;if(!strcmp(argv[i],"--top-k")){uint32_t v=0;if(!parse_u32(argv[++i],&v))return 2;topk=v;}else if(!strcmp(argv[i],"--cache-mib")){char *e=NULL;cache=strtoull(argv[++i],&e,10);if(!e||*e)return 2;}else return 2;}if(!topk||topk>RB_MAX_TOPK||!cache)return 2;
    const char *model=argv[2];char error[512]={0};model_meta m;tensor_info tt[LT_COUNT];if(!audit_layer(model,layer,&m,tt,error,sizeof(error))){fprintf(stderr,"block audit failed: %s\n",error);return 1;}uint32_t hidden=0,ch=0,qk=0,hv=0;if(!validate_layout(&m,tt,&hidden,&ch,&qk,&hv,error,sizeof(error)))return 1;
    char pn[96];snprintf(pn,sizeof(pn),"blk.%u.post_attention_norm.weight",layer);rb_tensor post;if(!rb_find_tensor(model,pn,&post,error,sizeof(error))||post.type!=0u||post.dims!=1u||post.shape[0]!=hidden||post.span<(uint64_t)hidden*4){fprintf(stderr,"post norm audit failed: %s\n",error);return 1;}float *pw=malloc((size_t)hidden*4),*input=calloc(hidden,4),*ca=calloc(hidden,4),*ga=calloc(hidden,4),*cres=calloc(hidden,4),*gres=calloc(hidden,4),*cn=calloc(hidden,4),*gn=calloc(hidden,4),*cf=calloc(hidden,4),*gf=calloc(hidden,4),*cout=calloc(hidden,4),*gout=calloc(hidden,4);size_t ccap=(size_t)(m.d_conv-1u)*ch,scap=(size_t)m.dt_rank*hv*hv;float *ccs=calloc(ccap,4),*gcs=calloc(ccap,4),*css=calloc(scap,4),*gss=calloc(scap,4);if(!pw||!input||!ca||!ga||!cres||!gres||!cn||!gn||!cf||!gf||!cout||!gout||!ccs||!gcs||!css||!gss){fprintf(stderr,"block allocation failed\n");return 1;}make_input(input,hidden);int fd=open(model,O_RDONLY);uint64_t calls=0;if(fd<0||!pread_full(fd,pw,(size_t)hidden*4,post.offset,&calls)){if(fd>=0)close(fd);fprintf(stderr,"post norm read failed\n");return 1;}close(fd);size_t cc=0,ss=0;float eps=0;if(!rb_attention(model,layer,input,hidden,ca,ga,ccs,gcs,css,gss,&cc,&ss,&eps,error,sizeof(error))){fprintf(stderr,"composed attention failed: %s\n",error);return 1;}if(cc!=ccap||ss!=scap){fprintf(stderr,"state geometry changed\n");return 1;}
    rb_cpu_resnorm(input,ca,pw,hidden,eps,cres,cn);rl_block_ops_telemetry rnt={0},frt={0};if(!rl_block_residual_rmsnorm_gpu(input,ga,pw,hidden,eps,gres,gn,&rnt,error,sizeof(error))){fprintf(stderr,"Metal post norm failed: %s\n",error);return 1;}if(!rb_ffn(model,layer,topk,cache,cn,gn,hidden,cf,gf,error,sizeof(error))){fprintf(stderr,"composed FFN failed: %s\n",error);return 1;}for(uint32_t i=0;i<hidden;++i)cout[i]=cres[i]+cf[i];if(!rl_block_residual_gpu(gres,gf,hidden,gout,&frt,error,sizeof(error))){fprintf(stderr,"Metal final residual failed: %s\n",error);return 1;}
    error_stats ae=compare_arrays(ga,ca,hidden),re=compare_arrays(gres,cres,hidden),ne=compare_arrays(gn,cn,hidden),fe=compare_arrays(gf,cf,hidden),oe=compare_arrays(gout,cout,hidden),ce=compare_arrays(gcs,ccs,ccap),se=compare_arrays(gss,css,scap);int aok=within(ga,ca,hidden,5e-4f,1e-3f),rok=within(gres,cres,hidden,7e-4f,1e-3f),nok=within(gn,cn,hidden,8e-4f,1.5e-3f),fok=within(gf,cf,hidden,2e-3f,2e-3f),ook=within(gout,cout,hidden,3e-3f,2e-3f),cok=within(gcs,ccs,ccap,2e-3f,2e-4f),sok=within(gss,css,scap,5e-5f,5e-4f);int all=aok&&rok&&nok&&fok&&ook&&cok&&sok;
    printf("runtime              : complete native Qwen3-Next recurrent transformer block parity\n");printf("layer / top-k        : %u / %u\n",layer,topk);printf("hidden               : %u\n",hidden);printf("attention abs/rel    : %.6g / %.6g parity=%s\n",ae.max_abs,ae.max_rel,aok?"YES":"NO");printf("attention residual   : %.6g / %.6g parity=%s\n",re.max_abs,re.max_rel,rok?"YES":"NO");printf("post-attn RMSNorm    : %.6g / %.6g parity=%s\n",ne.max_abs,ne.max_rel,nok?"YES":"NO");printf("full FFN abs/rel     : %.6g / %.6g parity=%s\n",fe.max_abs,fe.max_rel,fok?"YES":"NO");printf("conv state abs/rel   : %.6g / %.6g parity=%s\n",ce.max_abs,ce.max_rel,cok?"YES":"NO");printf("recurrent state      : %.6g / %.6g parity=%s\n",se.max_abs,se.max_rel,sok?"YES":"NO");printf("FINAL block abs/rel  : %.6g / %.6g parity=%s\n",oe.max_abs,oe.max_rel,ook?"YES":"NO");printf("resnorm/final Metal  : %.3f / %.3f ms\n",rnt.compute_ms,frt.compute_ms);printf("COMPLETE BLOCK       : %s\n",all?"YES":"NO");for(uint32_t i=0;i<4;++i)printf("row %-3u             : gpu=%+.7f cpu=%+.7f delta=%+.3e\n",i,gout[i],cout[i],(double)gout[i]-cout[i]);
    free(pw);free(input);free(ca);free(ga);free(cres);free(gres);free(cn);free(gn);free(cf);free(gf);free(cout);free(gout);free(ccs);free(gcs);free(css);free(gss);return all?0:3;
}
