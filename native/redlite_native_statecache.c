#include "redlite_native_statecache.h"

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define RL_STATE_MAGIC "RLSTATE1"

typedef struct {
    char magic[8];
    uint64_t model_tag;
    uint64_t build_tag;
    uint32_t chunk;
    uint32_t n_ids;
    uint32_t vocab;
    uint32_t reserved;
    uint64_t state_bytes;
} rl_state_header;

static void set_error(char *dst, size_t cap, const char *msg) {
    if (dst && cap) snprintf(dst, cap, "%s", msg);
}

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const unsigned char *c = (const unsigned char *)p;
    for (size_t i = 0; i < n; ++i) { h ^= c[i]; h *= 1099511628211ull; }
    return h;
}

/* states are only comparable within one build: kernels decide the float rounding */
static uint64_t build_tag(void) {
    static const char stamp[] = __DATE__ " " __TIME__;
    return fnv(1469598103934665603ull, stamp, sizeof(stamp));
}

static void file_path(const rl_statecache *c, uint64_t model_tag, const uint32_t *ids, uint32_t n, char *out, size_t cap) {
    snprintf(out, cap, "%s/%016llx-%016llx.rlst", c->dir, (unsigned long long)model_tag,
             (unsigned long long)fnv(1469598103934665603ull, ids, (size_t)n * sizeof(uint32_t)));
}

static int header_ok(const rl_state_header *h, const rl_engine *e, uint32_t chunk) {
    return memcmp(h->magic, RL_STATE_MAGIC, 8) == 0 && h->model_tag == rl_engine_model_tag(e) &&
           h->build_tag == build_tag() && h->chunk == chunk && h->vocab == rl_engine_info_get(e)->vocab &&
           h->state_bytes == rl_engine_state_bytes(e, h->n_ids);
}

/* longest stored prefix of ids[0..count) for this engine; 0 if none */
static uint32_t find_prefix(const rl_statecache *c, rl_engine *e, uint32_t chunk, const uint32_t *ids, uint32_t count,
                            char *best, size_t cap) {
    DIR *d = opendir(c->dir);
    if (!d) return 0;
    char prefix[32];
    snprintf(prefix, sizeof(prefix), "%016llx-", (unsigned long long)rl_engine_model_tag(e));
    uint32_t best_n = 0;
    uint32_t *buf = NULL;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        const size_t len = strlen(de->d_name);
        if (len < 5 || strcmp(de->d_name + len - 5, ".rlst") != 0 || strncmp(de->d_name, prefix, 17) != 0) continue;
        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", c->dir, de->d_name);
        FILE *f = fopen(path, "rb");
        if (!f) continue;
        rl_state_header h;
        if (fread(&h, sizeof(h), 1, f) == 1 && header_ok(&h, e, chunk) && h.n_ids <= count && h.n_ids > best_n) {
            uint32_t *nb = (uint32_t *)realloc(buf, (size_t)h.n_ids * sizeof(uint32_t));
            if (nb) {
                buf = nb;
                if (fread(buf, sizeof(uint32_t), h.n_ids, f) == h.n_ids && memcmp(buf, ids, (size_t)h.n_ids * sizeof(uint32_t)) == 0) {
                    best_n = h.n_ids;
                    snprintf(best, cap, "%s", path);
                }
            }
        }
        fclose(f);
    }
    closedir(d);
    free(buf);
    return best_n;
}

static int load_file(rl_engine *e, rl_engine_backend b, const char *path, uint32_t n, float *logits, char *error, size_t cap) {
    FILE *f = fopen(path, "rb");
    if (!f) { set_error(error, cap, "state file open failed"); return 0; }
    const uint32_t vocab = rl_engine_info_get(e)->vocab;
    rl_state_header h;
    int ok = fread(&h, sizeof(h), 1, f) == 1 && h.n_ids == n &&
             fseeko(f, (off_t)sizeof(h) + (off_t)n * (off_t)sizeof(uint32_t), SEEK_SET) == 0;
    if (ok && logits) ok = fread(logits, sizeof(float), vocab, f) == vocab;
    else if (ok) ok = fseeko(f, (off_t)vocab * (off_t)sizeof(float), SEEK_CUR) == 0;
    ok = ok && rl_engine_state_read(e, b, f, n, error, cap);
    fclose(f);
    if (ok) utimensat(AT_FDCWD, path, NULL, 0);   /* recency for the trim */
    else if (error && cap && !error[0]) set_error(error, cap, "state file read failed");
    return ok;
}

static int save_file(const rl_statecache *c, rl_engine *e, rl_engine_backend b, uint32_t chunk, const uint32_t *ids, uint32_t n,
                     const float *logits, char *error, size_t cap) {
    char path[4096], tmp[4200];
    file_path(c, rl_engine_model_tag(e), ids, n, path, sizeof(path));
    snprintf(tmp, sizeof(tmp), "%s.%d.tmp", path, (int)getpid());
    FILE *f = fopen(tmp, "wb");
    if (!f) { set_error(error, cap, "state file create failed (does --state-dir exist?)"); return 0; }
    rl_state_header h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, RL_STATE_MAGIC, 8);
    h.model_tag = rl_engine_model_tag(e);
    h.build_tag = build_tag();
    h.chunk = chunk;
    h.n_ids = n;
    h.vocab = rl_engine_info_get(e)->vocab;
    h.state_bytes = rl_engine_state_bytes(e, n);
    int ok = fwrite(&h, sizeof(h), 1, f) == 1 && fwrite(ids, sizeof(uint32_t), n, f) == n &&
             fwrite(logits, sizeof(float), h.vocab, f) == h.vocab && rl_engine_state_write(e, b, f, error, cap);
    ok = (fclose(f) == 0) && ok;
    if (ok && rename(tmp, path) != 0) ok = 0;
    if (!ok) { unlink(tmp); if (error && cap && !error[0]) set_error(error, cap, "state file write failed"); }
    return ok;
}

typedef struct { char name[300]; off_t size; time_t mtime; } rl_state_entry;

static int by_mtime(const void *a, const void *b) {
    const time_t x = ((const rl_state_entry *)a)->mtime, y = ((const rl_state_entry *)b)->mtime;
    return x < y ? -1 : x > y;
}

/* delete least recently used .rlst files until the directory holds at most max_bytes of them */
static void trim(const rl_statecache *c) {
    if (!c->max_bytes) return;
    DIR *d = opendir(c->dir);
    if (!d) return;
    rl_state_entry *v = NULL;
    size_t n = 0, cap = 0;
    uint64_t total = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        const size_t len = strlen(de->d_name);
        if (len < 5 || len >= sizeof(v->name) || strcmp(de->d_name + len - 5, ".rlst") != 0) continue;
        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", c->dir, de->d_name);
        struct stat st;
        if (stat(path, &st) != 0) continue;
        if (n == cap) {
            rl_state_entry *nv = (rl_state_entry *)realloc(v, (cap ? cap * 2 : 16) * sizeof(*v));
            if (!nv) break;
            v = nv; cap = cap ? cap * 2 : 16;
        }
        snprintf(v[n].name, sizeof(v[n].name), "%s", de->d_name);
        v[n].size = st.st_size; v[n].mtime = st.st_mtime;
        total += (uint64_t)st.st_size;
        n++;
    }
    closedir(d);
    if (v) qsort(v, n, sizeof(*v), by_mtime);
    for (size_t i = 0; i < n && total > c->max_bytes; ++i) {
        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", c->dir, v[i].name);
        if (unlink(path) == 0) total -= (uint64_t)v[i].size;
    }
    free(v);
}

int rl_statecache_prefill(rl_engine *e, rl_engine_backend b, const rl_statecache *c, const uint32_t *ids, uint32_t count,
                          float *logits, rl_engine_step_stats *stats, uint32_t *loaded, uint32_t *saved, char *error, size_t cap) {
    if (loaded) *loaded = 0;
    if (saved) *saved = 0;
    if (error && cap) error[0] = '\0';
    if (!e || !ids || !count || !logits) { set_error(error, cap, "invalid state cache prefill arguments"); return 0; }
    const int enabled = c && c->dir && c->dir[0] && b == RL_BACKEND_GPU;
    const uint32_t chunk = rl_engine_prefill_batch(e);
    uint32_t pos = 0;
    if (enabled) {
        char path[4096];
        const uint32_t p = find_prefix(c, e, chunk, ids, count, path, sizeof(path));
        if (p) {
            char lerr[256] = "";
            if (load_file(e, b, path, p, p == count ? logits : NULL, lerr, sizeof(lerr))) pos = p;
            else unlink(path);   /* unreadable: drop it and run cold */
        }
    }
    if (!pos && !rl_engine_reset(e, b, error, cap)) return 0;
    if (loaded) *loaded = pos;
    if (stats) memset(stats, 0, sizeof(*stats));
    if (pos == count) return 1;
    /* checkpoint at the largest chunk-aligned position past the restored one (at least one chunk) */
    const uint32_t aligned = chunk > 1u ? count / chunk * chunk : 0u;
    if (enabled && aligned > pos && aligned >= chunk) {
        if (!rl_engine_prefill(e, b, ids + pos, aligned - pos, logits, stats, error, cap)) return 0;
        char serr[256] = "";
        if (save_file(c, e, b, chunk, ids, aligned, logits, serr, sizeof(serr))) {
            if (saved) *saved = aligned;
            trim(c);
        }
        pos = aligned;
        if (pos == count) return 1;
    }
    return rl_engine_prefill(e, b, ids + pos, count - pos, logits, stats, error, cap);
}
