#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include "redlite_native_reference.h"
#include "redlite_native_iq3.h"

#include "redlite_native_model.h"
#include "redlite_native_tables.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define QK_IQ 256u

static void set_error(char *error, size_t cap, const char *message) {
    if (error && cap) snprintf(error, cap, "%s", message ? message : "unknown native reference error");
}

static void set_errno_error(char *error, size_t cap, const char *prefix) {
    if (error && cap) snprintf(error, cap, "%s: %s", prefix, strerror(errno));
}

static uint16_t rd16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                e--;
            }
            mant &= 0x03ffu;
            bits = sign | (uint32_t)(e + 127) << 23 | mant << 13;
        }
    } else if (exp == 31u) {
        bits = sign | 0x7f800000u | mant << 13;
    } else {
        bits = sign | (exp + 112u) << 23 | mant << 13;
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint8_t sign_byte(uint16_t q) {
    uint8_t sign7 = (uint8_t)((q >> 9) & 0x7fu);
    uint8_t v = sign7;
    unsigned parity = 0;
    while (v) {
        parity ^= v & 1u;
        v >>= 1;
    }
    return (uint8_t)(sign7 | (parity << 7));
}

size_t rl_native_row_bytes(uint32_t ggml_type, uint32_t ncols) {
    if (!ncols || ncols % QK_IQ) return 0;
    if (ggml_type == 17u) return (size_t)(ncols / QK_IQ) * 74u;
    if (ggml_type == 29u) return (size_t)(ncols / QK_IQ) * 56u;
    return rl_iq3_row_bytes(ggml_type, ncols);   /* dev31: IQ3_XXS / IQ3_S experts (0 for other types) */
}

static double input_value(const float *xf, const double *xd, int use_double, uint32_t index) {
    return use_double ? xd[index] : (double)xf[index];
}

static int row_dot_impl(
        const uint8_t *row,
        size_t row_bytes,
        uint32_t ggml_type,
        const float *input_f,
        const double *input_d,
        int use_double,
        uint32_t ncols,
        const int8_t *grid,
        size_t grid_count,
        double *out,
        char *error,
        size_t error_cap) {
    if (!row || !out || (!input_f && !input_d) || (!grid && !rl_iq3_supported(ggml_type))) {
        set_error(error, error_cap, "invalid native quant row-dot arguments");
        return 0;
    }
    if (rl_iq3_supported(ggml_type)) {   /* dev31: exact f32 dequantization (bit-identical to ggml), double accumulation */
        (void)grid_count;
        if (row_bytes != rl_iq3_row_bytes(ggml_type, ncols) ||
            !(use_double ? rl_iq3_row_dot_d(ggml_type, row, input_d, ncols, out) : rl_iq3_row_dot(ggml_type, row, input_f, ncols, out))) {
            set_error(error, error_cap, "native IQ3 row dot failed");
            return 0;
        }
        if (error && error_cap) error[0] = '\0';
        return 1;
    }
    const size_t expected = rl_native_row_bytes(ggml_type, ncols);
    if (!expected || row_bytes != expected) {
        set_error(error, error_cap, "native quant row byte count mismatch");
        return 0;
    }

    double acc = 0.0;
    if (ggml_type == 17u) {
        if (grid_count != RL_IQ2_XS_GRID_COUNT) {
            set_error(error, error_cap, "IQ2_XS native grid size mismatch");
            return 0;
        }
        const uint32_t blocks = ncols / QK_IQ;
        for (uint32_t ib = 0; ib < blocks; ++ib) {
            const uint8_t *bp = row + (size_t)ib * 74u;
            const double d = (double)half_to_float(rd16(bp));
            const uint8_t *scales = bp + 66;
            for (uint32_t group = 0; group < 16u; ++group) {
                const uint32_t scale = (scales[group >> 1] >> (4u * (group & 1u))) & 0x0fu;
                const double db = d * (0.5 + (double)scale) * 0.25;
                for (uint32_t part = 0; part < 2u; ++part) {
                    const uint16_t q = rd16(bp + 2u + 2u * (2u * group + part));
                    const uint32_t grid_index = q & 0x01ffu;
                    const uint8_t signs = sign_byte(q);
                    const uint32_t base = ib * QK_IQ + group * 16u + part * 8u;
                    const uint32_t gbase = grid_index * 8u;
                    for (uint32_t j = 0; j < 8u; ++j) {
                        const double sign = (signs & (1u << j)) ? -1.0 : 1.0;
                        acc += input_value(input_f, input_d, use_double, base + j) *
                               (db * (double)grid[gbase + j] * sign);
                    }
                }
            }
        }
    } else if (ggml_type == 29u) {
        if (grid_count != RL_IQ1_M_GRID_COUNT) {
            set_error(error, error_cap, "IQ1_M native grid size mismatch");
            return 0;
        }
        const uint32_t blocks = ncols / QK_IQ;
        for (uint32_t ib = 0; ib < blocks; ++ib) {
            const uint8_t *bp = row + (size_t)ib * 56u;
            const uint8_t *qs = bp;
            const uint8_t *qh = bp + 32;
            uint16_t sc[4];
            for (uint32_t i = 0; i < 4u; ++i) sc[i] = rd16(bp + 48u + 2u * i);
            const uint16_t d_bits = (uint16_t)((sc[0] >> 12) |
                ((sc[1] >> 8) & 0x00f0u) | ((sc[2] >> 4) & 0x0f00u) | (sc[3] & 0xf000u));
            const double d = (double)half_to_float(d_bits);
            for (uint32_t group = 0; group < 32u; ++group) {
                const uint32_t nibble = (qh[group >> 1] >> (4u * (group & 1u))) & 0x0fu;
                const uint32_t grid_index = (uint32_t)qs[group] | ((nibble & 7u) << 8);
                const double delta = (nibble & 8u) ? -0.125 : 0.125;
                const uint32_t scale_index = group >> 1;
                const uint32_t scale = (sc[scale_index >> 2] >> (3u * (scale_index & 3u))) & 7u;
                const double dl = d * (double)(2u * scale + 1u);
                const uint32_t base = ib * QK_IQ + group * 8u;
                const uint32_t gbase = grid_index * 8u;
                for (uint32_t j = 0; j < 8u; ++j) {
                    const double weight = dl * ((double)grid[gbase + j] + delta);
                    acc += input_value(input_f, input_d, use_double, base + j) * weight;
                }
            }
        }
    } else {
        set_error(error, error_cap, "unsupported native routed quant type");
        return 0;
    }
    *out = acc;
    if (error && error_cap) error[0] = '\0';
    return 1;
}

int rl_native_quant_row_dot(
        const uint8_t *row,
        size_t row_bytes,
        uint32_t ggml_type,
        const float *input,
        uint32_t ncols,
        const int8_t *grid,
        size_t grid_count,
        double *out,
        char *error,
        size_t error_cap) {
    return row_dot_impl(row, row_bytes, ggml_type, input, NULL, 0, ncols,
                        grid, grid_count, out, error, error_cap);
}

static int pread_exact(int fd, uint8_t *dst, size_t len, uint64_t offset, char *error, size_t error_cap) {
    size_t pos = 0;
    while (pos < len) {
        ssize_t n;
        do {
            n = pread(fd, dst + pos, len - pos, (off_t)(offset + pos));
        } while (n < 0 && errno == EINTR);
        if (n <= 0) {
            set_errno_error(error, error_cap, n < 0 ? "native reference pread failed" : "native reference unexpected EOF");
            return 0;
        }
        pos += (size_t)n;
    }
    return 1;
}

static int reference_rows_fd(
        int fd,
        uint64_t matrix_offset,
        uint32_t ggml_type,
        uint32_t ncols,
        uint32_t row_start,
        uint32_t row_count,
        const float *input_f,
        const double *input_d,
        int use_double,
        const int8_t *grid,
        size_t grid_count,
        double *out,
        char *error,
        size_t error_cap) {
    const size_t row_bytes = rl_native_row_bytes(ggml_type, ncols);
    if (!row_bytes) {
        set_error(error, error_cap, "invalid native reference row width");
        return 0;
    }
    uint8_t *row = (uint8_t *)malloc(row_bytes);
    if (!row) {
        set_error(error, error_cap, "out of memory for native reference row");
        return 0;
    }
    for (uint32_t r = 0; r < row_count; ++r) {
        const uint64_t row_index = (uint64_t)row_start + r;
        if (!pread_exact(fd, row, row_bytes, matrix_offset + row_index * row_bytes, error, error_cap) ||
            !row_dot_impl(row, row_bytes, ggml_type, input_f, input_d, use_double, ncols,
                          grid, grid_count, &out[r], error, error_cap)) {
            free(row);
            return 0;
        }
    }
    free(row);
    return 1;
}

static double silu_mul(double gate, double up) {
    double sig;
    if (gate >= 0.0) sig = 1.0 / (1.0 + exp(-gate));
    else {
        const double eg = exp(gate);
        sig = eg / (1.0 + eg);
    }
    return (gate * sig) * up;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int reference_expert_fd(
        int fd,
        const rl_expert_layout *layout,
        uint32_t hidden_size,
        uint32_t ffn_size,
        uint32_t row_start,
        uint32_t row_count,
        const float *input,
        const rl_native_quant_tables *tables,
        double *out,
        char *error,
        size_t error_cap) {
    const int8_t *grid = layout->ggml_type == 17u ? tables->iq2_xs : layout->ggml_type == 29u ? tables->iq1_m : NULL;
    const size_t grid_count = layout->ggml_type == 17u ? RL_IQ2_XS_GRID_COUNT : RL_IQ1_M_GRID_COUNT;
    const int8_t *dgrid = layout->down_type == 17u ? tables->iq2_xs : layout->down_type == 29u ? tables->iq1_m : NULL;
    const size_t dgrid_count = layout->down_type == 17u ? RL_IQ2_XS_GRID_COUNT : RL_IQ1_M_GRID_COUNT;
    double *gate = (double *)malloc((size_t)ffn_size * sizeof(double));
    double *up = (double *)malloc((size_t)ffn_size * sizeof(double));
    double *act = (double *)malloc((size_t)ffn_size * sizeof(double));
    if (!gate || !up || !act) {
        free(gate); free(up); free(act);
        set_error(error, error_cap, "out of memory for native expert reference activations");
        return 0;
    }
    int ok = reference_rows_fd(fd, layout->gate_offset, layout->ggml_type, hidden_size,
                               0u, ffn_size, input, NULL, 0, grid, grid_count,
                               gate, error, error_cap) &&
             reference_rows_fd(fd, layout->up_offset, layout->ggml_type, hidden_size,
                               0u, ffn_size, input, NULL, 0, grid, grid_count,
                               up, error, error_cap);
    if (ok) {
        for (uint32_t i = 0; i < ffn_size; ++i) act[i] = silu_mul(gate[i], up[i]);
        ok = reference_rows_fd(fd, layout->down_offset, layout->down_type, ffn_size,
                               row_start, row_count, NULL, act, 1, dgrid, dgrid_count,
                               out, error, error_cap);
    }
    free(gate); free(up); free(act);
    return ok;
}

int rl_native_reference_topk(
        const char *model_path,
        const rl_expert_map *map,
        uint32_t layer,
        const uint32_t *expert_ids,
        const float *router_weights,
        uint32_t top_k,
        uint32_t row_start,
        uint32_t row_count,
        const float *input,
        uint32_t input_count,
        double *output,
        uint32_t output_count,
        double *elapsed_ms,
        char *error,
        size_t error_cap) {
    if (!model_path || !map || !expert_ids || !router_weights || !top_k || !input || !output ||
        !row_count || output_count < row_count) {
        set_error(error, error_cap, "invalid native top-k reference arguments");
        return 0;
    }
    rl_native_layer_info info;
    if (!rl_native_get_layer_info(map, layer, &info, error, error_cap)) return 0;
    if (input_count != info.hidden_size || row_start >= info.hidden_size || row_count > info.hidden_size - row_start) {
        set_error(error, error_cap, "native top-k reference dimension mismatch");
        return 0;
    }

    rl_native_quant_tables tables;
    if (!rl_native_quant_tables_init(&tables, error, error_cap)) return 0;

    int fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        rl_native_quant_tables_free(&tables);
        set_errno_error(error, error_cap, "open native reference model failed");
        return 0;
    }
    for (uint32_t r = 0; r < row_count; ++r) output[r] = 0.0;
    double *expert_out = (double *)malloc((size_t)row_count * sizeof(double));
    if (!expert_out) {
        close(fd);
        rl_native_quant_tables_free(&tables);
        set_error(error, error_cap, "out of memory for native top-k reference output");
        return 0;
    }

    const double t0 = now_ms();
    int ok = 1;
    for (uint32_t i = 0; i < top_k && ok; ++i) {
        if (expert_ids[i] >= map->expert_count) {
            set_error(error, error_cap, "native reference expert id out of range");
            ok = 0;
            break;
        }
        rl_expert_layout layout;
        if (!rl_native_expert_layout(map, layer, expert_ids[i], &layout, error, error_cap) ||
            layout.ggml_type != info.ggml_type || layout.down_type != info.down_type ||
            !reference_expert_fd(fd, &layout, info.hidden_size, info.ffn_size,
                                 row_start, row_count, input, &tables,
                                 expert_out, error, error_cap)) {
            ok = 0;
            break;
        }
        for (uint32_t r = 0; r < row_count; ++r)
            output[r] += (double)router_weights[i] * expert_out[r];
    }
    const double ms = now_ms() - t0;

    free(expert_out);
    close(fd);
    rl_native_quant_tables_free(&tables);
    if (!ok) return 0;
    if (elapsed_ms) *elapsed_ms = ms;
    if (error && error_cap) error[0] = '\0';
    return 1;
}
