#include "redlite_native_iq2_xxs.h"
#include "redlite_native_shared_exec.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    char error[256] = {0};
    uint8_t grid[RL_IQ2_XXS_GRID_COUNT];
    if (!rl_native_iq2_xxs_build_grid(grid, error, sizeof(error))) {
        fprintf(stderr, "IQ2_XXS grid test failed: %s\n", error);
        return 1;
    }
    float x[256];
    for (unsigned i = 0; i < 256; ++i) x[i] = 1.0f;

    uint8_t q6[210];
    memset(q6, 0, sizeof(q6));
    memset(q6 + 0, 0x11, 128);     /* low nibbles = 1 for all four 32-wide groups */
    memset(q6 + 128, 0xaa, 64);    /* upper two bits = 2 -> quant 33 -> signed value +1 */
    memset(q6 + 192, 1, 16);       /* sub-block scales = 1 */
    q6[208] = 0x00; q6[209] = 0x3c; /* f16 1.0 */
    double q6_dot = 0.0;
    if (rl_native_shared_row_bytes(14u, 256u) != 210u ||
        !rl_native_shared_quant_row_dot(q6, sizeof(q6), 14u, x, 256u, grid, sizeof(grid), &q6_dot, error, sizeof(error)) ||
        fabs(q6_dot - 256.0) > 1e-9) {
        fprintf(stderr, "Q6_K synthetic decoder failed: dot=%.12f error=%s\n", q6_dot, error);
        return 1;
    }

    /* Real Q6_K blocks can use FP16 subnormal d values. The smallest positive
     * half subnormal is 2^-24, so 256 unit quants must sum to 2^-16. This
     * catches exponent-normalization mistakes that the f16=1.0 case cannot. */
    q6[208] = 0x01; q6[209] = 0x00; /* f16 smallest positive subnormal = 2^-24 */
    double q6_subnormal_dot = 0.0;
    const double q6_subnormal_expected = ldexp(1.0, -16);
    if (!rl_native_shared_quant_row_dot(q6, sizeof(q6), 14u, x, 256u, grid, sizeof(grid),
            &q6_subnormal_dot, error, sizeof(error)) ||
        fabs(q6_subnormal_dot - q6_subnormal_expected) > 1e-12) {
        fprintf(stderr, "Q6_K FP16-subnormal decoder failed: dot=%.12g expected=%.12g error=%s\n",
            q6_subnormal_dot, q6_subnormal_expected, error);
        return 1;
    }

    uint8_t iq2[66];
    memset(iq2, 0, sizeof(iq2));
    iq2[0] = 0x00; iq2[1] = 0x3c; /* f16 d=1, grid 0 magnitude 8, scale 0 -> value +1 */
    double iq2_dot = 0.0;
    if (rl_native_shared_row_bytes(16u, 256u) != 66u ||
        !rl_native_shared_quant_row_dot(iq2, sizeof(iq2), 16u, x, 256u, grid, sizeof(grid), &iq2_dot, error, sizeof(error)) ||
        fabs(iq2_dot - 256.0) > 1e-9) {
        fprintf(stderr, "IQ2_XXS synthetic decoder failed: dot=%.12f error=%s\n", iq2_dot, error);
        return 1;
    }

    printf("shared Q6_K decoder : OK\n");
    printf("shared Q6_K subnorm.: OK\n");
    printf("shared IQ2_XXS dec. : OK\n");
    printf("shared row sizes    : Q6_K=210 IQ2_XXS=66 / 256 values\n");
    return 0;
}
