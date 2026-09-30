#pragma once

#include "redlite_native_deltanet_proj.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __APPLE__
int rl_deltanet_proj_gpu_execute_full(
    const char *model_path,
    const rl_dn_proj_tensor_info tensors[RL_DN_PROJ_TENSOR_COUNT],
    const float *input,
    uint32_t input_count,
    float rms_eps,
    float *norm_out,
    uint32_t norm_out_count,
    float *qkv_out,
    uint32_t qkv_out_count,
    float *z_out,
    uint32_t z_out_count,
    float *ba_out,
    uint32_t ba_out_count,
    rl_dn_proj_telemetry *telemetry,
    char *error,
    size_t error_cap);
#endif

#ifdef __cplusplus
}
#endif
