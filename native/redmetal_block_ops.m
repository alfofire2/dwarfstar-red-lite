#define _POSIX_C_SOURCE 200809L
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "redlite_native_block_ops.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static void set_error(char *dst, size_t cap, const char *msg) {
    if (dst && cap) snprintf(dst, cap, "%s", msg ? msg : "unknown block Metal error");
}

static NSString * const kBlockOpsSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void block_residual_rmsnorm(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device const float *w [[buffer(2)]], device float *sum [[buffer(3)]], device float *norm [[buffer(4)]], constant uint &n [[buffer(5)]], constant float &eps [[buffer(6)]], uint gid [[thread_position_in_grid]]) {\n"
"  if (gid != 0u) return; float ss = 0.0f;\n"
"  for (uint i=0;i<n;++i) { float v=a[i]+b[i]; sum[i]=v; ss=fma(v,v,ss); }\n"
"  float inv=rsqrt(ss/float(n)+eps); for(uint i=0;i<n;++i) norm[i]=sum[i]*inv*w[i];\n"
"}\n"
"kernel void block_residual(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device float *out [[buffer(2)]], constant uint &n [[buffer(3)]], uint gid [[thread_position_in_grid]]) { if(gid<n) out[gid]=a[gid]+b[gid]; }\n";

static id<MTLComputePipelineState> pipe(id<MTLDevice> dev, id<MTLLibrary> lib, NSString *name, char *error, size_t cap) {
    id<MTLFunction> fn=[lib newFunctionWithName:name]; if(!fn){snprintf(error,cap,"missing Metal function %s",name.UTF8String);return nil;}
    NSError *e=nil; id<MTLComputePipelineState> p=[dev newComputePipelineStateWithFunction:fn error:&e];
    if(!p) snprintf(error,cap,"pipeline %s failed: %s",name.UTF8String,e.localizedDescription.UTF8String ?: "unknown"); return p;
}

int rl_block_residual_rmsnorm_gpu(const float *residual,const float *branch,const float *weight,uint32_t n,float eps,float *sum_out,float *norm_out,rl_block_ops_telemetry *tel,char *error,size_t cap) {
    if(!residual||!branch||!weight||!n||!sum_out||!norm_out){set_error(error,cap,"invalid residual RMSNorm args");return 0;}
    @autoreleasepool {
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); if(!dev||!dev.hasUnifiedMemory){set_error(error,cap,"Apple unified memory required");return 0;}
        id<MTLCommandQueue> q=[dev newCommandQueue]; NSError *e=nil; id<MTLLibrary> lib=[dev newLibraryWithSource:kBlockOpsSource options:nil error:&e];
        if(!q||!lib){snprintf(error,cap,"block Metal setup failed: %s",e.localizedDescription.UTF8String ?: "unknown");return 0;}
        id<MTLComputePipelineState> p=pipe(dev,lib,@"block_residual_rmsnorm",error,cap); if(!p)return 0;
        size_t bytes=(size_t)n*sizeof(float);
        id<MTLBuffer> a=[dev newBufferWithBytes:residual length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> b=[dev newBufferWithBytes:branch length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> w=[dev newBufferWithBytes:weight length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> s=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared]; id<MTLBuffer> o=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if(!a||!b||!w||!s||!o){set_error(error,cap,"block Metal allocation failed");return 0;}
        id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder]; [enc setComputePipelineState:p];
        [enc setBuffer:a offset:0 atIndex:0];[enc setBuffer:b offset:0 atIndex:1];[enc setBuffer:w offset:0 atIndex:2];[enc setBuffer:s offset:0 atIndex:3];[enc setBuffer:o offset:0 atIndex:4];[enc setBytes:&n length:sizeof(n) atIndex:5];[enc setBytes:&eps length:sizeof(eps) atIndex:6];
        [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[enc endEncoding]; double t0=now_ms();[cb commit];[cb waitUntilCompleted];double t1=now_ms();
        if(cb.status!=MTLCommandBufferStatusCompleted||cb.error){snprintf(error,cap,"block residual RMSNorm failed: %s",cb.error.localizedDescription.UTF8String ?: "unknown");return 0;}
        memcpy(sum_out,s.contents,bytes);memcpy(norm_out,o.contents,bytes);if(tel)tel->compute_ms=t1-t0;
    }
    if(error&&cap)error[0]='\0';return 1;
}

int rl_block_residual_gpu(const float *residual,const float *branch,uint32_t n,float *out,rl_block_ops_telemetry *tel,char *error,size_t cap) {
    if(!residual||!branch||!n||!out){set_error(error,cap,"invalid residual args");return 0;}
    @autoreleasepool {
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice();id<MTLCommandQueue> q=[dev newCommandQueue];NSError *e=nil;id<MTLLibrary> lib=[dev newLibraryWithSource:kBlockOpsSource options:nil error:&e];
        if(!dev||!q||!lib){set_error(error,cap,"block residual Metal setup failed");return 0;}id<MTLComputePipelineState> p=pipe(dev,lib,@"block_residual",error,cap);if(!p)return 0;
        size_t bytes=(size_t)n*sizeof(float);id<MTLBuffer>a=[dev newBufferWithBytes:residual length:bytes options:MTLResourceStorageModeShared];id<MTLBuffer>b=[dev newBufferWithBytes:branch length:bytes options:MTLResourceStorageModeShared];id<MTLBuffer>o=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer>cb=[q commandBuffer];id<MTLComputeCommandEncoder>enc=[cb computeCommandEncoder];[enc setComputePipelineState:p];[enc setBuffer:a offset:0 atIndex:0];[enc setBuffer:b offset:0 atIndex:1];[enc setBuffer:o offset:0 atIndex:2];[enc setBytes:&n length:sizeof(n) atIndex:3];NSUInteger tg=MIN((NSUInteger)256,p.maxTotalThreadsPerThreadgroup);[enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(tg?tg:1,1,1)];[enc endEncoding];double t0=now_ms();[cb commit];[cb waitUntilCompleted];double t1=now_ms();
        if(cb.status!=MTLCommandBufferStatusCompleted||cb.error){set_error(error,cap,"block residual Metal failed");return 0;}memcpy(out,o.contents,bytes);if(tel)tel->compute_ms=t1-t0;
    }if(error&&cap)error[0]='\0';return 1;
}
