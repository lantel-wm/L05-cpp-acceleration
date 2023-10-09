#ifndef __GPU_HPP__
#define __GPU_HPP__

#include <cuda_runtime.h>

#define CHECK(call)                                                         \
{                                                                           \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess)                                               \
    {                                                                       \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                            \
    }                                                                       \
}

__global__ void calx_kernel(double* xens, double* zens_wrap, double* a, int ensemble_size, int model_size, int ss2, int smooth_steps);

void calx_gpu(double* xens, double* zens_wrap, double* a, int ensemble_size, int model_size, int ss2, int smooth_steps);

#endif // __GPU_HPP__