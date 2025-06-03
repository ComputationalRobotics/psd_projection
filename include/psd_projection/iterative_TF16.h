#ifndef PSD_PROJECTION_ITERATIVE_TF16_H
#define PSD_PROJECTION_ITERATIVE_TF16_H

#include <cublas_v2.h>

__global__ void float2half_kernel(const float* A, __half* B, size_t N);

__global__ void float4_to_half_kernel(
    const float4* __restrict__ A4,
    __half2 * __restrict__ B2,
    size_t N4
);

void convertFloatToHalf4(const float* dA, __half* dB, size_t N);

void projection_TF16(
    cublasHandle_t cublasH,
    double* mat,
    const int mat_size,
    const int mat_offset = 0,
    const int n_iter = 100
);

#endif // PSD_PROJECTION_ITERATIVE_TF16_H