#ifndef PSD_PROJECTION_HAOYU_FP32_H
#define PSD_PROJECTION_HAOYU_FP32_H

#include <cublas_v2.h>

/// @brief Projects a symmetric matrix to its positive semidefinite (PSD) form using FP32 precision. The matrix must be scaled such that its eigenvalues are in [-1, 1].
/// @param cublasH a cuBLAS handle
/// @param mat the matrix to be projected, stored in column-major order
/// @param mat_size the size of the matrix (assumed to be square, i.e., mat_size x mat_size)
void haoyu_FP32(
    cublasHandle_t cublasH,
    float* mat,
    const int n
);

#endif // PSD_PROJECTION_HAOYU_FP32_H