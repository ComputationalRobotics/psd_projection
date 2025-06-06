#ifndef PSD_PROJECTION_EXPRESS_FP32_LT_H
#define PSD_PROJECTION_EXPRESS_FP32_LT_H

#include <cublas_v2.h>
#include <cublasLt.h>

/// @brief Projects a symmetric matrix to its positive semidefinite (PSD) form using FP32 precision.
/// @param cublasH a cuBLAS handle
/// @param cublasLtH a cuBLASLt handle
/// @param mat the matrix to be projected, stored in column-major order
/// @param mat_size the size of the matrix (assumed to be square, i.e., mat_size x mat_size)
/// @param mat_offset the offset in the matrix where the projection starts (default is 0)
void express_FP32_Lt(
    cublasHandle_t cublasH,
    cublasLtHandle_t cublasLtH,
    double* mat,
    const int mat_size,
    const int mat_offset = 0
);

#endif // PSD_PROJECTION_EXPRESS_FP32_LT_H