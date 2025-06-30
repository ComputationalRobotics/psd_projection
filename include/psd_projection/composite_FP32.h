#ifndef PSD_PROJECTION_COMPOSITE_FP32_H
#define PSD_PROJECTION_COMPOSITE_FP32_H

#include <cublas_v2.h>

/// @brief Projects a symmetric matrix to its positive semidefinite (PSD) form using FP32 precision. The matrix must be scaled such that its eigenvalues are in [-1, 1].
/// @param cublasH a cuBLAS handle
/// @param mat the matrix to be projected, stored in column-major order
/// @param mat_size the size of the matrix (assumed to be square, i.e., mat_size x mat_size)
/// @param mat_offset the offset in the matrix where the projection starts (default is 0)
void composite_FP32(
    cublasHandle_t cublasH,
    double* mat,
    const int mat_size,
    const int mat_offset = 0,
    const bool verbose = false
);

/// @brief Projects a symmetric matrix to its positive semidefinite (PSD) form using FP32 precision. The matrix is automatically scaled to ensure its eigenvalues are in [-1, 1].
/// @param cublasH a cuBLAS handle
/// @param mat the matrix to be projected, stored in column-major order
/// @param mat_size the size of the matrix (assumed to be square, i.e., mat_size x mat_size)
/// @param mat_offset the offset in the matrix where the projection starts (default is 0)
void composite_FP32_auto_scale(
    cublasHandle_t cublasH,
    cusolverDnHandle_t solverH,
    double* mat,
    const int mat_size,
    const int mat_offset = 0
);

/// @brief Projects a symmetric matrix to its positive semidefinite (PSD) form using FP32 precision. The matrix is automatically scaled to ensure its eigenvalues are in [-1, 1], and the largest eigenvalues are deflated to improve numerical stability.
/// @param cublasH a cuBLAS handle
/// @param mat the matrix to be projected, stored in column-major order
/// @param mat_size the size of the matrix (assumed to be square, i.e., mat_size x mat_size)
/// @param mat_offset the offset in the matrix where the projection starts (default is 0)
void composite_FP32_auto_scale_deflate(
    cublasHandle_t cublasH,
    cusolverDnHandle_t solverH,
    double* mat,
    const int mat_size,
    const int mat_offset = 0,
    const size_t k = 40,
    const double tol = 1e-10,
    const double ortho_tol = 1e-3,
    const bool verbose = false
);

#endif // PSD_PROJECTION_COMPOSITE_FP32_H