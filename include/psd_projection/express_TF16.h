#ifndef PSD_PROJECTION_EXPRESS_TF16_H
#define PSD_PROJECTION_EXPRESS_TF16_H

#include <cublas_v2.h>

/// @brief Projects a symmetric matrix to its positive semidefinite (PSD) form using TF16 precision. The matrix must be scaled such that its eigenvalues are in [-1, 1].
/// @param cublasH a cuBLAS handle
/// @param mat the matrix to be projected, stored in column-major order
/// @param mat_size the size of the matrix (assumed to be square, i.e., mat_size x mat_size)
/// @param mat_offset the offset in the matrix where the projection starts (default is 0)
void express_TF16(
    cublasHandle_t cublasH,
    double* mat,
    const int n,
    const int mat_offset = 0
);

#endif // PSD_PROJECTION_EXPRESS_TF16_H