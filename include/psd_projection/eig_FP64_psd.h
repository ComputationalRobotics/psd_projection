#ifndef PSD_PROJECTION_EIG_FP64_PSD_H
#define PSD_PROJECTION_EIG_FP64_PSD_H

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

/// @brief Projects a symmetric matrix to its positive semidefinite (PSD) form using FP64 precision with cuSOLVER package. 
/// @param cusolverH a cuSOLVER handle
/// @param cublasH a cuBLAS handle
/// @param mat the matrix to be projected, stored in column-major order
/// @param mat_size the size of the matrix (assumed to be square, i.e., mat_size x mat_size)
/// @param return_eigenvalues if true, the function will return the eigenvalues of the input matrix
double* eig_FP64_psd(
    cusolverDnHandle_t solverH, 
    cublasHandle_t cublasH,
    double* mat,
    size_t mat_size,
    bool return_eigenvalues = false
);

void eig_FP64_deflate(
    cusolverDnHandle_t solverH, 
    cublasHandle_t cublasH,
    double* mat,
    size_t mat_size,
    const size_t k = 40,
    const int maxiter = 100,
    const double tol = 1e-10,
    const bool verbose = false
);

#endif // PSD_PROJECTION_EIG_FP64_PSD_H