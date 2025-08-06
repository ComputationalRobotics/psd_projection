#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <chrono>
#include <curand_kernel.h>
#include <algorithm>
#include <iomanip>
#include <assert.h>
#include <cuda.h>
#include <random>

#include "psd_projection/check.h"
#include "psd_projection/utils.h"
#include "psd_projection/eig_FP64_psd.h"

double* eig_FP64_psd(cusolverDnHandle_t solverH, cublasHandle_t cublasH, double* dA, size_t n, bool return_eigenvalues) {
    int *devInfo; CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));
    size_t nn = n * n;
    double one_d = 1.0;
    double zero_d = 0.0;

    double *dW, *dW_out;
    CHECK_CUDA(cudaMalloc(&dW, n*sizeof(double)));

    int lwork_ev = 0;
    CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(
        solverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
        n, dA, n, dW, &lwork_ev));
    double *dWork_ev; CHECK_CUDA(cudaMalloc(&dWork_ev, lwork_ev*sizeof(double)));
    CHECK_CUSOLVER(cusolverDnDsyevd(
        solverH,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        n, dA, n, dW,
        dWork_ev, lwork_ev, devInfo));
    CHECK_CUDA(cudaDeviceSynchronize());

    if (return_eigenvalues) { // save the eigevalues before zeroing them out
        CHECK_CUDA(cudaMalloc(&dW_out, n*sizeof(double)));
        CHECK_CUDA(cudaMemcpy(dW_out, dW, n*sizeof(double), cudaMemcpyDeviceToDevice));
    }

    max_dense_vector_zero(dW, n);

    // Copy eigenvectors from dA to dV
    double *dV; CHECK_CUDA(cudaMalloc(&dV, nn*sizeof(double)));
    CHECK_CUDA(cudaMemcpy(dV, dA, nn*sizeof(double), cudaMemcpyDeviceToDevice));

    
    // Scale columns of dV by W_h
    CHECK_CUBLAS(cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_DEVICE));
    for(int i = 0; i < n; i++){
        CHECK_CUBLAS(cublasDscal(cublasH, n, &dW[i], dV + i*n, 1));
    }
    CHECK_CUBLAS(cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_HOST));

    // Reconstruct A_psd = V * V^T
    double *dTmp; CHECK_CUDA(cudaMalloc(&dTmp, nn*sizeof(double)));
    CHECK_CUBLAS(cublasGemmEx(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
        n, n, n,
        &one_d,
        dV, CUDA_R_64F, n,
        dA, CUDA_R_64F, n,
        &zero_d,
        dTmp, CUDA_R_64F, n,
        CUDA_R_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaMemcpy(dA, dTmp, nn*sizeof(double), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaFree(dTmp));
    CHECK_CUDA(cudaFree(dW));
    CHECK_CUDA(cudaFree(dV));
    CHECK_CUDA(cudaFree(dWork_ev));
    CHECK_CUDA(cudaFree(devInfo));
    CHECK_CUDA(cudaDeviceSynchronize());
    
    return dW_out;
}