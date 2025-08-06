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
#include "psd_projection/eig_FP32_psd.h"

double* eig_FP32_psd(cusolverDnHandle_t solverH, cublasHandle_t cublasH, double* dA, size_t n, bool return_eigenvalues) {
    size_t nn = n * n;
    float one_s = 1.0;
    float zero_s = 0.0;
    
    int *devInfo; CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));
    float *sA;
    CHECK_CUDA(cudaMalloc(&sA, nn*sizeof(float)));
    
    // convert dA from double to float
    convert_double_to_float(dA, sA, nn);

    float *sW; CHECK_CUDA(cudaMalloc(&sW, n*sizeof(float)));
    int lwork_ev = 0;
    CHECK_CUSOLVER(cusolverDnSsyevd_bufferSize(
        solverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
        n, sA, n, sW, &lwork_ev));
    float *sWork_ev; CHECK_CUDA(cudaMalloc(&sWork_ev, lwork_ev*sizeof(float)));
    CHECK_CUSOLVER(cusolverDnSsyevd(
        solverH,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        n, sA, n, sW,
        sWork_ev, lwork_ev, devInfo));
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> W_h(n);
    CHECK_CUDA(cudaMemcpy(W_h.data(), sW, n*sizeof(float), D2H));
    for (int i = 0; i < n; i++) {
        if (W_h[i] < 0)
            W_h[i] = 0;
    }

    // Copy eigenvectors from dA to dV
    float *sV; CHECK_CUDA(cudaMalloc(&sV, nn*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(sV, sA, nn*sizeof(float), D2D));

    // Scale columns of dV by W_h
    for (int i = 0; i < n; i++)
        CHECK_CUBLAS(cublasSscal(cublasH, n, &W_h[i], sV + i*n, 1));

    // Reconstruct A_psd = V * V^T
    float *sTmp; CHECK_CUDA(cudaMalloc(&sTmp, nn*sizeof(float)));
    CHECK_CUBLAS(cublasGemmEx(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
        n, n, n,
        &one_s,
        sV, CUDA_R_32F, n,
        sA, CUDA_R_32F, n,
        &zero_s,
        sTmp, CUDA_R_32F, n,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaMemcpy(sA, sTmp, nn*sizeof(float), D2D));

    convert_float_to_double(sA, dA, nn);

    // Cleanup
    CHECK_CUDA(cudaFree(sWork_ev));
    CHECK_CUDA(cudaFree(sA));
    CHECK_CUDA(cudaFree(sTmp));
    CHECK_CUDA(cudaFree(sV));
    CHECK_CUDA(cudaFree(devInfo));
    CHECK_CUDA(cudaDeviceSynchronize());

    if (!return_eigenvalues) {
        CHECK_CUDA(cudaFree(sW));
        return nullptr;
    } else {
        double *dW; CHECK_CUDA(cudaMalloc(&dW, n*sizeof(double)));
        convert_float_to_double(sW, dW, n);
        CHECK_CUDA(cudaFree(sW));
        return dW;
    }
}