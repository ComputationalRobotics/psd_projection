#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <cmath>
#include <iomanip> 
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <cusolverDn.h>

#include "psd_projection/express_FP32.h"
#include "psd_projection/check.h"
#include "psd_projection/utils.h"
#include "psd_projection/lanczos.h"
#include "test_utils.hpp"

TEST(ExpressFP32, UniformScaled)
{
    cusolverDnHandle_t solverH; cublasHandle_t cublasH;
    CHECK_CUSOLVER(cusolverDnCreate(&solverH));
    CHECK_CUBLAS(cublasCreate(&cublasH));
    CHECK_CUBLAS(cublasSetMathMode(cublasH, CUBLAS_TENSOR_OP_MATH));

    size_t n = 1024;
    size_t nn = n*n;

    double *dA, *dA_psd;
    CHECK_CUDA(cudaMalloc(&dA, nn*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dA_psd, nn*sizeof(double)));

    generateAndProject(n, dA, dA_psd, solverH, cublasH); // cuSOLVER
    express_FP32(cublasH, dA, n, 0);

    // check if dA and dA_psd are approximately equal
    double *dDiff; CHECK_CUDA(cudaMalloc(&dDiff, nn*sizeof(double)));
    double one = 1.0, neg1 = -1.0;
    CHECK_CUBLAS(cublasDgeam(
        cublasH,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n,
        &one,  dA_psd, n,
        &neg1, dA, n,
        dDiff,       n));
    double final_err = 0.0f;
    CHECK_CUBLAS(cublasDnrm2(cublasH, nn, dDiff, 1, &final_err));

    ASSERT_LE(final_err, 1e-2) << "Final error: " << final_err;

    // cleanup
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dA_psd));
    CHECK_CUDA(cudaFree(dDiff));
    CHECK_CUBLAS(cublasDestroy(cublasH));
    CHECK_CUSOLVER(cusolverDnDestroy(solverH));
}