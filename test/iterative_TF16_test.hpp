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

#include "psd_projection/iterative_TF16.h"
#include "psd_projection/check.h"
#include "psd_projection/utils.h"
#include "psd_projection/lanczos.h"
#include "test_utils.hpp"

TEST(IterativeTF16, UniformScaled)
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
    projection_TF16(cublasH, dA, n, 0, 100); // iterative TF16 projection

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

TEST(IterativeTF16, UniformNonScaled1024)
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

    // we generate a random matrix with values in [-10/n, 10/n]
    generateAndProject(n, dA, dA_psd, solverH, cublasH, 10.0); // cuSOLVER


    /* Rescale the matrix */
    // compute an approximation of the spectral 2-norm using Lanczos method
    double lo, up;
    approximate_two_norm(
        cublasH, solverH, dA, n, &lo, &up
    );

    // scale to have eigenvalues in [-1, 1]
    const double scale = up > 0.0f ? up : 1.0f;
    // const double scale = 1.0f;
    const double inv_scale = 1.0f/scale;
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, dA, 1) );

    /* Project */
    projection_TF16(cublasH, dA, n, 0, 100); // iterative TF16 projection

    // scale back dA to original range
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA, 1) );

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

    double dA_psd_norm = 0.0f;
    CHECK_CUBLAS(cublasDnrm2(cublasH, nn, dA_psd, 1, &dA_psd_norm));
    double relative_err = final_err / dA_psd_norm;

    std::cout << "Relative error: " << std::scientific << std::setprecision(6) << relative_err << std::endl;
    std::cout << "Final error: " << std::scientific << std::setprecision(6) << final_err << std::endl;
    ASSERT_LE(final_err, 1e-1) << "Final error: " << final_err;

    // cleanup
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dA_psd));
    CHECK_CUDA(cudaFree(dDiff));
    CHECK_CUBLAS(cublasDestroy(cublasH));
    CHECK_CUSOLVER(cusolverDnDestroy(solverH));
}

TEST(IterativeTF16, UniformNonScaled10000)
{
    GTEST_SKIP();
    cusolverDnHandle_t solverH; cublasHandle_t cublasH;
    CHECK_CUSOLVER(cusolverDnCreate(&solverH));
    CHECK_CUBLAS(cublasCreate(&cublasH));
    CHECK_CUBLAS(cublasSetMathMode(cublasH, CUBLAS_TENSOR_OP_MATH));

    size_t n = 10000;
    size_t nn = n*n;

    double *dA, *dA_psd;
    CHECK_CUDA(cudaMalloc(&dA, nn*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dA_psd, nn*sizeof(double)));

    // we generate a random matrix with values in [-100/n, 100/n]
    generateAndProject(n, dA, dA_psd, solverH, cublasH, 100.0); // cuSOLVER


    auto start = std::chrono::high_resolution_clock::now();
    /* Rescale the matrix */
    // compute an approximation of the spectral 2-norm using Lanczos method
    double lo, up;
    approximate_two_norm(
        cublasH, solverH, dA, n, &lo, &up
    );

    // scale to have eigenvalues in [-1, 1]
    const double scale = up > 0.0f ? up : 1.0f;
    // const double scale = 1.0f;
    const double inv_scale = 1.0f/scale;
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, dA, 1) );

    /* Project */
    projection_TF16(cublasH, dA, n, 0, 100); // iterative TF16 projection

    // scale back dA to original range
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA, 1) );

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time (our method): " 
              << std::fixed << std::setprecision(3) 
              << elapsed.count() << " seconds" << std::endl;

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

    double dA_psd_norm = 0.0f;
    CHECK_CUBLAS(cublasDnrm2(cublasH, nn, dA_psd, 1, &dA_psd_norm));

    double dA_norm = 0.0f;
    CHECK_CUBLAS(cublasDnrm2(cublasH, nn, dA, 1, &dA_norm));

    std::cout << "Final error: " << final_err << std::endl;
    double relative_err = final_err / dA_psd_norm;
    std::cout << "Relative error: " << std::scientific << std::setprecision(6) << relative_err << std::endl;
    ASSERT_LE(relative_err, 1e-3);

    // cleanup
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dA_psd));
    CHECK_CUDA(cudaFree(dDiff));
    CHECK_CUBLAS(cublasDestroy(cublasH));
    CHECK_CUSOLVER(cusolverDnDestroy(solverH));
}