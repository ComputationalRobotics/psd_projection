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

TEST(ExpressFP32, Deterministic)
{
    cusolverDnHandle_t solverH; cublasHandle_t cublasH;
    CHECK_CUSOLVER(cusolverDnCreate(&solverH));
    CHECK_CUBLAS(cublasCreate(&cublasH));

    size_t n = 3;
    size_t nn = n*n;

    std::vector<double> A = {
        1.0, 2.0, 3.0,
        2.0, 5.0, 6.0,
        3.0, 6.0, 9.0
    };
    // divide A by 10 to ensure convergence
    for (size_t i = 0; i < nn; ++i)
        A[i] /= 15.0;

    double *dA;
    CHECK_CUDA(cudaMalloc(&dA, nn*sizeof(double)));
    CHECK_CUDA(cudaMemcpy(dA, A.data(), nn*sizeof(double), cudaMemcpyHostToDevice));

    auto start = std::chrono::high_resolution_clock::now();
    express_FP32(cublasH, dA, n, 0);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time (express): " << std::fixed << std::setprecision(6) << elapsed.count() << " seconds" << std::endl;

    // check if dA is approximately equal to the expected PSD matrix
    std::vector<double> expected = {
        0.0667, 0.1333, 0.2000,
        0.1333, 0.3333, 0.4000,
        0.2000, 0.4000, 0.6000
    };
    std::vector<double> dA_h(nn);
    CHECK_CUDA(cudaMemcpy(dA_h.data(), dA, nn*sizeof(double), cudaMemcpyDeviceToHost));

    // // print dA_h for debugging
    // std::cout << "dA_h: " << std::endl;
    // for (size_t i = 0; i < n; ++i) {
    //     for (size_t j = 0; j < n; ++j) {
    //         std::cout << dA_h[i*n + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // // print expected for debugging
    // std::cout << "expected: " << std::endl;
    // for (size_t i = 0; i < n; ++i) {
    //     for (size_t j = 0; j < n; ++j) {
    //         std::cout << expected[i*n + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
    
    for (size_t i = 0; i < nn; ++i) {
        ASSERT_NEAR(dA_h[i], expected[i], 1e-3);
    }
}

TEST(ExpressFP32, UniformScaled)
{
    cusolverDnHandle_t solverH; cublasHandle_t cublasH;
    CHECK_CUSOLVER(cusolverDnCreate(&solverH));
    CHECK_CUBLAS(cublasCreate(&cublasH));

    size_t n = 1000;
    size_t nn = n*n;

    double *dA, *dA_psd;
    CHECK_CUDA(cudaMalloc(&dA, nn*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dA_psd, nn*sizeof(double)));
    
    generateAndProject(n, dA, dA_psd, solverH, cublasH); // cuSOLVER

    auto start = std::chrono::high_resolution_clock::now();
    express_FP32(cublasH, dA, n, 0);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time (express): " << std::fixed << std::setprecision(6) << elapsed.count() << " seconds" << std::endl;

    // check if dA and dA_psd are approximately equal
    double *dDiff;
    CHECK_CUDA(cudaMalloc(&dDiff, nn*sizeof(double)));
    double one = 1.0, neg1 = -1.0;
    CHECK_CUBLAS(cublasDgeam(
        cublasH,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n,
        &one,  dA_psd, n,
        &neg1, dA, n,
        dDiff, n));
    double final_err = 0.0f;
    CHECK_CUBLAS(cublasDnrm2(cublasH, nn, dDiff, 1, &final_err));

    double dA_psd_norm = 0.0f;
    CHECK_CUBLAS(cublasDnrm2(cublasH, nn, dA_psd, 1, &dA_psd_norm));
    double relative_err = final_err / dA_psd_norm;

    std::cout << "Relative error: " << std::scientific << std::setprecision(6) << relative_err << std::endl;
    std::cout << "Final error: " << std::scientific << std::setprecision(6) << final_err << std::endl;
    ASSERT_LE(relative_err, 1e-3);

    // cleanup
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dA_psd));
    CHECK_CUDA(cudaFree(dDiff));
    CHECK_CUBLAS(cublasDestroy(cublasH));
    CHECK_CUSOLVER(cusolverDnDestroy(solverH));
}

TEST(ExpressFP32, UniformNonScaled1024)
{
    cusolverDnHandle_t solverH; cublasHandle_t cublasH;
    CHECK_CUSOLVER(cusolverDnCreate(&solverH));
    CHECK_CUBLAS(cublasCreate(&cublasH));

    size_t n = 1024;
    size_t nn = n*n;

    double *dA, *dA_psd;
    CHECK_CUDA(cudaMalloc(&dA, nn*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dA_psd, nn*sizeof(double)));

    // we generate a random matrix with values in [-10/n, 10/n]
    generateAndProject(n, dA, dA_psd, solverH, cublasH, 10.0); // cuSOLVER


    /* Rescale the matrix */
    // compute an approximation of the spectral 2-norm using Lanczos method
    auto start = std::chrono::high_resolution_clock::now();
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
    express_FP32(cublasH, dA, n, 0);

    // scale back dA to original range
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA, 1) );

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time (express): " << std::fixed << std::setprecision(6) << elapsed.count() << " seconds" << std::endl;

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

TEST(ExpressFP32, UniformAutoScaled1024)
{
    cusolverDnHandle_t solverH; cublasHandle_t cublasH;
    CHECK_CUSOLVER(cusolverDnCreate(&solverH));
    CHECK_CUBLAS(cublasCreate(&cublasH));

    size_t n = 1024;
    size_t nn = n*n;

    double *dA, *dA_psd;
    CHECK_CUDA(cudaMalloc(&dA, nn*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dA_psd, nn*sizeof(double)));

    // we generate a random matrix with values in [-10/n, 10/n]
    generateAndProject(n, dA, dA_psd, solverH, cublasH, 10.0); // cuSOLVER


    express_FP32_auto_scale(cublasH, solverH, dA, n, 0);

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