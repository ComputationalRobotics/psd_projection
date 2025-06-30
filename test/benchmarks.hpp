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
#include <cublasLt.h>

#include "psd_projection/composite_FP32_Lt.h"
#include "psd_projection/sample_cublasLt_LtSgemm.h"
#include "psd_projection/check.h"
#include "psd_projection/utils.h"
#include "psd_projection/lanczos.h"
#include "test_utils.hpp"

void benchmark_composite(
    double radius = 1.0,
    size_t n = 1024,
    size_t nb_mat = 1,
    int lanczos_max_iter = 50,
    double lanczos_tol = 1e-6
) {
    cusolverDnHandle_t solverH;
    cublasHandle_t cublasH;
    cublasLtHandle_t cublasLtH;
    CHECK_CUSOLVER(cusolverDnCreate(&solverH));
    CHECK_CUBLAS(cublasCreate(&cublasH));
    CHECK_CUBLAS( cublasLtCreate(&cublasLtH) );

    size_t nn = n*n;

    double *dA, *dA_Lt, *dA_psd, *dDiff;
    CHECK_CUDA(cudaMalloc(&dA,     nn*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dA_Lt,  nn*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dA_psd, nn*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dDiff,  nn*sizeof(double)));

    std::chrono::duration<double> time_cusolver(0.0), time_composite(0.0), time_composite_Lt(0.0);
    double error_composite = 0.0, error_composite_Lt = 0.0;
    double relative_error_composite = 0.0, relative_error_composite_Lt = 0.0;

    double one = 1.0, neg1 = -1.0;

    for (int i = 0; i < nb_mat; i++) {
        /* Generate a random matrix and project it using cuSOLVER */
        // TODO: check generation code
        std::chrono::duration<double> time;
        generateAndProject(n, dA, dA_psd, solverH, cublasH, 10.0, &time);
        time_cusolver += time;

        // copy dA to dA_Lt
        CHECK_CUDA(cudaMemcpy(dA_Lt, dA, nn*sizeof(double), cudaMemcpyDeviceToDevice));

        /* Solve using composite_FP32 */
        auto start = std::chrono::high_resolution_clock::now();
        double lo, up;
        approximate_two_norm(
            cublasH, solverH, dA, n, &lo, &up, lanczos_max_iter, lanczos_tol
        );

        // scale to have eigenvalues in [-1, 1]
        double scale = up > 0.0f ? up : 1.0f;
        // const double scale = 1.0f;
        double inv_scale = 1.0f/scale;
        CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, dA, 1) );

        // project
        composite_FP32(cublasH, dA, n, 0);

        // scale back dA to original range
        CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA, 1) );

        time_composite += std::chrono::high_resolution_clock::now() - start;

        // compute error
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
        error_composite += final_err;
        relative_error_composite += relative_err;


        /* Solve using composite_FP32_Lt */
        start = std::chrono::high_resolution_clock::now();
        approximate_two_norm(
            cublasH, solverH, dA_Lt, n, &lo, &up, lanczos_max_iter, lanczos_tol
        );

        // scale to have eigenvalues in [-1, 1]
        scale = up > 0.0f ? up : 1.0f;
        // const double scale = 1.0f;
        inv_scale = 1.0f/scale;
        CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, dA_Lt, 1) );

        // project
        composite_FP32_Lt(cublasH, cublasLtH, dA_Lt, n, 0);

        // scale back dA_Lt to original range
        CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_Lt, 1) );

        time_composite_Lt += std::chrono::high_resolution_clock::now() - start;

        // compute error
        CHECK_CUBLAS(cublasDgeam(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n,
            &one,  dA_psd, n,
            &neg1, dA_Lt, n,
            dDiff,       n));
        double final_err_Lt = 0.0f;
        CHECK_CUBLAS(cublasDnrm2(cublasH, nn, dDiff, 1, &final_err_Lt));
        double dA_psd_norm_Lt = 0.0f;
        CHECK_CUBLAS(cublasDnrm2(cublasH, nn, dA_psd, 1, &dA_psd_norm_Lt));
        double relative_err_Lt = final_err_Lt / dA_psd_norm_Lt;
        error_composite_Lt += final_err_Lt;
        relative_error_composite_Lt += relative_err_Lt;
    }

    /* Print info */
    std::cout << "Benchmarking with " << nb_mat << " matrices of size " << n << "x" << n << " and radius " << std::fixed << radius << std::endl;
    std::cout << std::endl;
    std::cout << "  Time (cuSOLVER):             " << std::scientific << std::setprecision(4) << time_cusolver.count() / (double)nb_mat << "s" << std::endl;
    std::cout << "  Time (composite):              " << std::scientific << std::setprecision(4) << time_composite.count() / (double)nb_mat << "s" << std::endl;
    std::cout << "  Time (composite Lt):           " << std::scientific << std::setprecision(4) << time_composite_Lt.count() / (double)nb_mat << "s" << std::endl;
    std::cout << std::endl;
    std::cout << "  Error (composite):             " << std::scientific << std::setprecision(4) << error_composite / (double)nb_mat << std::endl;
    std::cout << "  Error (composite Lt):          " << std::scientific << std::setprecision(4) << error_composite_Lt / (double)nb_mat << std::endl;
    std::cout << "  Relative Error (composite):    " << std::scientific << std::setprecision(4) << relative_error_composite / (double)nb_mat << std::endl;
    std::cout << "  Relative Error (composite Lt): " << std::scientific << std::setprecision(4) << relative_error_composite_Lt / (double)nb_mat << std::endl;
    std::cout << std::endl;

    /* Cleanup */
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dA_Lt));
    CHECK_CUDA(cudaFree(dA_psd));
    CHECK_CUDA(cudaFree(dDiff));
    CHECK_CUBLAS(cublasDestroy(cublasH));
    CHECK_CUSOLVER(cusolverDnDestroy(solverH));
    CHECK_CUBLAS(cublasLtDestroy(cublasLtH));
}

TEST(Benchmarks, Uniform)
{
    benchmark_composite(1.0, 1000, 10);
    benchmark_composite(10.0, 1000, 10);
    // benchmark_composite(1.0, 5000, 10);
    // benchmark_composite(10.0, 5000, 10);
    // benchmark_composite(10.0, 5000, 10, 10, 1e-3);
}

TEST(Benchmarks, MatMul)
{
    GTEST_SKIP(); // only activate for benchmarks

    cusolverDnHandle_t solverH;
    cublasHandle_t cublasH;
    cublasLtHandle_t cublasLtH;
    CHECK_CUSOLVER(cusolverDnCreate(&solverH));
    CHECK_CUBLAS(cublasCreate(&cublasH));
    CHECK_CUBLAS( cublasLtCreate(&cublasLtH) );

    // create a workspace for cublasLt
    size_t workspace_size = 32 * 1024 * 1024;
    void* workspace;
    CHECK_CUDA( cudaMalloc(&workspace, workspace_size) );
    
    size_t n = 10000;
    size_t nn = n*n;

    // create two random matrices
    float *A, *B, *AB;
    CHECK_CUDA(cudaMalloc(&A, nn*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&B, nn*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&AB, nn*sizeof(float)));

    // generate random matrices
    std::vector<float> A_h(nn), B_h(nn);
    srand(1);
    for(size_t i = 0; i < nn; i++) {
        A_h[i] = rand() / (float)RAND_MAX - 0.5;
        B_h[i] = rand() / (float)RAND_MAX - 0.5;
    }
    CHECK_CUDA(cudaMemcpy(A, A_h.data(), nn*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B, B_h.data(), nn*sizeof(float), cudaMemcpyHostToDevice));
    const float one = 1.0, zero = 0.0;

    // warmup
    for (int i = 0; i < 100; i++) {
        CHECK_CUBLAS(cublasSgemm(
            cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one, A, n,
            B, n,
            &zero, AB, n
        ));
    }
    
    // compute AB using cuBLAS
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        CHECK_CUBLAS(cublasSgemm(
            cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one, A, n,
            B, n,
            &zero, AB, n
        ));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "cuBLAS time:   " 
              << std::scientific << milliseconds
              << " ms" << std::endl;

    // compute ABLt using LtSgemm
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        LtSgemm(cublasLtH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A, n, B, n, &zero, AB, n, workspace, workspace_size);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "cublasLt time: " 
              << std::scientific << milliseconds 
              << " ms" << std::endl;

    // free
    CHECK_CUDA(cudaFree(A));
    CHECK_CUDA(cudaFree(B));
    CHECK_CUDA(cudaFree(AB));
}