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

#include "psd_projection/express_FP32_Lt.h"
#include "psd_projection/check.h"
#include "psd_projection/utils.h"
#include "psd_projection/lanczos.h"
#include "test_utils.hpp"

void benchmark_express(
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

    std::chrono::duration<double> time_cusolver(0.0), time_express(0.0), time_express_Lt(0.0);
    double error_express = 0.0, error_express_Lt = 0.0;
    double relative_error_express = 0.0, relative_error_express_Lt = 0.0;

    double one = 1.0, neg1 = -1.0;

    for (int i = 0; i < nb_mat; i++) {
        /* Generate a random matrix and project it using cuSOLVER */
        // TODO: check generation code
        std::chrono::duration<double> time;
        generateAndProject(n, dA, dA_psd, solverH, cublasH, 10.0, &time);
        time_cusolver += time;

        // copy dA to dA_Lt
        CHECK_CUDA(cudaMemcpy(dA_Lt, dA, nn*sizeof(double), cudaMemcpyDeviceToDevice));

        /* Solve using express_FP32 */
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
        express_FP32(cublasH, dA, n, 0);

        // scale back dA to original range
        CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA, 1) );

        time_express += std::chrono::high_resolution_clock::now() - start;

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
        error_express += final_err;
        relative_error_express += relative_err;


        /* Solve using express_FP32_Lt */
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
        express_FP32_Lt(cublasH, cublasLtH, dA_Lt, n, 0);

        // scale back dA_Lt to original range
        CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_Lt, 1) );

        time_express_Lt += std::chrono::high_resolution_clock::now() - start;

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
        error_express_Lt += final_err_Lt;
        relative_error_express_Lt += relative_err_Lt;
    }

    /* Print info */
    std::cout << "Benchmarking with " << nb_mat << " matrices of size " << n << "x" << n << " and radius " << std::fixed << radius << std::endl;
    std::cout << std::endl;
    std::cout << "  Time (cuSOLVER):             " << std::scientific << std::setprecision(4) << time_cusolver.count() / (double)nb_mat << "s" << std::endl;
    std::cout << "  Time (Express):              " << std::scientific << std::setprecision(4) << time_express.count() / (double)nb_mat << "s" << std::endl;
    std::cout << "  Time (Express Lt):           " << std::scientific << std::setprecision(4) << time_express_Lt.count() / (double)nb_mat << "s" << std::endl;
    std::cout << std::endl;
    std::cout << "  Error (Express):             " << std::scientific << std::setprecision(4) << error_express / (double)nb_mat << std::endl;
    std::cout << "  Error (Express Lt):          " << std::scientific << std::setprecision(4) << error_express_Lt / (double)nb_mat << std::endl;
    std::cout << "  Relative Error (Express):    " << std::scientific << std::setprecision(4) << relative_error_express / (double)nb_mat << std::endl;
    std::cout << "  Relative Error (Express Lt): " << std::scientific << std::setprecision(4) << relative_error_express_Lt / (double)nb_mat << std::endl;
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
    benchmark_express(1.0, 1000, 10);
    benchmark_express(10.0, 1000, 10);
    benchmark_express(1.0, 5000, 10);
    benchmark_express(10.0, 5000, 10);
    benchmark_express(10.0, 5000, 10, 10, 1e-3);
}