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

#include "psd_projection/eig_FP64_psd.h"
#include "psd_projection/composite_FP16.h"
#include "psd_projection/check.h"
#include "psd_projection/utils.h"
#include "psd_projection/lanczos.h"
#include "test_utils.hpp"

TEST(CompositeFP16, UniformAutoScaleDeflate1024)
{
    cusolverDnHandle_t solverH; cublasHandle_t cublasH;
    CHECK_CUSOLVER(cusolverDnCreate(&solverH));
    CHECK_CUBLAS(cublasCreate(&cublasH));

    size_t n = 1024;
    size_t k = 100;
    size_t nn = n*n;

    double *dA, *dA_psd;
    CHECK_CUDA(cudaMalloc(&dA, nn*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dA_psd, nn*sizeof(double)));

    // we generate a random matrix with values in [-10/n, 10/n]
    generateAndProject(n, dA, dA_psd, solverH, cublasH, 10.0); // cuSOLVER

    composite_FP16_auto_scale_deflate(cublasH, solverH, dA, n, nullptr, nullptr, k);

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
    ASSERT_LE(relative_err, 5e-3);

    // cleanup
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dA_psd));
    CHECK_CUDA(cudaFree(dDiff));
    CHECK_CUBLAS(cublasDestroy(cublasH));
    CHECK_CUSOLVER(cusolverDnDestroy(solverH));
}

TEST(CompositeFP16, Random)
{
    std::vector<int> ns = {100, 1000};
    for (int i = 0; i < ns.size(); i++) {
        int n = ns[i]; int nn = n * n;
        int m = n / 10;

        for (int restart = 0; restart < 5; ++restart) {
            // allocate device memory
            double *A, *A_tmp;
            CHECK_CUDA(cudaMalloc(&A,    nn * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&A_tmp, nn * sizeof(double)));

            cublasHandle_t cublasH;
            cusolverDnHandle_t solverH;
            CHECK_CUBLAS(cublasCreate(&cublasH));
            CHECK_CUSOLVER(cusolverDnCreate(&solverH));

            double one = 1.0;

            // generate a random matrix of size 10000
            std::vector<double> h_A(nn);
            for (auto &val : h_A) {
                val = static_cast<double>(rand()) / RAND_MAX;
            }

            // copy it to the device
            CHECK_CUDA(cudaMemcpy(A, h_A.data(), nn * sizeof(double), cudaMemcpyHostToDevice));

            // symmetrize A
            CHECK_CUDA(cudaMemcpy(A_tmp, A, nn * sizeof(double), cudaMemcpyDeviceToDevice));
            // A <- A^T + A
            CHECK_CUBLAS(cublasDgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &one, A_tmp, n, &one, A, n, A, n));

            // copy A to A_tmp
            CHECK_CUDA(cudaMemcpy(A_tmp, A, nn * sizeof(double), cudaMemcpyDeviceToDevice));

            // project with composite FP16
            composite_FP16_auto_scale_deflate(cublasH, solverH, A, n, nullptr, nullptr, m);

            // project with eig_FP64
            eig_FP64_psd(solverH, cublasH, A_tmp, n);

            // check if A and A_tmp are approximately equal
            double *dDiff; CHECK_CUDA(cudaMalloc(&dDiff, nn*sizeof(double)));
            double neg1 = -1.0;
            CHECK_CUBLAS(cublasDgeam(
                cublasH,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n,
                &one,  A_tmp, n,
                &neg1, A, n,
                dDiff,       n));
            double final_err = 0.0f;
            CHECK_CUBLAS(cublasDnrm2(cublasH, nn, dDiff, 1, &final_err));

            double A_tmp_norm = 0.0f;
            CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A_tmp, 1, &A_tmp_norm));
            double relative_err = final_err / A_tmp_norm;

            std::cout << "Relative error: " << std::scientific << std::setprecision(6) << relative_err << std::endl;
            std::cout << "Final error: " << std::scientific << std::setprecision(6) << final_err << std::endl;
            ASSERT_LE(relative_err, 1e-3);

            // cleanup
            CHECK_CUDA(cudaFree(A));
            CHECK_CUDA(cudaFree(A_tmp));
            CHECK_CUBLAS(cublasDestroy(cublasH));
            CHECK_CUSOLVER(cusolverDnDestroy(solverH));
        }
    }
}