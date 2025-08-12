#include <vector>
#include <cmath>

#include "psd_projection/lopbcg.h"
#include "psd_projection/check.h"
#include "psd_projection/utils.h"
#include "psd_projection/eig_FP64_psd.h"


TEST(LOPBCG, Simple)
{
    size_t n = 6;
    size_t m = 2; // number of eigenpairs to compute
    size_t nn = n * n;

    // allocate device memory for A, V, D
    double *A, *V, *D;
    CHECK_CUDA(cudaMalloc(&A,    nn * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&V, n * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&D,     m * sizeof(double)));

    // initialize A with a simple symmetric matrix
    std::vector<double> h_A = {
        4.0, 1.0, 2.0, 0.0, 0.0, 0.0,
        1.0, 3.0, 0.0, 1.0, 0.0, 0.0,
        2.0, 0.0, 5.0, 1.0, 1.0, 0.0,
        0.0, 1.0, 1.0, 2.0, 1.0, 1.0,
        0.0, 0.0, 1.0, 1.0, 3.0, 2.0,
        0.0, 0.0, 0.0, 1.0, 2.0, 4.0
    };
    CHECK_CUDA(cudaMemcpy(A, h_A.data(), nn * sizeof(double), cudaMemcpyHostToDevice));

    cublasHandle_t cublasH;
    cusolverDnHandle_t solverH;
    CHECK_CUBLAS(cublasCreate(&cublasH));
    CHECK_CUSOLVER(cusolverDnCreate(&solverH));

    // run the LOPBCG algorithm
    lopbcg(cublasH, solverH, A, V, D, n, m, 100, 1e-8, true);

    // check if the eigenvalues are close to the expected values
    std::vector<double> expected_eigenvalues = {7.3191, 5.6639}; // expected eigenvalues for this matrix
    std::vector<double> h_D(m);
    CHECK_CUDA(cudaMemcpy(h_D.data(), D, m * sizeof(double), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < m; ++i) {
        EXPECT_NEAR(h_D[i], expected_eigenvalues[i], 1e-4);
    }

    // cleanup
    CHECK_CUDA(cudaFree(A));
    CHECK_CUDA(cudaFree(V));
    CHECK_CUDA(cudaFree(D));
    CHECK_CUBLAS(cublasDestroy(cublasH));
    CHECK_CUSOLVER(cusolverDnDestroy(solverH));
}

TEST(LOPBCG, Random)
{
    std::vector<int> ns = {100, 1000};
    for (int i = 0; i < ns.size(); i++) {
        int n = ns[i]; int nn = n * n;
        int m = n / 10;

        for (int restart = 0; restart < 5; ++restart) {
            // allocate device memory
            double *A, *V, *D, *A_tmp;
            CHECK_CUDA(cudaMalloc(&A,    nn * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&V, n * m * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&D,     m * sizeof(double)));
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


            // run the LOPBCG algorithm
            lopbcg(cublasH, solverH, A, V, D, n, m);

            // check if the eigenvalues are close to the expected values
            std::vector<double> expected_eigenvalues(m); // expected eigenvalues for this matrix
            
            // run cuSOLVER to get the eigenvalues
            double *expected = eig_FP64_psd(solverH, cublasH, A, n, true);

            std::vector<double> h_D(m);
            std::vector<double> h_expected(n);
            
            CHECK_CUDA(cudaMemcpy(h_D.data(), D, m * sizeof(double), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_expected.data(), expected, n * sizeof(double), cudaMemcpyDeviceToHost));

            for (size_t i = 0; i < m; ++i) {
                EXPECT_NEAR(h_D[i], h_expected[n-i-1], 1e-8);
            }

            // cleanup
            CHECK_CUDA(cudaFree(A));
            CHECK_CUDA(cudaFree(A_tmp));
            CHECK_CUDA(cudaFree(V));
            CHECK_CUDA(cudaFree(D));
            CHECK_CUDA(cudaFree(expected));
            CHECK_CUBLAS(cublasDestroy(cublasH));
            CHECK_CUSOLVER(cusolverDnDestroy(solverH));
        }
    }
}