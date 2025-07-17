#include <vector>
#include <cmath>

#include "psd_projection/lopbcg.h"
#include "psd_projection/check.h"
#include "psd_projection/utils.h"


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

    // run the LOPBCG algorithm
    lopbcg(A, V, D, n, m, 100, 1e-8, true);

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
}