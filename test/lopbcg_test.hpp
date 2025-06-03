#include <vector>
#include <cmath>

#include "psd_projection/lopbcg.h"
#include "psd_projection/check.h"
#include "psd_projection/utils.h"


TEST(LOPBCG, Simple)
{
    size_t n = 7;
    size_t m = 2; // number of eigenpairs to compute
    size_t nn = n * n;

    // allocate device memory for A, V, D
    double *A, *V, *D;
    CHECK_CUDA(cudaMalloc(&A, nn * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&V, nn * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&D, n  * sizeof(double)));

    // initialize A with a simple symmetric matrix
    std::vector<double> h_A = {
        4, 1, 2, 3,
        1, 4, 3, 2,
        2, 3, 4, 1,
        3, 2, 1, 4
    };
    CHECK_CUDA(cudaMemcpy(A, h_A.data(), nn * sizeof(double), cudaMemcpyHostToDevice));

    // run the LOPBCG algorithm
    lopbcg(A, V, D, n, m);
}