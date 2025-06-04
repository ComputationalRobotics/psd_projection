#include <vector>
#include <cmath>

#include "psd_projection/lanczos.h"
#include "psd_projection/check.h"
#include "psd_projection/utils.h"

TEST(Lanczos, Simple)
{
    size_t n = 5;
    double *A;
    CHECK_CUDA(cudaMalloc(&A, n * n * sizeof(double)));

    std::vector<double> h_A = {
        4.0, 1.0, 2.0, 0.0, 0.0,
        1.0, 3.0, 0.0, 1.0, 0.0,
        2.0, 0.0, 5.0, 1.0, 1.0,
        0.0, 1.0, 1.0, 2.0, 1.0,
        0.0, 0.0, 1.0, 1.0, 3.0
    };
    CHECK_CUDA(cudaMemcpy(A, h_A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    double lo, up;
    approximate_two_norm(A, n, &lo, &up, 50, 1e-6);
}