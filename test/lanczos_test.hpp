#include <vector>
#include <cmath>

#include "psd_projection/lanczos.h"
#include "psd_projection/check.h"
#include "psd_projection/utils.h"

TEST(Lanczos, Simple)
{
    size_t n = 5;

    // create a simple symmetric matrix A
    double *A;
    CHECK_CUDA(cudaMalloc(&A, n * n * sizeof(double)));
    std::vector<double> h_A = {
        4.0, 1.0, 2.0, 0.0, 0.0,
        1.0, 3.0, 0.0, 1.0, 0.0,
        2.0, 0.0, 5.0, 1.0, 1.0,
        0.0, 1.0, 1.0, 2.0, 1.0,
        0.0, 0.0, 1.0, 1.0, 3.0
    };

    // compute its two-norm for reference
    double two_norm = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        two_norm += h_A[i] * h_A[i];
    }
    two_norm = std::sqrt(two_norm);

    // copy the matrix to device memory
    CHECK_CUDA(cudaMemcpy(A, h_A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    // approximate the two-norm using the Lanczos method
    double lo, up;
    approximate_two_norm(A, n, &lo, &up, 50, 1e-6);

    std::cout << "Actual two-norm: " << two_norm << std::endl;
    std::cout << "Approximate two-norm: [" << lo << ", " << up << "]" << std::endl;
}