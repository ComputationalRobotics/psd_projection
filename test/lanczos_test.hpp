#include <vector>
#include <cmath>

#include "psd_projection/lanczos.h"
#include "psd_projection/check.h"
#include "psd_projection/utils.h"

TEST(Lanczos, Simple)
{
    cublasHandle_t cublasH;
    cusolverDnHandle_t cusolverH ;
    CHECK_CUBLAS(cublasCreate(&cublasH));
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));

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

    // copy the matrix to device memory
    CHECK_CUDA(cudaMemcpy(A, h_A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    double two_norm = 7.0890; // actual two-norm of the matrix A

    // approximate the two-norm using the Lanczos method
    double lo, up;
    approximate_two_norm(cublasH, cusolverH, A, n, &lo, &up, 50, 1e-6);

    std::cout << "Actual two-norm: " << two_norm << std::endl;
    std::cout << "Approximate two-norm: [" << lo << ", " << up << "]" << std::endl;

    ASSERT_LE(lo, two_norm);
    ASSERT_GE(up, two_norm);
}