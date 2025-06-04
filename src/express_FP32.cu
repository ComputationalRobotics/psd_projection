#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <cusolverDn.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>
#include <iostream>

#include "psd_projection/express_FP32.h"
#include "psd_projection/check.h"
#include "psd_projection/utils.h"

void express_FP32(
    cublasHandle_t cublasH,
    double* mat,
    const int n,
    const int mat_offset
) {
    const int nn = n * n;

    /* Allocations */
    // device memory
    float *A, *A2, *A3, *A5, *I, *W;
    CHECK_CUDA( cudaMalloc(&A,  nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&A2, nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&A3, nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&A5, nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&I,  nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&W,  nn * sizeof(float)) );

    // useful constants
    const float half       =  0.5f;
    const float minus_half = -0.5f;
    const float one        =  1.0f;
    const float one_n_half =  1.5f;
    const float zero       =  0.0f;

    // build identity I on device
    std::vector<float> I_h(nn, 0.0f);
    for (int i = 0; i < n; i++) I_h[i*n + i] = 1.0f;
    CHECK_CUDA( cudaMemcpy(I, I_h.data(), nn * sizeof(float), H2D) );

    /* Convert the initial matrix*/
    // copy the double matrix back to the host
    std::vector<double> A_h_d(nn);
    CHECK_CUDA( cudaMemcpy(A_h_d.data(), mat + mat_offset, nn * sizeof(double), D2H) );

    // convert the host matrix to float
    std::vector<float> A_h(nn);
    for (int i = 0; i < nn; i++) {
        A_h[i] = static_cast<float>(A_h_d[i]);
    }

    // copy the float host matrix to the device
    CHECK_CUDA( cudaMemcpy(A, A_h.data(), nn * sizeof(float), H2D) );
    CHECK_CUDA( cudaMemcpy(W, A_h.data(), nn * sizeof(float), H2D) );

    /* Coefficients */
    std::vector<std::vector<float>> coeff = {
        {8.4724206924, -24.5001735687, 17.7268180847},
        {4.2052841187, -3.0549299717, 0.5567536354},
        {4.0443077087, -2.9473149776, 0.5449726582},
        {3.5078327656, -2.5842490196, 0.5067413449},
        {2.5075511932, -1.8485442400, 0.4358045161}
    };

    /* Approximation of the step function */
    for (int i = 0; i < coeff.size(); i++) {
        const float a = coeff[i][0];
        const float b = coeff[i][1];
        const float c = coeff[i][2];

        /* Compute the powers of A*/
        // A2 = A * A
        CHECK_CUBLAS( cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A, n, A, n, &zero, A2, n) );

        // A3 = A2 * A
        CHECK_CUBLAS( cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A2, n, A, n, &zero, A3, n) );

        // A5 = A3 * A2
        CHECK_CUBLAS( cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A3, n, A2, n, &zero, A5, n) );

        /* Compute A = a * A + b * A3 + c * A5 */
        // A = a * A
        CHECK_CUBLAS( cublasSscal(cublasH, nn, &a, A, 1) );
        // A = b * A3 + A
        CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &b, A3, 1, A, 1) );
        // A = c * A5 + A
        CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &c, A5, 1, A, 1) );
    }

    /* Smoothing function */
    for (int i =0; i < 3; i++) {
        // A2 = A * A
        CHECK_CUBLAS( cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A, n, A, n, &zero, A2, n) );

        // A3 = A2 * A
        CHECK_CUBLAS( cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A2, n, A, n, &zero, A3, n) );

        /* Compute A = 1.5 * A - 0.5 * A3 */
        // A = 1.5 * A
        CHECK_CUBLAS( cublasSscal(cublasH, nn, &one_n_half, A, 1) );
        // A = -0.5 * A3 + A
        CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &minus_half, A3, 1, A, 1) );
    }

    /* Compute A = (I + A)/2 */
    // A = 1 * I + A
    CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &one, A, 1, I, 1) );
    // A = 0.5 * A
    CHECK_CUBLAS( cublasSscal(cublasH, nn, &half, I, 1) );

    /* Multiply the original matrix by A coefficient-wise */
    // W = A .* W
    CHECK_CUBLAS( cublasSdgmm(cublasH, CUBLAS_SIDE_LEFT, n, n, W, n, A, 1, W, n) );

    /* Copy the result back to mat */
    std::vector<float> A_h_f(nn);
    CHECK_CUDA( cudaMemcpy(A_h_f.data(), A, nn * sizeof(float), D2H) );
    for (size_t i = 0; i < nn; i++) {
        A_h_d[i] = static_cast<double>(A_h_f[i]);
    }
    CHECK_CUDA( cudaMemcpy(mat + mat_offset, A_h_d.data(), nn * sizeof(double), H2D) );
    CHECK_CUDA( cudaDeviceSynchronize() );

    /* Free device memory */
    CHECK_CUDA( cudaFree(A) );
    CHECK_CUDA( cudaFree(A2) );
    CHECK_CUDA( cudaFree(A3) );
    CHECK_CUDA( cudaFree(A5) );
    CHECK_CUDA( cudaFree(I) );
    CHECK_CUDA( cudaFree(W) );
    CHECK_CUBLAS( cublasDestroy(cublasH) );
}