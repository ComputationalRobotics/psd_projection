#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <cusolverDn.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>
#include <iostream>

#include "psd_projection/composite_FP32.h"
#include "psd_projection/lanczos.h"
#include "psd_projection/check.h"
#include "psd_projection/utils.h"

void composite_FP32(
    cublasHandle_t cublasH,
    double* mat,
    const int n,
    const bool verbose
) {
    const int nn = n * n;

    /* Allocations */
    // device memory
    float *A, *A2, *A3;
    CHECK_CUDA( cudaMalloc(&A,  nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&A2, nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&A3, nn * sizeof(float)) );

    // useful constants
    const float half       =  0.5f;
    const float minus_half = -0.5f;
    const float one        =  1.0f;
    const float one_n_half =  1.5f;
    const float zero       =  0.0f;

    /* Convert the initial matrix*/
    convert_double_to_float(mat, A, nn);

    /* Coefficients */
    // const std::vector<std::vector<float>> coeff = { 
    //     { 8.3885353390, -23.7796270883, 16.8664591580 }, 
    //     { 4.1636476423, -2.9650849331, 0.5297319805 }, 
    //     { 4.0042650581, -2.8606348801, 0.5185227850 }, 
    //     { 3.4731017481, -2.5082466382, 0.4821470022 }, 
    //     { 2.4827239537, -1.7941788274, 0.4146530436 }, 
    // }; const size_t smoothing_steps = 3;
    const std::vector<std::vector<float>> coeff = { 
        { 8.5018632351, -24.6330845767, 17.8466614026 },
        { 4.2394319792, -3.0803745982, 0.5596805290 },
        { 4.2371780379, -3.0779047407, 0.5594995022 },
        { 4.1553447421, -3.0255808203, 0.5534594007 },
        { 3.8719053120, -2.8289969308, 0.5331377564 },
        { 3.0503282930, -2.2392300982, 0.4703818765 },
        { 2.1450160790, -1.4976204044, 0.3936105784 }
    }; const size_t smoothing_steps = 4;
    

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

        // A = a * A
        CHECK_CUBLAS( cublasSscal(cublasH, nn, &a, A, 1) );
        // A = b * A3 + A
        CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &b, A3, 1, A, 1) );
        // at this point, A = a * A + b * A3

        // A = c * A3 * A2 + A
        CHECK_CUBLAS( cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &c, A3, n, A2, n, &one, A, n) );

        /* Symmetrize A */
        symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace
    }

    /* Smoothing function */
    for (int i = 0; i < smoothing_steps; i++) {
        // A2 = A * A
        CHECK_CUBLAS( cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A, n, A, n, &zero, A2, n) );

        // A3 = A2 * A
        CHECK_CUBLAS( cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A2, n, A, n, &zero, A3, n) );

        /* Symmetrize A3 */
        symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace

        /* Compute A = 1.5 * A - 0.5 * A3 */
        // A = 1.5 * A
        CHECK_CUBLAS( cublasSscal(cublasH, nn, &one_n_half, A, 1) );
        // A = -0.5 * A3 + A
        CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &minus_half, A3, 1, A, 1) );

        /* Symmetrize A */
        symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace
    }

    // A = I + A
    add_identity(cublasH, A, n);
    // A = 0.5 * A
    CHECK_CUBLAS( cublasSscal(cublasH, nn, &half, A, 1) );

    /* Multiply the original matrix by A */
    // W = A_origin * A
    convert_double_to_float(mat, A2, nn);
    CHECK_CUBLAS( cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A2, n, A, n, &zero, A3, n) );

    /* Symmetrize W */
    symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace

    /* Copy the result back to mat */
    convert_float_to_double(A3, mat, nn);

    /* Free device memory */
    CHECK_CUDA( cudaFree(A) );
    CHECK_CUDA( cudaFree(A2) );
    CHECK_CUDA( cudaFree(A3) );
}

void composite_FP32_auto_scale(
    cublasHandle_t cublasH,
    cusolverDnHandle_t solverH,
    double* mat,
    const int n
) {
    size_t nn = n * n;
    
    // Use the Lanczos method to approximate the two-norm of the matrix
    double lo, up;
    approximate_two_norm(
        cublasH, solverH, mat, n, &lo, &up
    );

    // scale to have eigenvalues in [-1, 1]
    const double scale = up > 0.0 ? up : 1.0;
    const double inv_scale = 1.0/scale;
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, mat, 1) );

    // project the matrix using the composite_FP32 function
    composite_FP32(
        cublasH, mat, n, 0
    );

    // rescale the result back to the original scale
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale,  mat, 1) );
}

void composite_FP32_auto_scale_deflate(
    cublasHandle_t cublasH,
    cusolverDnHandle_t solverH,
    double* mat,
    const int n,
    const size_t k,
    const double tol,
    const double ortho_tol,
    const bool verbose
) {
    size_t nn = n * n;
    
    /* Step 1: compute the largest eigenpairs of the matrix */
    size_t r;
    double *eigenvalues, *eigenvectors;
    CHECK_CUDA( cudaMalloc(&eigenvalues,      k * sizeof(double)) );
    CHECK_CUDA( cudaMalloc(&eigenvectors, n * k * sizeof(double)) );

    double _ = compute_eigenpairs(
        cublasH, solverH, mat, n, k, &r, eigenvalues, eigenvectors, false, 0, tol, ortho_tol, verbose
    );

    std::vector<double> eigenvalues_host(r);
    CHECK_CUDA( cudaMemcpy(eigenvalues_host.data(), eigenvalues, r * sizeof(double), D2H) );

    /* Step 2: remove the largest eigenvalues from the matrix */
    for (int i = 0; i < r; i++) {
        // X <- X - \lambda_i * v_i v_i^T
        double lambda = -eigenvalues_host[i];
        double *v_i = eigenvectors + i * n;
        CHECK_CUBLAS( cublasDger(cublasH, n, n, &lambda, v_i, 1, v_i, 1, mat, n) );
    }

    /* Step 3: scale the deflated matrix */
    double up = compute_eigenpairs(
        cublasH, solverH, mat, n, 0, nullptr, nullptr, nullptr, true, 100, 1e-10, 1e-5, verbose
    );

    // scale to have eigenvalues in [-1, 1]
    const double scale = up > 0.0 ? 1.1 * up + 1e-5 : 1.0;
    const double inv_scale = 1.0/scale;
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, mat, 1) );

    /* Step 4: project the matrix using the composite_FP32 function */
    composite_FP32(
        cublasH, mat, n, verbose
    );

    /* Step 5: rescale the matrix back and add the deflated eigenvalues back */
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, mat, 1) );

    for (int i = 0; i < r; i++) {
        // X <- X + \lambda_i * v_i v_i^T
        double lambda = eigenvalues_host[i];
        if (lambda > 0.0) { // only add positive eigenvalues
            double *v_i = eigenvectors + i * n;
            CHECK_CUBLAS( cublasDger(cublasH, n, n, &lambda, v_i, 1, v_i, 1, mat, n) );
        }
    }

    /* Free device memory */
    CHECK_CUDA( cudaFree(eigenvalues) );
    CHECK_CUDA( cudaFree(eigenvectors) );
}