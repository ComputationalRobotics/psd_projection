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

void composite_FP32_emulated(
    cublasHandle_t cublasH,
    double* mat,
    const int n,
    const bool verbose = false
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

    // std::vector<std::vector<float>> coeff = { 
    //     { 8.5018632351, -24.6330845767, 17.8466614026 },
    //     { 4.2394319792, -3.0803745982, 0.5596805290 },
    //     { 4.2371780379, -3.0779047407, 0.5594995022 },
    //     { 4.1553447421, -3.0255808203, 0.5534594007 },
    //     { 3.8719053120, -2.8289969308, 0.5331377564 },
    //     { 3.0503282930, -2.2392300982, 0.4703818765 },
    //     { 2.1450160790, -1.4976204044, 0.3936105784 }
    // }; const size_t smoothing_steps = 4;

    // const std::vector<std::vector<float>> coeff = {
    //     { 8.513860379623477, -25.280005082715576,  18.766059564488327 },
    //     { 4.256391883949461,  -3.159693659036471,   0.586422854272915 },
    //     { 4.253921206146349,  -3.157984594990142,   0.586227906363050 },
    //     { 4.243442810924305,  -3.150733464368069,   0.585400800614286 },
    //     { 4.199542905437564,  -3.120304290489466,   0.581930053860478 },
    //     { 4.024728598959554,  -2.998292294405710,   0.568017312836389 },
    //     { 3.452591186626141,  -2.587800781708363,   0.521308704062118 },
    //     { 2.430558807049796,  -1.783214052500875,   0.431237717248089 },
    //     { 1.907795624713394,  -1.285976165390910,   0.378615420076569 },
    //     { 1.875011744441277,  -1.250013049314579,   0.375001304931425 }
    // }; const size_t smoothing_steps = 0; // pre-lunch

    const std::vector<std::vector<float>> coeff = {
        { 8.509885302586273, -25.264304190830892,  18.753567899739625 },
        { 4.249573478922877,   -3.154976488114228,   0.585884782491327 },
        { 4.225122190777846,   -3.138044435084575,   0.583953455129916 },
        { 4.124838686994395,   -3.068332452805990,   0.576002953645695 },
        { 3.758010335802897,   -2.809273892403287,   0.546484206587685 },
        { 2.856177541291611,   -2.134056233175483,   0.470110769180275 },
        { 2.020600415776305,   -1.403721150466785,   0.390673896852026 },
        { 1.875875100481076,   -1.250971990481385,   0.375097212342072 },
        { 1.875,   -1.25,   0.375},
        { 1.875,   -1.25,   0.375},
    }; const size_t smoothing_steps = 0; // best

    float scale_factor = 1.001f;

    /* Approximation of the step function */
    for (int i = 0; i < coeff.size(); i++) {
        float a = coeff[i][0];
        float b = coeff[i][1];
        float c = coeff[i][2];

        if (i < 3) {
            a /= scale_factor;
            b /= scale_factor * scale_factor * scale_factor;
            c /= scale_factor * scale_factor * scale_factor * scale_factor 
                * scale_factor;
        }

        /* Compute the powers of A*/
        // A2 = A * A
        CHECK_CUBLAS( cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            A, CUDA_R_32F, n,
            A, CUDA_R_32F, n,
            &zero,
            A2, CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ) );

        // A3 = A2 * A
        CHECK_CUBLAS( cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            A, CUDA_R_32F, n,
            A2, CUDA_R_32F, n,
            &zero,
            A3, CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ) );

        // A = a * A
        CHECK_CUBLAS( cublasSscal(cublasH, nn, &a, A, 1) );
        // A = b * A3 + A
        CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &b, A3, 1, A, 1) );
        // at this point, A = a * A + b * A3

        // A = c * A3 * A2 + A
        CHECK_CUBLAS( cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &c,
            A3, CUDA_R_32F, n,
            A2, CUDA_R_32F, n,
            &one,
            A, CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ) );

        /* Symmetrize A */
        symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace
    }

    /* Smoothing function */
    for (int i = 0; i < smoothing_steps; i++) {
        // A2 = A * A
        CHECK_CUBLAS( cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            A, CUDA_R_32F, n,
            A, CUDA_R_32F, n,
            &zero,
            A2, CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ) );

        // A3 = A2 * A
        CHECK_CUBLAS( cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            A, CUDA_R_32F, n,
            A2, CUDA_R_32F, n,
            &zero,
            A3, CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ) );

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
    CHECK_CUBLAS( cublasGemmEx(
        cublasH,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &one,
        A2, CUDA_R_32F, n,
        A, CUDA_R_32F, n,
        &zero,
        A3, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ) );

    /* Symmetrize W */
    symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace

    /* Copy the result back to mat */
    convert_float_to_double(A3, mat, nn);

    /* Free device memory */
    CHECK_CUDA( cudaFree(A) );
    CHECK_CUDA( cudaFree(A2) );
    CHECK_CUDA( cudaFree(A3) );
}

void composite_FP32_emulated_auto_scale(
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
    composite_FP32_emulated(
        cublasH, mat, n, 0
    );

    // rescale the result back to the original scale
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale,  mat, 1) );
}

void composite_FP32_emulated_auto_scale_deflate(
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
    composite_FP32_emulated(
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