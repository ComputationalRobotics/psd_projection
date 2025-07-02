#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "psd_projection/composite_TF16.h"
#include "psd_projection/utils.h"

void composite_TF16(
    cublasHandle_t cublasH,
    double* mat,
    const int n
) {
    const int nn = n * n;

    /* Allocations */
    // device memory
    float *A, *A2, *A3;
    CHECK_CUDA( cudaMalloc(&A,  nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&A2, nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&A3, nn * sizeof(float)) );

    __half *hA, *hA2, *hA3;
    CHECK_CUDA(cudaMalloc(&hA,  nn * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&hA2, nn * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&hA3, nn * sizeof(__half)));

    // useful constants
    const float half       =  0.5f;
    const float minus_half = -0.5f;
    const float one        =  1.0f;
    const float one_n_half =  1.5f;
    const float zero       =  0.0f;

    convert_double_to_float(mat, A, nn);

    /* Coefficients */
    // std::vector<std::vector<float>> coeff = {
    //     {8.4724206924, -24.5001735687, 17.7268180847},
    //     {4.2052841187, -3.0549299717, 0.5567536354},
    //     {4.0443077087, -2.9473149776, 0.5449726582},
    //     {3.5078327656, -2.5842490196, 0.5067413449},
    //     {2.5075511932, -1.8485442400, 0.4358045161}
    // };
    // std::vector<std::vector<float>> coeff = { 
    //     { 8.3885353390, -23.7796270883, 16.8664591580 }, 
    //     { 4.1636476423, -2.9650849331, 0.5297319805 }, 
    //     { 4.0042650581, -2.8606348801, 0.5185227850 }, 
    //     { 3.4731017481, -2.5082466382, 0.4821470022 }, 
    //     { 2.4827239537, -1.7941788274, 0.4146530436 }, 
    // };

    std::vector<std::vector<float>> coeff = { 
        { 8.3937001154, -23.7945582332, 16.8758390904 }, 
        { 4.1803895500, -2.9788012917, 0.5318143742 }, 
        { 4.0578478573, -2.9013956514, 0.5233571836 }, 
        { 3.6289664769, -2.6254124593, 0.4963343458 }, 
        { 2.7619020904, -1.9865006927, 0.4388497859 }, 
        { 2.0674922563, -1.4317903208, 0.3876934521 }, 
        { 1.8438914749, -1.1872290786, 0.3433825749 }, 
    };

    /* Approximation of the step function */
    for (int i = 0; i < coeff.size(); i++) {
        const float a = coeff[i][0];
        const float b = coeff[i][1];
        const float c = coeff[i][2];

        /* Compute the powers of A*/
        // A2 = A * A
        convert_float_to_half4(A, hA, nn);
        CHECK_CUBLAS(cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            hA, CUDA_R_16F, n,
            hA, CUDA_R_16F, n,
            &zero,
            A2,    CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // A3 = A2 * A
        convert_float_to_half4(A2, hA2, nn);
        CHECK_CUBLAS(cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            hA, CUDA_R_16F, n,
            hA2, CUDA_R_16F, n,
            &zero,
            A3,    CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // A = a * A
        CHECK_CUBLAS( cublasSscal(cublasH, nn, &a, A, 1) );
        // A = b * A3 + A
        CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &b, A3, 1, A, 1) );
        // at this point, A = a * A + b * A3

        /* Symmetrize A3, A5 */
        // symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace
        // symmetrizeFloat(cublasH, A5, n, A2); // we use A2 as a workspace

        // A = c * A3 * A2 + A
        convert_float_to_half4(A3, hA3, nn);
        CHECK_CUBLAS(cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &c,
            hA2, CUDA_R_16F, n,
            hA3, CUDA_R_16F, n,
            &one,
            A,    CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        /* Symmetrize A */
        symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace
    }

    // /* Smoothing function */
    // for (int i = 0; i < 3; i++) {
    //     // A2 = A * A
    //     convert_float_to_half4(A, hA, nn);
    //     CHECK_CUBLAS(cublasGemmEx(
    //         cublasH,
    //         CUBLAS_OP_N, CUBLAS_OP_N,
    //         n, n, n,
    //         &one,
    //         hA, CUDA_R_16F, n,
    //         hA, CUDA_R_16F, n,
    //         &zero,
    //         A2,    CUDA_R_32F, n,
    //         CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    //     // A3 = A2 * A
    //     convert_float_to_half4(A2, hA2, nn);
    //     CHECK_CUBLAS(cublasGemmEx(
    //         cublasH,
    //         CUBLAS_OP_N, CUBLAS_OP_N,
    //         n, n, n,
    //         &one,
    //         hA, CUDA_R_16F, n,
    //         hA2, CUDA_R_16F, n,
    //         &zero,
    //         A3,    CUDA_R_32F, n,
    //         CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    //     /* Symmetrize A3 */
    //     symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace

    //     /* Compute A = 1.5 * A - 0.5 * A3 */
    //     // A = 1.5 * A
    //     CHECK_CUBLAS( cublasSscal(cublasH, nn, &one_n_half, A, 1) );
    //     // A = -0.5 * A3 + A
    //     CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &minus_half, A3, 1, A, 1) );

    //     /* Symmetrize A */
    //     symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace
    // }

    add_identity(cublasH, A, n);
    // A = 0.5 * A
    CHECK_CUBLAS( cublasSscal(cublasH, nn, &half, A, 1) );

    /* Multiply the original matrix by A */
    convert_double_to_float(mat, A2, nn);
    convert_float_to_half4(A, hA, nn);
    convert_float_to_half4(A2, hA2, nn);
    CHECK_CUBLAS(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &one,
        hA2, CUDA_R_16F, n,
        hA, CUDA_R_16F, n,
        &zero,
        A3,    CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    /* Symmetrize W */
    symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace

    /* Copy the result back to mat */
    convert_float_to_double(A3, mat, nn);

    /* Free device memory */
    CHECK_CUDA( cudaFree(A) );
    CHECK_CUDA( cudaFree(A2) );
    CHECK_CUDA( cudaFree(A3) );
    CHECK_CUDA( cudaFree(hA) );
    CHECK_CUDA( cudaFree(hA2) );
    CHECK_CUDA( cudaFree(hA3) );
}