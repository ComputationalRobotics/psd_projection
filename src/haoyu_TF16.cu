#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_fp16.h>
#include <cmath>

#include "psd_projection/haoyu_TF16.h"
#include "psd_projection/check.h"
#include "psd_projection/utils.h"

void haoyu_TF16(
    cublasHandle_t cublasH,
    float* mat,
    const int n
) {
    const int nn = n * n;

    // Allocate device buffers
    float *dA_our, *dTmp, *dT1, *dT2, *dF;
    CHECK_CUDA(cudaMalloc(&dA_our,  nn*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dTmp,    nn*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dT1,     nn*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dT2,     nn*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dF,      nn*sizeof(float)));

    // Copy host to device
    CHECK_CUDA(cudaMemcpy(dA_our, mat, nn*sizeof(float), D2D));

    // half buffers
    __half *dT3_half, *dT4_half; 
    CHECK_CUDA(cudaMalloc(&dT3_half, nn*sizeof(__half))); 
    CHECK_CUDA(cudaMalloc(&dT4_half, nn*sizeof(__half)));

    const float one = 1.0f, zero = 0.0f;
    float half = 0.5f;

    // Iterative algorithm in float, printing after each iter
    for (int iter = 1; iter <= 4; iter++) {
        // T1 = A_our * A_our
        convert_float_to_half4(dA_our, dT3_half, nn);
        CHECK_CUBLAS(cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            dT3_half, CUDA_R_16F, n,
            dT3_half, CUDA_R_16F, n,
            &zero,
            dT1,    CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // T2 = I - T1
        identity_minus(dT1, dT2, n);

        // T1 = T2 * T2
        convert_float_to_half4(dT2, dT3_half, nn);
        CHECK_CUBLAS(cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            dT3_half, CUDA_R_16F, n,
            dT3_half, CUDA_R_16F, n,
            &zero,
            dT1, CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // F = I + log(iter+10)*T1
        float logv = std::log(iter + 10.0f);
        CHECK_CUBLAS(cublasSscal(cublasH, nn, &logv, dT1, 1));
        identity_plus(dT1, dF, n);

        // A_our = A_our * F (to dTmp, then copy back)
        convert_float_to_half4(dA_our, dT3_half, nn);
        convert_float_to_half4(dF, dT4_half, nn);
        CHECK_CUBLAS(cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            dT3_half, CUDA_R_16F, n,
            dT4_half, CUDA_R_16F, n,
            &zero,
            dTmp,   CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        
        CHECK_CUDA(cudaMemcpy(dA_our, dTmp, nn*sizeof(float), cudaMemcpyDeviceToDevice));

        // T1 = A_our^2, T2 = I - T1
        convert_float_to_half4(dA_our, dT3_half, nn);
        CHECK_CUBLAS(cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            dT3_half, CUDA_R_16F, n,
            dT3_half, CUDA_R_16F, n,
            &zero,
            dT1,    CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        identity_minus(dT1, dT2, n);

        // F = I + (1/sqrt(iter))*T2
        float invs = 1.0f / std::sqrt((float)iter);
        CHECK_CUBLAS(cublasSscal(cublasH, nn, &invs, dT2, 1));
        identity_plus(dT2, dF, n);
            
        // A_our = A_our * F (to dTmp)
        convert_float_to_half4(dA_our, dT3_half, nn);
        convert_float_to_half4(dF, dT4_half, nn);
        CHECK_CUBLAS(cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            dT3_half, CUDA_R_16F, n,
            dT4_half, CUDA_R_16F, n,
            &zero,
            dTmp,   CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // A_our <-- Tmp
        CHECK_CUDA(cudaMemcpy(dA_our, dTmp, nn*sizeof(float), cudaMemcpyDeviceToDevice));

        // force symmetry: A_our <-- 0.5 * (A_our + Tmp')
        CHECK_CUBLAS(cublasSgeam(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_T,
            n, n,
            &half, dA_our, n,
            &half, dTmp, n,
            dA_our, n));
    }
    
    // Final combine: mat = mat * (A_our + I) / 2
    identity_plus(dA_our, dF, n);

    convert_float_to_half4(mat, dT3_half, nn);
    convert_float_to_half4(dF, dT4_half, nn);
    CHECK_CUBLAS(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &one,
        dT3_half, CUDA_R_16F, n,
        dT4_half, CUDA_R_16F, n,
        &zero,
        dTmp,    CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaMemcpy(mat, dTmp, nn*sizeof(float), D2D));
    CHECK_CUBLAS(cublasSscal(cublasH, nn, &half, mat, 1));

    /* Free memory */
    CHECK_CUDA(cudaFree(dA_our));
    CHECK_CUDA(cudaFree(dTmp));
    CHECK_CUDA(cudaFree(dT1));
    CHECK_CUDA(cudaFree(dT2));
    CHECK_CUDA(cudaFree(dF));
    CHECK_CUDA(cudaFree(dT3_half));
    CHECK_CUDA(cudaFree(dT4_half));

    return;
}