#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "psd_projection/composite_generic.h"
#include "psd_projection/utils.h"
#include "psd_projection/check.h"


void apply_composite_FP32(
    cublasHandle_t cublasH, float* A, int n,
    const FilterStage* stages, int T,
    float* A2, float* A3)
{
    const int nn = n * n;
    const float one = 1.0f, zero = 0.0f;

    for (int t = 0; t < T; ++t) {
        float a = stages[t].a, b = stages[t].b, c = stages[t].c;

        // A2 = A * A ;  A3 = A2 * A
        CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                 &one, A, n, A, n, &zero, A2, n));
        CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                 &one, A2, n, A, n, &zero, A3, n));

        // A = a*A + b*A3 ;  then A += c * (A3 * A2)   [ = c * A^5 ]
        CHECK_CUBLAS(cublasSscal(cublasH, nn, &a, A, 1));
        CHECK_CUBLAS(cublasSaxpy(cublasH, nn, &b, A3, 1, A, 1));
        CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                 &c, A3, n, A2, n, &one, A, n));

        // re-symmetrise (A2 as scratch)
        symmetrizeFloat(cublasH, A, n, A2);
    }
}

void apply_composite_FP16(
    cublasHandle_t cublasH, float* A, int n,
    const FilterStage* stages, int T,
    float* A2, float* A3,
    __half* hA, __half* hA2, __half* hA3)
{
    const int nn = n * n;
    const float one = 1.0f, zero = 0.0f;

    // Per-iteration input rescaling for FP16 numerical stability.  Dividing the
    // stage coefficients by powers of s applies f(x/s) instead of f(x), which
    // pulls the iterate slightly inward each stage.  Without it, FP16 rounding on
    // dense-edge spectra lets edge eigenvalues drift past the composite's unstable
    // fixed point (~1.53 for the g5 tail); the high-degree terms then amplify them
    // into a runaway -> FP16 overflow -> nan.
    // This is the same guard the base paper composite uses; s=1.01 restores
    // stability at every T with negligible accuracy cost.  FP32 does not overflow
    // and keeps the un-rescaled coefficients (see apply_composite_FP32).
    const float s = 1.01f;
    const float s3 = s * s * s, s5 = s3 * s * s;

    for (int t = 0; t < T; ++t) {
        float a = stages[t].a / s, b = stages[t].b / s3, c = stages[t].c / s5;

        // A2 = A * A
        convert_float_to_half4(A, hA, nn);
        CHECK_CUBLAS(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                  &one, hA, CUDA_R_16F, n, hA, CUDA_R_16F, n,
                                  &zero, A2, CUDA_R_32F, n,
                                  CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        // A3 = A2 * A
        convert_float_to_half4(A2, hA2, nn);
        CHECK_CUBLAS(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                  &one, hA, CUDA_R_16F, n, hA2, CUDA_R_16F, n,
                                  &zero, A3, CUDA_R_32F, n,
                                  CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // A = a*A + b*A3
        CHECK_CUBLAS(cublasSscal(cublasH, nn, &a, A, 1));
        CHECK_CUBLAS(cublasSaxpy(cublasH, nn, &b, A3, 1, A, 1));

        // A += c * (A2 * A3)   [ = c * A^5 ]
        convert_float_to_half4(A3, hA3, nn);
        CHECK_CUBLAS(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                  &c, hA2, CUDA_R_16F, n, hA3, CUDA_R_16F, n,
                                  &one, A, CUDA_R_32F, n,
                                  CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        symmetrizeFloat(cublasH, A, n, A2);
    }
}

