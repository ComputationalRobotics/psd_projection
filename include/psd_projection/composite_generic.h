#ifndef PSD_PROJECTION_COMPOSITE_GENERIC_H
#define PSD_PROJECTION_COMPOSITE_GENERIC_H

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "psd_projection/filter_bank.h"


/// @brief Apply a composite odd-quintic filter p = f_{T-1} o ... o f_0 to a
/// scaled symmetric matrix, in single precision, IN PLACE (A <- p(A)).
///
/// The matrix must already be scaled so its eigenvalues lie in [-1, 1].
/// Each stage computes  A <- a*A + b*A^3 + c*A^5  and re-symmetrises.
///
/// @param cublasH cuBLAS handle
/// @param A       device float matrix, n x n column-major, eigenvalues in [-1,1]
/// @param n       matrix dimension
/// @param stages  array of T stages (a,b,c)
/// @param T       number of stages
/// @param A2,A3   device float workspaces, each n*n  (A2 also used to symmetrise)
void apply_composite_FP32(
    cublasHandle_t cublasH, float* A, int n,
    const FilterStage* stages, int T,
    float* A2, float* A3);

/// @brief Half-precision variant: GEMMs run on Tensor Cores via cublasGemmEx
/// (CUBLAS_COMPUTE_32F_FAST_16F).  ``A``, ``A2``, ``A3`` are float; ``hA``,
/// ``hA2``, ``hA3`` are __half scratch of n*n each.
///
/// @note ALL SIX buffers must be allocated with a length padded up to a multiple
///       of 4, not just the __half scratch: the float->half conversion reads the
///       source in ``float4`` units (``convert_float_to_half4`` launches over
///       ``(n*n + 3)/4`` of them), so an unpadded ``A``/``A2``/``A3`` of odd
///       ``n*n`` is over-read past its end.
void apply_composite_FP16(
    cublasHandle_t cublasH, float* A, int n,
    const FilterStage* stages, int T,
    float* A2, float* A3,
    __half* hA, __half* hA2, __half* hA3);


#endif // PSD_PROJECTION_COMPOSITE_GENERIC_H
