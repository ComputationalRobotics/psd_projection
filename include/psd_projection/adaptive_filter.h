#ifndef PSD_PROJECTION_ADAPTIVE_FILTER_H
#define PSD_PROJECTION_ADAPTIVE_FILTER_H

#include <cublas_v2.h>
#include <cusolverDn.h>
#include "psd_projection/spectral_sketch.h"


enum class Precision { FP32, FP16 };

struct AdaptiveOptions {
    double    tol            = 1e-3;            ///< target RELATIVE Frobenius error
    Precision precision      = Precision::FP32;
    // SLQ sketch default = 8 probes x 20 steps, blocked:
    // 8x20 is decision-identical to the old 8x40 (same chosen T, same deflation count k)
    // on every tested spectrum but ~1.15x faster at n~1e4.  Steps (quadrature resolution)
    // are cheaper to cut than probes (which set the outlier-count variance) -- keep 8 probes.
    int       slq_steps      = 20;             ///< Lanczos steps per SLQ probe
    int       slq_probes     = 8;              ///< number of SLQ probes
    bool      sketch_block   = true;           ///< block all SLQ probes into one GEMM/step (runs in FP64; ~n_probes x less A traffic; batched reorth; scale from the sketch's own residual-based Ritz bound -- no separate two-norm pass)
    double    safety         = 1.5;            ///< margin multiplier on predicted error
    bool      enable_deflation = true;         ///< deflate outliers when beneficial (no n size gate; self-suppresses via count<frac*n, n/3 clamp)
    double    deflation_frac = 0.01;           ///< per side, deflate only if #outliers < frac*n (else skip that side)
    double    outlier_thresh = 0.05;           ///< an eigenvalue counts as an outlier when |lambda| > outlier_thresh*scale (scaled |y|>outlier_thresh)
    unsigned long long seed  = 1234ULL;
    bool      verbose        = false;
};

struct AdaptiveReport {
    int    filter_index      = -1;
    int    T                 = 0;
    int    gemms             = 0;    ///< composite GEMMs actually used (3T+1)
    double eps               = 0.0;  ///< dead-zone of the chosen filter
    double scale             = 0.0;  ///< scaling applied (||A||_2 upper bound)
    double lam_min_scaled    = 0.0;
    double lam_max_scaled    = 0.0;
    double predicted_rel_err = 0.0;  ///< model prediction (before safety factor)
    int    deflated          = 0;    ///< number of eigenpairs deflated
    double deflation_ms      = 0.0;  ///< measured wall time of the deflation stage (ms)
    bool   qualified         = false;///< a filter met tol*safety (else best-effort)
};

/// @brief Pure filter-selection over the offline bank given a spectral sketch.
/// Returns the index into FILTER_BANK and stores the predicted relative error.
/// (Exposed for unit testing of the error model.)
///
/// @param deflated_pos_scaled  sum_{deflated lambda>0} (lambda/S.scale)^2 -- the
///   scaled Frobenius mass of the positive eigenpairs already removed by deflation
///   and added back EXACTLY.  It belongs in ||Pi_+(A)||^2 (the relative-error
///   denominator) even though it is not in the remainder sketch; omitting it makes
///   the selector over-estimate the relative error and over-provision T.
int select_filter(const SpectralSketch& S, const AdaptiveOptions& opts,
                  double* predicted_rel_err, bool* qualified,
                  double deflated_pos_scaled = 0.0);

/// @brief Spectrum-aware adaptive PSD-cone projection.
///
/// Overwrites the symmetric matrix ``mat`` (double, device, column-major) with
/// an approximation of Pi_+(mat), choosing the cheapest composite polynomial
/// filter whose model-predicted error meets ``opts.tol``.  Pipeline:
///   1. cheap spectral sketch (Lanczos range + SLQ near-zero mass);
///   2. optional outlier deflation, followed by a re-sketch of the remainder;
///   3. adaptive filter selection (minimum GEMMs meeting tolerance);
///   4-5. tight scaling into [-1,1], composite filtering, projection recovery
///        and unscaling (all inside project_core);
///   6. add back the deflated positive eigenpairs.
///
/// @param cublasH a cuBLAS handle
/// @param cusolverH a cuSOLVER handle
/// @param mat the matrix to be projected, stored in column-major order; it is
///        OVERWRITTEN in place with the approximation of Pi_+(mat)
/// @param n the size of the matrix (assumed square, i.e. `n x n`)
/// @param opts tolerance, precision, sketch and deflation settings; the defaults
///        target a 1e-3 relative Frobenius error in FP32
/// @return a report describing what was chosen (filter, GEMMs, prediction, ...).
/// @note Unlike the fixed composite routines, the matrix does NOT need to be
///       pre-scaled: the spectral sketch derives an a-posteriori upper bound on
///       ||mat||_2 and scales into [-1,1] internally.
AdaptiveReport psd_projection_adaptive(
    cublasHandle_t cublasH, cusolverDnHandle_t cusolverH,
    double* mat, int n,
    const AdaptiveOptions& opts = AdaptiveOptions());


#endif // PSD_PROJECTION_ADAPTIVE_FILTER_H
