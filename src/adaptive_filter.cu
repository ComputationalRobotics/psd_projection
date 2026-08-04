#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <limits>

#include "psd_projection/adaptive_filter.h"
#include "psd_projection/composite_generic.h"
#include "psd_projection/filter_bank.h"
#include "psd_projection/spectral_sketch.h"
#include "psd_projection/lanczos.h"
#include "psd_projection/lobpcg.h"
#include "psd_projection/utils.h"
#include "psd_projection/check.h"


// Clamp a device float array element-wise into [-1, 1].  Applied to the composite
// filter INPUT as a cheap entry-magnitude guard.  NOTE: this bounds the matrix
// ENTRIES, which does NOT by itself bound the eigenvalues.  What keeps the scaled
// spectrum inside [-1,1], where the quintic is stable, is S.scale (the sketch's
// a-posteriori ||A||_2 bound); the clamp only stops individual entries from blowing
// up if that estimate comes out slightly too small.  (The final projection multiply
// uses the un-clamped scaled matrix, preserving eigenvalue magnitudes.)
__global__ void clamp_unit_kernel(float* x, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = v > 1.0f ? 1.0f : (v < -1.0f ? -1.0f : v);
    }
}
static void clamp_unit(float* x, size_t n) {
    const int tpb = 256;
    size_t blocks = (n + tpb - 1) / tpb;
    clamp_unit_kernel<<<(unsigned)blocks, tpb>>>(x, n);
    CHECK_CUDA(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Error model + filter selection (pure; depends only on the sketch + options).
//
// Scaled per-eigenvalue ReLU residual  d(y) = |0.5 y (1+p(y)) - relu(y)|:
//    |y| >= eps :  d(y) <= e_relu               (measured sup of the filter on [eps,1])
//    |y| <  eps :  d(y) ~ 0.5 |y|               (near 0, p(y)->0 so |p-sign|~1)
// so the scaled Frobenius error obeys
//    err_scaled^2 ~ e_relu^2 * N_out + 0.25 * sum_{|y|<eps} y^2 ,
// with N_out = #{|y| >= eps} and the near-zero mass estimated from a
// Gaussian-broadened density (see SpectralSketch).  Unscaled error =
// scale * err_scaled; the scale cancels in the RELATIVE error against
// ||Pi_+(A)||_F = scale*sqrt(pos + deflated_pos_scaled) -- the sketched positive
// mass PLUS the positive mass already removed by deflation and added back exactly
// (see the deflated_pos_scaled argument).  A per-precision arithmetic floor accounts
// for the finite-precision GEMM error that no polynomial can beat.
// ---------------------------------------------------------------------------
int select_filter(const SpectralSketch& S, const AdaptiveOptions& opts,
                  double* predicted_rel_err, bool* qualified,
                  double deflated_pos_scaled)
{
    const double pos = S.positive_frob_sq_scaled();
    const double tot = S.total_frob_sq_scaled();
    const double floor = (opts.precision == Precision::FP16) ? 5e-4 : 1e-6;

    // fallback = the most accurate filter available (min eps, then max T)
    int fb_idx = 0;
    for (int i = 1; i < FILTER_BANK_SIZE; ++i) {
        if (FILTER_BANK[i].eps < FILTER_BANK[fb_idx].eps - 1e-15 ||
            (std::fabs(FILTER_BANK[i].eps - FILTER_BANK[fb_idx].eps) <= 1e-15 &&
             FILTER_BANK[i].gemms > FILTER_BANK[fb_idx].gemms))
            fb_idx = i;
    }

    // matrix has (essentially) no positive part -> Pi_+(A) is the zero matrix.
    // Return a filter only for the report; psd_projection_adaptive detects this
    // case and zeroes the output directly (this filter is NOT applied).
    if (pos <= 1e-12 * (tot + 1e-300)) {
        if (predicted_rel_err) *predicted_rel_err = 0.0;
        if (qualified) *qualified = true;
        return fb_idx;
    }

    // Relative error is measured against the FULL ||Pi_+(A)||, which includes the
    // exactly-deflated positive eigenpairs, not just the remainder the filter acts on.
    const double denom = std::sqrt(pos + deflated_pos_scaled);

    int    best_idx = -1;
    int    best_gemms = std::numeric_limits<int>::max();
    double best_pred = 1e300;
    double fb_pred = 1e300;

    for (int i = 0; i < FILTER_BANK_SIZE; ++i) {
        const Filter& f = FILTER_BANK[i];
        double n_out = (double)S.n * (1.0 - S.near_zero_count_frac(f.eps));
        if (n_out < 0.0) n_out = 0.0;
        double m_in  = S.near_zero_frob_sq_scaled(f.eps);
        double err2  = f.e_relu * f.e_relu * n_out + 0.25 * m_in;
        double pred  = std::sqrt(err2 / (denom * denom) + floor * floor);

        if (i == fb_idx) fb_pred = pred;

        // qualifying = predicted error with safety margin is within tolerance
        if (opts.safety * pred <= opts.tol) {
            if (f.gemms < best_gemms ||
                (f.gemms == best_gemms && pred < best_pred)) {
                best_gemms = f.gemms; best_pred = pred; best_idx = i;
            }
        }
    }

    if (best_idx >= 0) {
        if (predicted_rel_err) *predicted_rel_err = best_pred;
        if (qualified) *qualified = true;
        return best_idx;
    }
    // nothing meets the tolerance -> fall back to the most accurate filter
    if (predicted_rel_err) *predicted_rel_err = fb_pred;
    if (qualified) *qualified = false;
    return fb_idx;
}

// Scale ``mat`` into [-1,1] using the sketch's ||A||_2 bound, apply the
// chosen composite filter, recover Pi_+ and unscale, IN PLACE.
static void project_core(cublasHandle_t cublasH, double* mat, int n,
                         const SpectralSketch& S, const Filter& f,
                         Precision precision)
{
    const int    ne = n * n;                       // int element count for cuBLAS/convert
    const size_t nn = (size_t)n * n;               // 64-bit element count (allocations)
    // pad to a multiple of 4 so the float4 packing in the FP16 path never reads
    // past the end of the source buffers (safe for FP32 too).
    const size_t stride = ((nn + 3) / 4) * 4;

    float *Af, *Xf, *A2, *A3;
    CHECK_CUDA(cudaMalloc(&Af, stride * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&Xf, stride * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&A2, stride * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&A3, stride * sizeof(float)));

    convert_double_to_float(mat, Af, ne);
    float inv_scale = (float)(1.0 / S.scale);
    CHECK_CUBLAS(cublasSscal(cublasH, ne, &inv_scale, Af, 1));   // Af = A / scale
    CHECK_CUDA(cudaMemcpy(Xf, Af, nn * sizeof(float), cudaMemcpyDeviceToDevice)); // true magnitudes
    clamp_unit(Af, nn);   // stability guard: filter input strictly in [-1,1]

    if (precision == Precision::FP16) {
        __half *hA, *hA2, *hA3;
        CHECK_CUDA(cudaMalloc(&hA,  stride * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&hA2, stride * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&hA3, stride * sizeof(__half)));
        apply_composite_FP16(cublasH, Af, n, f.stages, f.T, A2, A3, hA, hA2, hA3);
        CHECK_CUDA(cudaFree(hA)); CHECK_CUDA(cudaFree(hA2)); CHECK_CUDA(cudaFree(hA3));
    } else {
        apply_composite_FP32(cublasH, Af, n, f.stages, f.T, A2, A3);
    }

    // Af <- 0.5 (I + Af)   (projector onto the positive eigenspace)
    add_identity(cublasH, Af, n);
    float half = 0.5f;
    CHECK_CUBLAS(cublasSscal(cublasH, ne, &half, Af, 1));

    // A3 = Xf * Af = Pi_+(A/scale)  ; symmetrise.  This final projection is the last
    // GEMM producing the answer, so its precision caps the result.  We match it to the
    // FILTER precision: for the FP16 filter, run it on TF32 tensor cores (~2x faster and
    // the FP16 filter already sets the error floor, so no accuracy is lost); for the FP32
    // filter, keep a true FP32 GEMM (the filter is accurate to ~1e-5..1e-4, and TF32's
    // 10-bit mantissa would otherwise cap the result near ~2.5e-4).  Xf, Af are O(1) in
    // [-1,1] (true magnitudes recovered by the scale rescale below).
    const float one = 1.0f, zero = 0.0f;
    const bool proj_tf32 = (precision == Precision::FP16);
    if (proj_tf32) CHECK_CUBLAS(cublasSetMathMode(cublasH, CUBLAS_TF32_TENSOR_OP_MATH));
    CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                             &one, Xf, n, Af, n, &zero, A3, n));
    if (proj_tf32) CHECK_CUBLAS(cublasSetMathMode(cublasH, CUBLAS_DEFAULT_MATH));
    symmetrizeFloat(cublasH, A3, n, A2);

    // mat = scale * double(A3)
    convert_float_to_double(A3, mat, ne);
    double sc = S.scale;
    CHECK_CUBLAS(cublasDscal(cublasH, ne, &sc, mat, 1));

    CHECK_CUDA(cudaFree(Af)); CHECK_CUDA(cudaFree(Xf));
    CHECK_CUDA(cudaFree(A2)); CHECK_CUDA(cudaFree(A3));
}

// ---------------------------------------------------------------------------
// Full adaptive projection pipeline.
// ---------------------------------------------------------------------------
AdaptiveReport psd_projection_adaptive(
    cublasHandle_t cublasH, cusolverDnHandle_t cusolverH,
    double* mat, int n,
    const AdaptiveOptions& opts)
{
    AdaptiveReport rep;
    const size_t nn = (size_t)n * n;
    const double minus_one = -1.0;
    // size_t-safe whole-matrix negation: cublasDscal's count is int, so a single
    // cublasDscal((int)nn,...) silently overflows for n >= 46341 (nn > INT_MAX)
    // and becomes a no-op; fall back to a per-column scal in that regime.
    auto negate_mat = [&]() {
        if (nn <= (size_t)2147483647)
            CHECK_CUBLAS(cublasDscal(cublasH, (int)nn, &minus_one, mat, 1));
        else
            for (int j = 0; j < n; ++j)
                CHECK_CUBLAS(cublasDscal(cublasH, n, &minus_one, mat + (size_t)j * n, 1));
    };

    // ---- 1. spectral sketch -------------------------------------------------
    SpectralSketch S = compute_spectral_sketch(
        cublasH, cusolverH, mat, n, opts.slq_steps, opts.slq_probes, opts.seed,
        /*use_fp32=*/false, opts.sketch_block);

    // ---- 2. optional outlier deflation (LOBPCG on A and -A) ----------------
    // Deflation helps only when a *few* eigenvalues dominate the radius.  From the
    // density, count -- PER SIDE -- how many scaled eigenvalues exceed outlier_thresh
    // (default 0.05).  Each side is treated INDEPENDENTLY: deflate that side (extract
    // exactly its count) only if the count is strictly below lambda*n; a count >=
    // lambda*n means the side is dense (not outlier-dominated), so leave it to the
    // filter.  No size gate: the deflation_frac*n threshold and the 3k<=n (n/3) clamp
    // are the hard safety clamps, and they suppress deflation on their own for tiny n.
    int k_pos = 0, k_neg = 0;
    double defl_pos_mass = 0.0;   // sum_{deflated lambda>0} lambda^2 (original units)
    double *evec_max = nullptr, *eval_max = nullptr;
    double *evec_min = nullptr, *eval_min = nullptr;
    if (opts.enable_deflation) {
        double kp = 0.0, kn = 0.0;
        const double ot = opts.outlier_thresh;                        // |y| > ot counts as an outlier
        for (size_t j = 0; j < S.theta.size(); ++j) {
            if (S.theta[j] >  ot) kp += S.tau[j];
            if (S.theta[j] < -ot) kn += S.tau[j];
        }
        kp *= (double)S.n / (double)S.n_probes;
        kn *= (double)S.n / (double)S.n_probes;
        const double thresh = opts.deflation_frac * (double)n;        // lambda * n
        int kp_est = (int)std::ceil(kp);
        int kn_est = (int)std::ceil(kn);
        // Per side, independently: deflate that side only if its outlier count is
        // STRICTLY BELOW lambda*n (few enough to be genuine outliers).  A count at
        // or above lambda*n means that side is dense, not outlier-dominated, so we
        // do nothing on it and leave it to the polynomial filter.
        if (kp_est >= 1 && (double)kp_est < thresh) k_pos = kp_est;
        if (kn_est >= 1 && (double)kn_est < thresh) k_neg = kn_est;
        // Hard safety clamp (never binds for lambda <= 1/3 in the tested regime):
        // LOBPCG requires 3k <= n.
        k_pos = std::min(k_pos, n / 3);
        k_neg = std::min(k_neg, n / 3);
    }

    if (k_pos > 0 || k_neg > 0) {
        cudaEvent_t d0, d1; cudaEventCreate(&d0); cudaEventCreate(&d1);
        cudaEventRecord(d0);   // time the whole deflation stage
        const int k_ritz = std::max(k_pos, k_neg);   // one Lanczos seeds both sides
        CHECK_CUDA(cudaMalloc(&evec_max, (size_t)n * k_ritz * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&eval_max,          (size_t)k_ritz * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&evec_min, (size_t)n * k_ritz * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&eval_min,          (size_t)k_ritz * sizeof(double)));
        // If a Lanczos breakdown makes compute_extremal_ritz return fewer than
        // k_ritz pairs (degenerate / low-rank outliers), the tail seed columns
        // would be uninitialised; zero them so the LOBPCG warm start sees a finite
        // (zero) seed there and re-orthonormalises it, rather than reading garbage.
        CHECK_CUDA(cudaMemset(evec_max, 0, (size_t)n * k_ritz * sizeof(double)));
        CHECK_CUDA(cudaMemset(eval_max, 0,          (size_t)k_ritz * sizeof(double)));
        CHECK_CUDA(cudaMemset(evec_min, 0, (size_t)n * k_ritz * sizeof(double)));
        CHECK_CUDA(cudaMemset(eval_min, 0,          (size_t)k_ritz * sizeof(double)));

        // (D) warm-start seeds: the k_ritz largest AND smallest Ritz pairs from one
        // reorthogonalised Lanczos run on A (first k_pos / k_neg columns are used).
        compute_extremal_ritz(cublasH, cusolverH, mat, n, k_ritz,
                              std::max(3 * k_ritz + 10, 40),
                              evec_max, eval_max, evec_min, eval_min, opts.seed);

        // positive side: refine the k_pos largest-algebraic eigenpairs of A, subtract
        if (k_pos > 0) {
            // relative_tol=true: the residual test must be scale-invariant here,
            // since `mat` is unscaled at this point.
            lobpcg(cublasH, cusolverH, mat, evec_max, eval_max, n, k_pos, true, 50, 1e-6,
                   false, /*relative_tol=*/true);
            // Frobenius mass of the deflated POSITIVE eigenpairs (eval_max is still
            // +lambda here, before the negate below).  They are added back exactly, so
            // they belong in ||Pi_+||^2 when the composite degree is chosen.
            {
                std::vector<double> hev(k_pos);
                CHECK_CUDA(cudaMemcpy(hev.data(), eval_max, (size_t)k_pos * sizeof(double),
                                      cudaMemcpyDeviceToHost));
                for (int i = 0; i < k_pos; ++i)
                    if (hev[i] > 0.0) defl_pos_mass += hev[i] * hev[i];
            }
            CHECK_CUBLAS(cublasDscal(cublasH, k_pos, &minus_one, eval_max, 1));  // negate
            cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_DEVICE);
            for (int i = 0; i < k_pos; ++i)
                CHECK_CUBLAS(cublasDger(cublasH, n, n, eval_max + i,
                                        evec_max + (size_t)i * n, 1,
                                        evec_max + (size_t)i * n, 1, mat, n));
            cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_HOST);
        }

        // negative side: refine the k_neg most-negative eigenpairs via -A, subtract.
        // Warm-started from the sketch's smallest Ritz pairs (seed = eigenvalues of
        // -A = -lambda_min).
        if (k_neg > 0) {
            // lambda_min(A) -> eig of -A.  NOTE: lobpcg treats D as OUTPUT only (it
            // never reads the warm-start eigenvalues), so this negation does not
            // actually seed anything -- it is kept so eval_min carries consistent
            // signs if that contract ever changes.
            CHECK_CUBLAS(cublasDscal(cublasH, k_neg, &minus_one, eval_min, 1));
            negate_mat();                                                       // form -A
            lobpcg(cublasH, cusolverH, mat, evec_min, eval_min, n, k_neg, true, 50, 1e-6,
                   false, /*relative_tol=*/true);
            negate_mat();                                                       // restore A
            cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_DEVICE);
            for (int i = 0; i < k_neg; ++i)  // eval_min are eigenvalues of -A = -lambda
                CHECK_CUBLAS(cublasDger(cublasH, n, n, eval_min + i,
                                        evec_min + (size_t)i * n, 1,
                                        evec_min + (size_t)i * n, 1, mat, n));
            cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_HOST);
        }

        rep.deflated = k_pos + k_neg;
        // re-sketch the deflated residual (smaller radius, larger effective gap)
        S = compute_spectral_sketch(cublasH, cusolverH, mat, n,
                                    opts.slq_steps, opts.slq_probes, opts.seed + 7ULL,
                                    /*use_fp32=*/false, opts.sketch_block);
        cudaEventRecord(d1); cudaEventSynchronize(d1);
        float dms = 0.0f; cudaEventElapsedTime(&dms, d0, d1); rep.deflation_ms = dms;
        cudaEventDestroy(d0); cudaEventDestroy(d1);
    }

    // ---- 3. select filter ---------------------------------------------------
    // Fold the exactly-deflated positive mass (in the remainder's scaled units) into
    // ||Pi_+||^2 so the degree is not over-provisioned after deflation.
    double pred = 0.0; bool qual = false;
    const double defl_pos_scaled = defl_pos_mass / (S.scale * S.scale);
    int idx = select_filter(S, opts, &pred, &qual, defl_pos_scaled);
    const Filter& f = FILTER_BANK[idx];

    // If the (deflated) matrix has no positive part, its PSD projection is the
    // zero matrix; zero it directly rather than applying a filter that would
    // otherwise leave spurious O(scale) mass near the negative end of the spectrum.
    bool no_positive = (S.positive_frob_sq_scaled()
                        <= 1e-12 * (S.total_frob_sq_scaled() + 1e-300));

    rep.filter_index = idx;
    rep.T = no_positive ? 0 : f.T;
    rep.gemms = no_positive ? 0 : f.gemms;
    rep.eps = f.eps;
    rep.scale = S.scale;
    rep.lam_min_scaled = S.lam_min_scaled;
    rep.lam_max_scaled = S.lam_max_scaled;
    rep.predicted_rel_err = pred;
    rep.qualified = qual;

    if (opts.verbose) {
        std::printf("[adaptive] scale=%.4e lam=[%.3f,%.3f] deflated=%d "
                    "-> %s eps=%.3g T=%d gemms=%d pred_rel=%.3e %s\n",
                    S.scale, S.lam_min_scaled, S.lam_max_scaled, rep.deflated,
                    no_positive ? "ZERO(no +part)" : "filter", f.eps, rep.T, rep.gemms,
                    pred, qual ? "(ok)" : "(best-effort)");
    }

    // ---- 4-5. scale, filter, recover projection of the (deflated) matrix ----
    if (no_positive)
        CHECK_CUDA(cudaMemset(mat, 0, nn * sizeof(double)));
    else
        project_core(cublasH, mat, n, S, f, opts.precision);

    // ---- 6. add back the positive parts of the deflated eigenpairs ----------
    if (k_pos > 0 || k_neg > 0) {
        // Un-negate to lambda and clamp to max(lambda,0).  cublasDscal here takes
        // a HOST scalar (&minus_one) and max_dense_vector_zero is a custom kernel,
        // so this MUST run in the default HOST pointer mode -- switching to DEVICE
        // first would make cuBLAS read the scalar from a host address as device
        // memory (illegal access).  The DEVICE mode is set only for the rank-1
        // add-backs below, whose scalar (eval_* + i) is genuine device memory.
        if (k_pos > 0) {                                   // positive side: lambda >= 0
            CHECK_CUBLAS(cublasDscal(cublasH, k_pos, &minus_one, eval_max, 1));  // -> lambda
            max_dense_vector_zero(eval_max, k_pos);                             // max(lambda, 0)
        }
        if (k_neg > 0) {                                   // negative side: eig(-A) -> lambda(A)
            CHECK_CUBLAS(cublasDscal(cublasH, k_neg, &minus_one, eval_min, 1));  // -> lambda (negative)
            max_dense_vector_zero(eval_min, k_neg);                             // max(lambda, 0) = 0
        }
        cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_DEVICE);
        for (int i = 0; i < k_pos; ++i)
            CHECK_CUBLAS(cublasDger(cublasH, n, n, eval_max + i,
                                    evec_max + (size_t)i * n, 1,
                                    evec_max + (size_t)i * n, 1, mat, n));
        for (int i = 0; i < k_neg; ++i)
            CHECK_CUBLAS(cublasDger(cublasH, n, n, eval_min + i,
                                    evec_min + (size_t)i * n, 1,
                                    evec_min + (size_t)i * n, 1, mat, n));
        cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_HOST);
        CHECK_CUDA(cudaFree(evec_max)); CHECK_CUDA(cudaFree(eval_max));
        CHECK_CUDA(cudaFree(evec_min)); CHECK_CUDA(cudaFree(eval_min));
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    return rep;
}

