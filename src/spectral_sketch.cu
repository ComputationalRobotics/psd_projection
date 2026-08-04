#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand_kernel.h>
#include <vector>
#include <algorithm>
#include <cmath>

#include "psd_projection/spectral_sketch.h"
#include "psd_projection/lanczos.h"
#include "psd_projection/utils.h"
#include "psd_projection/check.h"


// Fill a device vector with Rademacher (+/-1) entries.
__global__ void fill_rademacher_kernel(double* v, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandStatePhilox4_32_10_t st;
        curand_init(seed, idx, 0, &st);
        v[idx] = (curand_uniform_double(&st) < 0.5) ? -1.0 : 1.0;
    }
}
__global__ void fill_rademacher_float_kernel(float* v, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandStatePhilox4_32_10_t st;
        curand_init(seed, idx, 0, &st);
        v[idx] = (curand_uniform(&st) < 0.5f) ? -1.0f : 1.0f;
    }
}

// --- Batched column kernels for the blocked Lanczos (one launch handles all p
//     probes, and per-probe scalars alpha/beta stay on the device so the m-step
//     loop needs ZERO per-step host synchronisation).  ALPHA/BETA are p*m arrays,
//     probe pr's scalar for step k at index pr*m+k.  All FP64. -------------------
// alpha[col*m+k] = dot(A[:,col], B[:,col])   (one block per column, shared-mem reduce)
__global__ void colblk_dot_kernel(const double* A, const double* B, int n, int p,
                           double* ALPHA, int m, int k) {
    int col = blockIdx.x; if (col >= p) return;
    const double* a = A + (size_t)col * n; const double* b = B + (size_t)col * n;
    __shared__ double sh[256];
    double acc = 0.0; for (int i = threadIdx.x; i < n; i += blockDim.x) acc += a[i] * b[i];
    sh[threadIdx.x] = acc; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s]; __syncthreads(); }
    if (threadIdx.x == 0) ALPHA[(size_t)col * m + k] = sh[0];
}
// beta[col*m+k] = ||W[:,col]||   (one block per column)
__global__ void colblk_norm_kernel(const double* W, int n, int p, double* BETA, int m, int k) {
    int col = blockIdx.x; if (col >= p) return;
    const double* w = W + (size_t)col * n;
    __shared__ double sh[256];
    double acc = 0.0; for (int i = threadIdx.x; i < n; i += blockDim.x) { double v = w[i]; acc += v * v; }
    sh[threadIdx.x] = acc; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s]; __syncthreads(); }
    if (threadIdx.x == 0) BETA[(size_t)col * m + k] = sqrt(sh[0]);
}
// W[:,col] -= alpha_k[col]*Qk[:,col] + beta_{k-1}[col]*Qkm1[:,col]   (grid over n*p)
__global__ void colblk_sub_kernel(double* W, const double* Qk, const double* Qkm1,
                           const double* ALPHA, const double* BETA, int n, int p, int m, int k) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; size_t tot = (size_t)n * p; if (idx >= tot) return;
    int col = (int)(idx / n);
    double a = ALPHA[(size_t)col * m + k];
    double bprev = (k > 0) ? BETA[(size_t)col * m + (k - 1)] : 0.0;
    W[idx] -= a * Qk[idx] + bprev * Qkm1[idx];
}
// Qk_new[:,col] = W[:,col] / beta_k[col]   (guarded: beta~0 breakdown -> 0, no NaN)
__global__ void colblk_normalize_into_kernel(double* dst, const double* W, const double* BETA,
                                      int n, int p, int m, int k) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; size_t tot = (size_t)n * p; if (idx >= tot) return;
    int col = (int)(idx / n); double b = BETA[(size_t)col * m + k];
    dst[idx] = W[idx] * ((b > 1e-30) ? (1.0 / b) : 0.0);
}

// One fully-reorthogonalised m-step Lanczos probe in FP64.  Fills the host
// tridiagonal diagonals ha[0..m_eff-1] / hb[0..m_eff-2]; returns m_eff.
static int lanczos_probe_f64(
    cublasHandle_t H, const double* A, int n, int m,
    double* Q, double* w, double* qk, double* qkm1, double* proj,
    unsigned long long seed_p, std::vector<double>& ha, std::vector<double>& hb)
{
    const double one = 1.0, zero = 0.0;
    int tpb = 256, blk = (n + tpb - 1) / tpb;
    fill_rademacher_kernel<<<blk, tpb>>>(qk, n, seed_p);
    CHECK_CUDA(cudaGetLastError());
    double nrm; CHECK_CUBLAS(cublasDnrm2(H, n, qk, 1, &nrm));
    double inv = (nrm > 0.0) ? 1.0 / nrm : 1.0;
    CHECK_CUBLAS(cublasDscal(H, n, &inv, qk, 1));
    CHECK_CUDA(cudaMemset(qkm1, 0, (size_t)n * sizeof(double)));
    double beta_prev = 0.0; int m_eff = m;
    for (int k = 0; k < m; ++k) {
        CHECK_CUBLAS(cublasDcopy(H, n, qk, 1, Q + (size_t)k * n, 1));
        CHECK_CUBLAS(cublasDgemv(H, CUBLAS_OP_N, n, n, &one, A, n, qk, 1, &zero, w, 1));
        double alpha; CHECK_CUBLAS(cublasDdot(H, n, qk, 1, w, 1, &alpha)); ha[k] = alpha;
        double m_alpha = -alpha, m_beta = -beta_prev;
        CHECK_CUBLAS(cublasDaxpy(H, n, &m_alpha, qk, 1, w, 1));
        CHECK_CUBLAS(cublasDaxpy(H, n, &m_beta, qkm1, 1, w, 1));
        for (int pass = 0; pass < 2; ++pass) {
            CHECK_CUBLAS(cublasDgemv(H, CUBLAS_OP_T, n, k + 1, &one, Q, n, w, 1, &zero, proj, 1));
            double neg = -1.0;
            CHECK_CUBLAS(cublasDgemv(H, CUBLAS_OP_N, n, k + 1, &neg, Q, n, proj, 1, &one, w, 1));
        }
        double beta; CHECK_CUBLAS(cublasDnrm2(H, n, w, 1, &beta));
        if (k < m - 1) hb[k] = beta;
        if (beta < 1e-14) { m_eff = k + 1; break; }
        CHECK_CUBLAS(cublasDcopy(H, n, qk, 1, qkm1, 1));
        double invb = 1.0 / beta;
        CHECK_CUBLAS(cublasDcopy(H, n, w, 1, qk, 1));
        CHECK_CUBLAS(cublasDscal(H, n, &invb, qk, 1));
        beta_prev = beta;
    }
    return m_eff;
}

// Same probe in FP32: the matvec reads a single-precision copy of A (half the
// HBM traffic of the FP64 path), reductions accumulate in FP32.  The scale is
// still computed in FP64 by the caller, so the composite filter's
// [-1,1] guarantee is unaffected; only the quadrature nodes (which feed the
// T-selector and outlier counts, both tolerant of ~1e-6 error) use FP32.
static int lanczos_probe_f32(
    cublasHandle_t H, const float* A, int n, int m,
    float* Q, float* w, float* qk, float* qkm1, float* proj,
    unsigned long long seed_p, std::vector<double>& ha, std::vector<double>& hb)
{
    const float one = 1.0f, zero = 0.0f;
    int tpb = 256, blk = (n + tpb - 1) / tpb;
    fill_rademacher_float_kernel<<<blk, tpb>>>(qk, n, seed_p);
    CHECK_CUDA(cudaGetLastError());
    float nrm; CHECK_CUBLAS(cublasSnrm2(H, n, qk, 1, &nrm));
    float inv = (nrm > 0.0f) ? 1.0f / nrm : 1.0f;
    CHECK_CUBLAS(cublasSscal(H, n, &inv, qk, 1));
    CHECK_CUDA(cudaMemset(qkm1, 0, (size_t)n * sizeof(float)));
    float beta_prev = 0.0f; int m_eff = m;
    for (int k = 0; k < m; ++k) {
        CHECK_CUBLAS(cublasScopy(H, n, qk, 1, Q + (size_t)k * n, 1));
        CHECK_CUBLAS(cublasSgemv(H, CUBLAS_OP_N, n, n, &one, A, n, qk, 1, &zero, w, 1));
        float alpha; CHECK_CUBLAS(cublasSdot(H, n, qk, 1, w, 1, &alpha)); ha[k] = (double)alpha;
        float m_alpha = -alpha, m_beta = -beta_prev;
        CHECK_CUBLAS(cublasSaxpy(H, n, &m_alpha, qk, 1, w, 1));
        CHECK_CUBLAS(cublasSaxpy(H, n, &m_beta, qkm1, 1, w, 1));
        for (int pass = 0; pass < 2; ++pass) {
            CHECK_CUBLAS(cublasSgemv(H, CUBLAS_OP_T, n, k + 1, &one, Q, n, w, 1, &zero, proj, 1));
            float neg = -1.0f;
            CHECK_CUBLAS(cublasSgemv(H, CUBLAS_OP_N, n, k + 1, &neg, Q, n, proj, 1, &one, w, 1));
        }
        float beta; CHECK_CUBLAS(cublasSnrm2(H, n, w, 1, &beta));
        if (k < m - 1) hb[k] = (double)beta;
        if (beta < 1e-12f) { m_eff = k + 1; break; }
        CHECK_CUBLAS(cublasScopy(H, n, qk, 1, qkm1, 1));
        float invb = 1.0f / beta;
        CHECK_CUBLAS(cublasScopy(H, n, w, 1, qk, 1));
        CHECK_CUBLAS(cublasSscal(H, n, &invb, qk, 1));
        beta_prev = beta;
    }
    return m_eff;
}

// Blocked (lockstep) FP64 Lanczos.  All `p` probes advance together, so step k's
// matvec is a SINGLE GEMM  W = A * Qk  (Qk is n x p): A is read once for all p
// probes instead of once per probe -- ~p x less HBM traffic than the sequential
// path.  Reorthogonalisation (two passes) is issued as a single strided-batched
// GEMV over all probes per direction, so its launch count is O(1) per step instead
// of O(p); the math is identical to p independent single-vector Lanczos runs, so
// the quadrature nodes/weights are unchanged.  Also returns each probe's final
// residual norm beta_{m_eff} in `beta_last`, used to certify a bound on ||A||_2.
//   Qhist layout: n x (p*m), probe-major -- probe `pr` owns columns [pr*m,(pr+1)*m).
//   ha/hb: size p*m, probe `pr` at [pr*m .. pr*m+m).  m_eff: size p.
static void lanczos_block(
    cublasHandle_t H, const double* Ad, int n, int m, int p,
    double* Qk, double* Qkm1, double* W, double* Qhist, double* proj,
    unsigned long long seed,
    std::vector<double>& ha, std::vector<double>& hb, std::vector<int>& m_eff,
    std::vector<double>& beta_last)
{
    const double one = 1.0, zero = 0.0, neg = -1.0;
    int tpb = 256, blk = (n + tpb - 1) / tpb;
    beta_last.assign(p, 0.0);   // residual norm beta_{m_eff} of each probe (for the upper bound)
    // random Rademacher start block (per-column seed), normalise each column
    for (int pr = 0; pr < p; ++pr) {
        fill_rademacher_kernel<<<blk, tpb>>>(Qk + (size_t)pr * n, n,
                                             seed + 0x9E3779B97F4A7C15ULL * (pr + 1));
        CHECK_CUDA(cudaGetLastError());
        double nrm; CHECK_CUBLAS(cublasDnrm2(H, n, Qk + (size_t)pr * n, 1, &nrm));
        double inv = (nrm > 0.0) ? 1.0 / nrm : 1.0;
        CHECK_CUBLAS(cublasDscal(H, n, &inv, Qk + (size_t)pr * n, 1));
    }
    CHECK_CUDA(cudaMemset(Qkm1, 0, (size_t)n * p * sizeof(double)));
    m_eff.assign(p, m);

    // Per-probe alpha/beta live on the DEVICE (indices pr*m+k) so the m-step loop
    // needs ZERO per-step host sync.  Phases 1 & 3 (formerly p x {dot,axpy,axpy} and
    // p x nrm2 host-scalar cuBLAS calls) become one batched column kernel each.
    double *dALPHA = nullptr, *dBETA = nullptr;
    CHECK_CUDA(cudaMalloc(&dALPHA, (size_t)p * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dBETA,  (size_t)p * m * sizeof(double)));
    const int    tpb2  = 256;
    const size_t blkNP = ((size_t)n * p + tpb2 - 1) / tpb2;

    // probe-major strides for the strided-batched reorthogonalisation
    const long long sQh = (long long)m * n;   // one probe's history block
    const long long sW  = (long long)n;       // one probe's current vector (column of W)
    const long long sPj = (long long)m;        // one probe's projection scratch

    for (int k = 0; k < m; ++k) {
        // stash the current block columns into each probe's history (single strided copy)
        CHECK_CUDA(cudaMemcpy2D(Qhist + (size_t)k * n, (size_t)m * n * sizeof(double),
                                Qk, (size_t)n * sizeof(double),
                                (size_t)n * sizeof(double), p, cudaMemcpyDeviceToDevice));
        // W = A * Qk   (single BLAS-3 GEMM -> reads A once for all probes)
        CHECK_CUBLAS(cublasDgemm(H, CUBLAS_OP_N, CUBLAS_OP_N, n, p, n,
                                 &one, Ad, n, Qk, n, &zero, W, n));
        // phase 1: alpha_k = colwise dot(Qk, W)  ;  w -= alpha_k qk + beta_{k-1} qkm1
        colblk_dot_kernel<<<p, 256>>>(Qk, W, n, p, dALPHA, m, k);
        colblk_sub_kernel<<<(unsigned)blkNP, tpb2>>>(W, Qk, Qkm1, dALPHA, dBETA, n, p, m, k);
        // phase 2: reorthogonalise (two batched passes)
        for (int pass = 0; pass < 2; ++pass) {
            CHECK_CUBLAS(cublasDgemvStridedBatched(H, CUBLAS_OP_T, n, k + 1, &one,
                Qhist, n, sQh, W, 1, sW, &zero, proj, 1, sPj, p));
            CHECK_CUBLAS(cublasDgemvStridedBatched(H, CUBLAS_OP_N, n, k + 1, &neg,
                Qhist, n, sQh, proj, 1, sPj, &one, W, 1, sW, p));
        }
        // phase 3: beta_k = colwise ||w||  ;  qkm1 <- qk ; qk <- w / beta_k (guarded)
        colblk_norm_kernel<<<p, 256>>>(W, n, p, dBETA, m, k);
        if (k < m - 1) {
            CHECK_CUDA(cudaMemcpy(Qkm1, Qk, (size_t)n * p * sizeof(double), cudaMemcpyDeviceToDevice));
            colblk_normalize_into_kernel<<<(unsigned)blkNP, tpb2>>>(Qk, W, dBETA, n, p, m, k);
        }
    }
    CHECK_CUDA(cudaGetLastError());

    // single D2H of the whole (alpha,beta) tridiagonal, then host-side breakdown
    // detection: the first beta < 1e-12 truncates that probe's Krylov space at that
    // step.  Everything past the truncation is simply DISCARDED -- note that
    // colblk_normalize_into_kernel only zeroes a column at beta <= 1e-30, so for beta
    // in (1e-30, 1e-12) it still divides by the tiny beta and the resulting blown-up
    // columns live on in Qk/Qhist; m_eff keeps them out of the quadrature, and the
    // probe-major layout keeps them out of the other probes.
    std::vector<double> aAll((size_t)p * m), bAll((size_t)p * m);
    CHECK_CUDA(cudaMemcpy(aAll.data(), dALPHA, (size_t)p * m * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(bAll.data(), dBETA,  (size_t)p * m * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(dALPHA)); CHECK_CUDA(cudaFree(dBETA));
    for (int pr = 0; pr < p; ++pr) {
        for (int k = 0; k < m; ++k) {
            ha[(size_t)pr * m + k] = aAll[(size_t)pr * m + k];
            if (k < m - 1) hb[(size_t)pr * m + k] = bAll[(size_t)pr * m + k];
        }
        int me = m;
        for (int k = 0; k < m; ++k)
            if (bAll[(size_t)pr * m + k] < 1e-12) { me = k + 1; break; }
        m_eff[pr]    = me;
        beta_last[pr] = bAll[(size_t)pr * m + (me - 1)];
    }
}

SpectralSketch compute_spectral_sketch(
    cublasHandle_t cublasH,
    cusolverDnHandle_t cusolverH,
    const double* A, int n,
    int m, int n_probes,
    unsigned long long seed,
    bool use_fp32,
    bool use_block
) {
    SpectralSketch S;
    S.n = n;
    S.n_probes = n_probes;


    // ---- scale (upper bound on ||A||_2) and exact Frobenius norm ----
    // The block path derives the scale from the sketch's OWN Lanczos data below -- the
    // extremal Ritz value plus its a-posteriori residual bound (symmetric A =>
    // ||A||_2 = max|eigenvalue|) -- which is far tighter than a separate FP64 two-norm
    // pass would give and costs no extra matvec.  Only the sequential paths (the ones
    // this function's own use_block=false default selects, though AdaptiveOptions sets
    // sketch_block=true) run approximate_two_norm.
    if (!use_block) {
        double lo = 0.0, up = 1.0;
        approximate_two_norm(cublasH, cusolverH, A, (size_t)n, &lo, &up);
        // Small relative margin: `up` is an a-posteriori Ritz bound that can slightly
        // under-estimate ||A||_2 for near-degenerate dominant spectra.  Since the
        // composite quintic filter diverges outside [-1,1], we pad the scale so the
        // scaled spectrum stays strictly inside (project_core also clamps as a guard).
        S.scale = (up > 0.0) ? up * 1.01 : 1.0;
    }

    double fro = 0.0;
    CHECK_CUBLAS(cublasDnrm2(cublasH, (size_t)n * n, A, 1, &fro));
    S.fro_sq = fro * fro;

    if (m < 2) m = 2;
    if (m > n) m = n;

    // ---- tridiagonal eigensolve workspace (always FP64; the m x m tridiagonal
    //      is tiny) ----
    double *dT, *dEvals;
    CHECK_CUDA(cudaMalloc(&dT,     (size_t)m * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dEvals,          (size_t)m * sizeof(double)));
    int lwork = 0, *devInfo;
    CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));
    CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(
        cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
        m, dT, m, dEvals, &lwork));
    double* dWork;
    CHECK_CUDA(cudaMalloc(&dWork, (size_t)lwork * sizeof(double)));

    std::vector<double> h_T(m * m), h_Z(m * m), h_evals(m);
    double gmin = 1e300, gmax = -1e300;

    // Turn one probe's tridiagonal diagonals (alpha[0..m_eff-1], beta[0..m_eff-2])
    // into quadrature nodes theta_j = eval_j/scale and weights tau_j = Z(0,j)^2.
    auto process_tridiag = [&](const double* alpha, const double* beta, int m_eff) {
        std::fill(h_T.begin(), h_T.begin() + (size_t)m_eff * m_eff, 0.0);
        for (int i = 0; i < m_eff; ++i) {
            h_T[(size_t)i * m_eff + i] = alpha[i];
            if (i + 1 < m_eff) {
                h_T[(size_t)i * m_eff + (i + 1)] = beta[i];
                h_T[(size_t)(i + 1) * m_eff + i] = beta[i];
            }
        }
        CHECK_CUDA(cudaMemcpy(dT, h_T.data(), (size_t)m_eff * m_eff * sizeof(double),
                              cudaMemcpyHostToDevice));
        CHECK_CUSOLVER(cusolverDnDsyevd(
            cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
            m_eff, dT, m_eff, dEvals, dWork, lwork, devInfo));
        CHECK_CUDA(cudaMemcpy(h_evals.data(), dEvals, (size_t)m_eff * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_Z.data(), dT, (size_t)m_eff * m_eff * sizeof(double),
                              cudaMemcpyDeviceToHost));
        for (int j = 0; j < m_eff; ++j) {
            double th = h_evals[j] / S.scale;
            if (th > 1.0) th = 1.0;
            if (th < -1.0) th = -1.0;
            double z0 = h_Z[(size_t)j * m_eff + 0];
            S.theta.push_back(th);
            S.tau.push_back(z0 * z0);
            gmin = std::min(gmin, h_evals[j]);
            gmax = std::max(gmax, h_evals[j]);
        }
    };

    // ---- Lanczos: blocked FP64, or per-probe (sequential) FP32/FP64 ----
    // The blocked path runs in FP64 and shares one A-read across all probes via a
    // BLAS-3 GEMM.  The per-probe path optionally stores a single-precision copy of
    // A (use_fp32) to halve matvec bytes.  The scale (computed above) is always
    // FP64, so the composite's [-1,1] domain is unaffected.
    double *Qd = nullptr, *wd = nullptr, *qkd = nullptr, *qkm1d = nullptr, *projd = nullptr;
    float  *Af = nullptr, *Qf = nullptr, *wf = nullptr, *qkf = nullptr, *qkm1f = nullptr, *projf = nullptr;
    double *Qkb = nullptr, *Qkm1b = nullptr, *Wb = nullptr, *Qhistb = nullptr, *projb = nullptr;

    if (use_block) {
        // n*n fits in int for n <= 46340 (benchmark sizes are <= 20000)
        std::vector<double> ha((size_t)n_probes * m, 0.0), hb((size_t)n_probes * m, 0.0);
        std::vector<int>    m_eff;
        std::vector<double> beta_last;
        CHECK_CUDA(cudaMalloc(&Qkb,    (size_t)n * n_probes * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&Qkm1b,  (size_t)n * n_probes * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&Wb,     (size_t)n * n_probes * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&Qhistb, (size_t)n * n_probes * m * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&projb,  (size_t)m * n_probes * sizeof(double)));
        lanczos_block(cublasH, A, n, m, n_probes, Qkb, Qkm1b, Wb, Qhistb, projb,
                      seed, ha, hb, m_eff, beta_last);

        // Eigensolve each probe's tridiagonal to RAW (unscaled) Ritz values + weights,
        // tracking the global spectral extremes and the a-posteriori norm bound:
        //   z0 = first eigenvector component -> quadrature weight tau
        //   zL = last  eigenvector component -> Lanczos residual  ||r_j|| = beta_last*|zL|
        // A true eigenvalue of A lies within ||r_j|| of each Ritz value e_j, so
        //   max_j ( |e_j| + ||r_j|| )   bounds the part of the spectrum the Krylov
        // spaces resolved.  This is an A-POSTERIORI bound, not a proof that it exceeds
        // ||A||_2 (Lanczos could in principle miss the dominant eigenvalue), but it is
        // tight in practice, costs no extra matvec (symmetric A), and the 1.001 pad plus
        // the [-1,1] clamps below absorb the slack.
        std::vector<double> rawE, rawW;
        rawE.reserve((size_t)n_probes * m); rawW.reserve((size_t)n_probes * m);
        double up_resid = 0.0;
        for (int pr = 0; pr < n_probes; ++pr) {
            const double* alpha = ha.data() + (size_t)pr * m;
            const double* beta  = hb.data() + (size_t)pr * m;
            int me = m_eff[pr];
            std::fill(h_T.begin(), h_T.begin() + (size_t)me * me, 0.0);
            for (int i = 0; i < me; ++i) {
                h_T[(size_t)i * me + i] = alpha[i];
                if (i + 1 < me) { h_T[(size_t)i*me+(i+1)] = beta[i]; h_T[(size_t)(i+1)*me+i] = beta[i]; }
            }
            CHECK_CUDA(cudaMemcpy(dT, h_T.data(), (size_t)me*me*sizeof(double), cudaMemcpyHostToDevice));
            CHECK_CUSOLVER(cusolverDnDsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
                CUBLAS_FILL_MODE_UPPER, me, dT, me, dEvals, dWork, lwork, devInfo));
            CHECK_CUDA(cudaMemcpy(h_evals.data(), dEvals, (size_t)me*sizeof(double), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_Z.data(), dT, (size_t)me*me*sizeof(double), cudaMemcpyDeviceToHost));
            for (int j = 0; j < me; ++j) {
                double e = h_evals[j], z0 = h_Z[(size_t)j*me + 0], zL = h_Z[(size_t)j*me + (me-1)];
                rawE.push_back(e); rawW.push_back(z0 * z0);
                gmin = std::min(gmin, e); gmax = std::max(gmax, e);
                up_resid = std::max(up_resid, std::fabs(e) + beta_last[pr] * std::fabs(zL));
            }
        }
        // tight scale from the residual bound (small pad for FP safety)
        S.scale = (up_resid > 0.0) ? up_resid * 1.001 : 1.0;
        for (size_t j = 0; j < rawE.size(); ++j) {
            double th = rawE[j] / S.scale;
            if (th > 1.0) th = 1.0; if (th < -1.0) th = -1.0;
            S.theta.push_back(th); S.tau.push_back(rawW[j]);
        }
    } else {
        std::vector<double> h_alpha(m), h_beta(m);
        if (use_fp32) {
            CHECK_CUDA(cudaMalloc(&Af, (size_t)n * n * sizeof(float)));
            convert_double_to_float(A, Af, (int)((size_t)n * n));
            CHECK_CUDA(cudaMalloc(&Qf,    (size_t)n * m * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&wf,             (size_t)n * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&qkf,            (size_t)n * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&qkm1f,          (size_t)n * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&projf,          (size_t)m * sizeof(float)));
        } else {
            CHECK_CUDA(cudaMalloc(&Qd,    (size_t)n * m * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&wd,             (size_t)n * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&qkd,            (size_t)n * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&qkm1d,          (size_t)n * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&projd,          (size_t)m * sizeof(double)));
        }
        for (int p = 0; p < n_probes; ++p) {
            unsigned long long sp = seed + 0x9E3779B97F4A7C15ULL * (p + 1);
            int m_eff = use_fp32
                ? lanczos_probe_f32(cublasH, Af, n, m, Qf, wf, qkf, qkm1f, projf, sp, h_alpha, h_beta)
                : lanczos_probe_f64(cublasH, A,  n, m, Qd, wd, qkd, qkm1d, projd, sp, h_alpha, h_beta);
            process_tridiag(h_alpha.data(), h_beta.data(), m_eff);
        }
    }

    S.lam_min_scaled = std::max(-1.0, gmin / S.scale);
    S.lam_max_scaled = std::min( 1.0, gmax / S.scale);

    if (Qkb)    CHECK_CUDA(cudaFree(Qkb));
    if (Qkm1b)  CHECK_CUDA(cudaFree(Qkm1b));
    if (Wb)     CHECK_CUDA(cudaFree(Wb));
    if (Qhistb) CHECK_CUDA(cudaFree(Qhistb));
    if (projb)  CHECK_CUDA(cudaFree(projb));
    if (Af)    CHECK_CUDA(cudaFree(Af));
    if (Qf)    CHECK_CUDA(cudaFree(Qf));
    if (wf)    CHECK_CUDA(cudaFree(wf));
    if (qkf)   CHECK_CUDA(cudaFree(qkf));
    if (qkm1f) CHECK_CUDA(cudaFree(qkm1f));
    if (projf) CHECK_CUDA(cudaFree(projf));
    if (Qd)    CHECK_CUDA(cudaFree(Qd));
    if (wd)    CHECK_CUDA(cudaFree(wd));
    if (qkd)   CHECK_CUDA(cudaFree(qkd));
    if (qkm1d) CHECK_CUDA(cudaFree(qkm1d));
    if (projd) CHECK_CUDA(cudaFree(projd));
    CHECK_CUDA(cudaFree(dT));
    CHECK_CUDA(cudaFree(dEvals));
    CHECK_CUDA(cudaFree(devInfo));
    CHECK_CUDA(cudaFree(dWork));
    CHECK_CUDA(cudaDeviceSynchronize());

    return S;
}

// ---------------------------------------------------------------------------
// (D) Extremal Ritz pairs from one reorthogonalised Lanczos run -- warm-start
// for LOBPCG deflation.
// ---------------------------------------------------------------------------
void compute_extremal_ritz(
    cublasHandle_t cublasH, cusolverDnHandle_t cusolverH,
    const double* A, int n, int k, int m,
    double* evecs_max, double* evals_max,
    double* evecs_min, double* evals_min,
    unsigned long long seed)
{
    const double one = 1.0, zero = 0.0;
    if (m < 2 * k + 4) m = 2 * k + 4;
    if (m > n) m = n;

    double *Q, *w, *qk, *qkm1, *proj;
    CHECK_CUDA(cudaMalloc(&Q,    (size_t)n * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&w,             (size_t)n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&qk,            (size_t)n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&qkm1,          (size_t)n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&proj,          (size_t)m * sizeof(double)));

    int tpb = 256, blk = (n + tpb - 1) / tpb;
    fill_rademacher_kernel<<<blk, tpb>>>(qk, n, seed);
    CHECK_CUDA(cudaGetLastError());
    double nrm; CHECK_CUBLAS(cublasDnrm2(cublasH, n, qk, 1, &nrm));
    double inv = (nrm > 0) ? 1.0 / nrm : 1.0;
    CHECK_CUBLAS(cublasDscal(cublasH, n, &inv, qk, 1));
    CHECK_CUDA(cudaMemset(qkm1, 0, (size_t)n * sizeof(double)));

    std::vector<double> h_alpha(m, 0.0), h_beta(m, 0.0);
    double beta_prev = 0.0;
    int m_eff = m;
    for (int j = 0; j < m; ++j) {
        CHECK_CUBLAS(cublasDcopy(cublasH, n, qk, 1, Q + (size_t)j * n, 1));
        CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_N, n, n, &one, A, n, qk, 1, &zero, w, 1));
        double alpha; CHECK_CUBLAS(cublasDdot(cublasH, n, qk, 1, w, 1, &alpha));
        h_alpha[j] = alpha;
        double ma = -alpha, mb = -beta_prev;
        CHECK_CUBLAS(cublasDaxpy(cublasH, n, &ma, qk, 1, w, 1));
        CHECK_CUBLAS(cublasDaxpy(cublasH, n, &mb, qkm1, 1, w, 1));
        for (int pass = 0; pass < 2; ++pass) {               // full reorthogonalisation
            CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_T, n, j + 1, &one, Q, n, w, 1, &zero, proj, 1));
            double neg = -1.0;
            CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_N, n, j + 1, &neg, Q, n, proj, 1, &one, w, 1));
        }
        double beta; CHECK_CUBLAS(cublasDnrm2(cublasH, n, w, 1, &beta));
        if (j < m - 1) h_beta[j] = beta;
        if (beta < 1e-14) { m_eff = j + 1; break; }
        CHECK_CUBLAS(cublasDcopy(cublasH, n, qk, 1, qkm1, 1));
        double ib = 1.0 / beta;
        CHECK_CUBLAS(cublasDcopy(cublasH, n, w, 1, qk, 1));
        CHECK_CUBLAS(cublasDscal(cublasH, n, &ib, qk, 1));
        beta_prev = beta;
    }
    if (k > m_eff) k = m_eff;   // degenerate safety

    // tridiagonal eigen-decomposition
    std::vector<double> h_T((size_t)m_eff * m_eff, 0.0);
    for (int i = 0; i < m_eff; ++i) {
        h_T[(size_t)i * m_eff + i] = h_alpha[i];
        if (i + 1 < m_eff) { h_T[(size_t)i * m_eff + i + 1] = h_beta[i];
                             h_T[(size_t)(i + 1) * m_eff + i] = h_beta[i]; }
    }
    double *dT, *dEvals; CHECK_CUDA(cudaMalloc(&dT, (size_t)m_eff * m_eff * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dEvals, (size_t)m_eff * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(dT, h_T.data(), (size_t)m_eff * m_eff * sizeof(double), cudaMemcpyHostToDevice));
    int lwork = 0, *info; CHECK_CUDA(cudaMalloc(&info, sizeof(int)));
    CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
                   CUBLAS_FILL_MODE_UPPER, m_eff, dT, m_eff, dEvals, &lwork));
    double* dwork; CHECK_CUDA(cudaMalloc(&dwork, (size_t)lwork * sizeof(double)));
    CHECK_CUSOLVER(cusolverDnDsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                   m_eff, dT, m_eff, dEvals, dwork, lwork, info));   // ascending; dT = eigenvectors

    // gather the k largest eigenvector columns in DESCENDING order into Zmax
    double* Zmax; CHECK_CUDA(cudaMalloc(&Zmax, (size_t)m_eff * k * sizeof(double)));
    for (int j = 0; j < k; ++j)
        CHECK_CUBLAS(cublasDcopy(cublasH, m_eff, dT + (size_t)(m_eff - 1 - j) * m_eff, 1,
                                 Zmax + (size_t)j * m_eff, 1));
    // Ritz vectors = Q * Z
    CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, k, m_eff,
                             &one, Q, n, Zmax, m_eff, &zero, evecs_max, n));
    CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, k, m_eff,
                             &one, Q, n, dT, m_eff, &zero, evecs_min, n));   // first k cols = smallest
    // Ritz values
    CHECK_CUDA(cudaMemcpy(evals_min, dEvals, (size_t)k * sizeof(double), cudaMemcpyDeviceToDevice));
    for (int j = 0; j < k; ++j)
        CHECK_CUDA(cudaMemcpy(evals_max + j, dEvals + (m_eff - 1 - j), sizeof(double),
                              cudaMemcpyDeviceToDevice));

    CHECK_CUDA(cudaFree(Q)); CHECK_CUDA(cudaFree(w)); CHECK_CUDA(cudaFree(qk));
    CHECK_CUDA(cudaFree(qkm1)); CHECK_CUDA(cudaFree(proj)); CHECK_CUDA(cudaFree(dT));
    CHECK_CUDA(cudaFree(dEvals)); CHECK_CUDA(cudaFree(dwork)); CHECK_CUDA(cudaFree(info));
    CHECK_CUDA(cudaFree(Zmax));
    CHECK_CUDA(cudaDeviceSynchronize());
}

