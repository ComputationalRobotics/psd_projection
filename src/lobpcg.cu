#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>

#include "psd_projection/lobpcg.h"
#include "psd_projection/utils.h"
#include "psd_projection/check.h"


// ---------------------------------------------------------------------------
// (C) Host cyclic-Jacobi symmetric eigensolver for the small Rayleigh-Ritz
// matrix.  A is nb x nb column-major (destroyed).  On return w holds the
// eigenvalues in ASCENDING order and V (nb x nb, column-major) their
// eigenvectors.  The caller only routes blocks with nb <= RR_GPU_NB (= 128) here
// -- larger ones go to cuSOLVER -- so the cost runs from microseconds on small
// blocks up to milliseconds at that cap.
// ---------------------------------------------------------------------------
static void jacobi_eigh(std::vector<double>& A, int nb,
                        std::vector<double>& w, std::vector<double>& V) {
    V.assign((size_t)nb * nb, 0.0);      // size the eigenvector output before indexing
    auto at = [&](int i, int j) -> double& { return A[(size_t)i + (size_t)j * nb]; };
    auto vt = [&](int i, int j) -> double& { return V[(size_t)i + (size_t)j * nb]; };
    for (int i = 0; i < nb; ++i) vt(i, i) = 1.0;

    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < nb; ++p)
            for (int q = p + 1; q < nb; ++q) off += at(p, q) * at(p, q);
        if (off < 1e-30) break;
        for (int p = 0; p < nb; ++p) {
            for (int q = p + 1; q < nb; ++q) {
                double apq = at(p, q);
                if (std::fabs(apq) < 1e-300) continue;
                double app = at(p, p), aqq = at(q, q);
                double theta = (aqq - app) / (2.0 * apq);
                double t = (theta >= 0 ? 1.0 : -1.0) /
                           (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                // rotate rows/cols p,q of A:  A <- J^T A J
                for (int i = 0; i < nb; ++i) {
                    double aip = at(i, p), aiq = at(i, q);
                    at(i, p) = c * aip - s * aiq;
                    at(i, q) = s * aip + c * aiq;
                }
                for (int i = 0; i < nb; ++i) {
                    double api = at(p, i), aqi = at(q, i);
                    at(p, i) = c * api - s * aqi;
                    at(q, i) = s * api + c * aqi;
                }
                for (int i = 0; i < nb; ++i) {
                    double vip = vt(i, p), viq = vt(i, q);
                    vt(i, p) = c * vip - s * viq;
                    vt(i, q) = s * vip + c * viq;
                }
            }
        }
    }
    w.resize(nb);
    for (int i = 0; i < nb; ++i) w[i] = at(i, i);
    // sort ascending, permuting eigenvector columns
    std::vector<int> idx(nb);
    for (int i = 0; i < nb; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return w[a] < w[b]; });
    std::vector<double> ws(nb), Vs((size_t)nb * nb);
    for (int j = 0; j < nb; ++j) {
        ws[j] = w[idx[j]];
        for (int i = 0; i < nb; ++i) Vs[(size_t)i + (size_t)j * nb] = V[(size_t)i + (size_t)idx[j] * nb];
    }
    w.swap(ws);
    V.swap(Vs);
}

void lobpcg(
    cublasHandle_t cublasH, cusolverDnHandle_t cusolverH,
    const double* A, double* V, double* D,
    const int n, const int m,
    const bool warmstart, const int maxiter, const double tol, const bool verbose,
    const bool relative_tol)
{
    const double one = 1.0, zero = 0.0, neg1 = -1.0;
    const int m3 = 3 * m;

    double *X, *AX, *Rr, *P, *Xnew, *AXnew, *tmp;   // n x m
    double *S, *W;                                   // n x 3m
    double *Td, *Yd, *Lam_d, *tau;
    CHECK_CUDA(cudaMalloc(&X,     (size_t)n * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&AX,    (size_t)n * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&Rr,    (size_t)n * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&P,     (size_t)n * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&Xnew,  (size_t)n * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&AXnew, (size_t)n * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&tmp,   (size_t)n * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&S,     (size_t)n * m3 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&W,     (size_t)n * m3 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&Td,    (size_t)m3 * m3 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&Yd,    (size_t)m3 * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&Lam_d,      (size_t)m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&tau,       (size_t)m3 * sizeof(double)));

    int *devInfo; CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));
    int lwg = 0, lwo = 0;
    CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(cusolverH, n, m3, S, n, &lwg));
    CHECK_CUSOLVER(cusolverDnDorgqr_bufferSize(cusolverH, n, m3, m3, S, n, tau, &lwo));
    int lwork = std::max(lwg, lwo);
    double* qrwork; CHECK_CUDA(cudaMalloc(&qrwork, (size_t)lwork * sizeof(double)));

    std::vector<double> h_T((size_t)m3 * m3), h_w, h_V, h_Y((size_t)m3 * m), h_lam(m);

    // Adaptive Rayleigh-Ritz: the RR block is 3m x 3m and is solved on the host
    // by cyclic Jacobi (O((3m)^3), microseconds when 3m ~ tens).  For large blocks
    // (many deflated eigenpairs) that host solve dominates, so switch to GPU
    // cuSOLVER syevd once the block exceeds RR_GPU_NB.  Buffers are only allocated
    // when the largest possible block (m3) can trigger the GPU path.
    const int RR_GPU_NB = 128;
    double *syev_eigs = nullptr, *syev_work = nullptr;
    int     syev_lwork = 0;
    std::vector<double> h_eig;
    if (m3 > RR_GPU_NB) {
        CHECK_CUDA(cudaMalloc(&syev_eigs, (size_t)m3 * sizeof(double)));
        CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(
            cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
            m3, Td, m3, syev_eigs, &syev_lwork));
        CHECK_CUDA(cudaMalloc(&syev_work, (size_t)syev_lwork * sizeof(double)));
    }

    // ---- initial block ------------------------------------------------------
    if (warmstart) {
        CHECK_CUDA(cudaMemcpy(X, V, (size_t)n * m * sizeof(double), cudaMemcpyDeviceToDevice));
    } else {
        double *d_work, *tau0; int lw0;
        CHECK_CUDA(cudaMalloc(&tau0, (size_t)m * sizeof(double)));
        CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(cusolverH, n, m, X, n, &lw0));
        CHECK_CUDA(cudaMalloc(&d_work, (size_t)lw0 * sizeof(double)));
        fill_random(X, n * m, 0);
        CHECK_CUSOLVER(cusolverDnDgeqrf(cusolverH, n, m, X, n, tau0, d_work, lw0, devInfo));
        CHECK_CUSOLVER(cusolverDnDorgqr(cusolverH, n, m, m, X, n, tau0, d_work, lw0, devInfo));
        CHECK_CUDA(cudaFree(d_work)); CHECK_CUDA(cudaFree(tau0));
    }

    // helper: RR eigensolve of an (nb x nb) block Td (ld nb); fill the top-m
    // eigenpairs (DESCENDING) into h_lam, Lam_d, and Yd (nb x m).  Small blocks
    // use the host Jacobi; large blocks use GPU cuSOLVER syevd.
    auto rr_topm = [&](int nb) {
        if (nb <= RR_GPU_NB) {                                // ---- host cyclic Jacobi ----
            CHECK_CUDA(cudaMemcpy(h_T.data(), Td, (size_t)nb * nb * sizeof(double), cudaMemcpyDeviceToHost));
            for (int i = 0; i < nb; ++i)                      // symmetrise on host
                for (int j = i + 1; j < nb; ++j) {
                    double a = 0.5 * (h_T[(size_t)i + (size_t)j * nb] + h_T[(size_t)j + (size_t)i * nb]);
                    h_T[(size_t)i + (size_t)j * nb] = a; h_T[(size_t)j + (size_t)i * nb] = a;
                }
            jacobi_eigh(h_T, nb, h_w, h_V);
            for (int j = 0; j < m; ++j) {
                int src = nb - 1 - j;                         // largest first
                h_lam[j] = h_w[src];
                for (int i = 0; i < nb; ++i) h_Y[(size_t)i + (size_t)j * nb] = h_V[(size_t)i + (size_t)src * nb];
            }
            CHECK_CUDA(cudaMemcpy(Yd, h_Y.data(), (size_t)nb * m * sizeof(double), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(Lam_d, h_lam.data(), (size_t)m * sizeof(double), cudaMemcpyHostToDevice));
        } else {                                              // ---- GPU cuSOLVER syevd ----
            // Td (packed nb x nb, ld nb) -> eigenvalues ascending in syev_eigs,
            // eigenvectors overwrite Td in place (lower triangle read).
            CHECK_CUSOLVER(cusolverDnDsyevd(
                cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                nb, Td, nb, syev_eigs, syev_work, syev_lwork, devInfo));
            // top-m = the m largest = last m columns, taken in descending order
            for (int j = 0; j < m; ++j)
                CHECK_CUDA(cudaMemcpy(Yd + (size_t)j * nb,
                                      Td + (size_t)(nb - 1 - j) * nb,
                                      (size_t)nb * sizeof(double), cudaMemcpyDeviceToDevice));
            h_eig.resize(nb);
            CHECK_CUDA(cudaMemcpy(h_eig.data(), syev_eigs, (size_t)nb * sizeof(double), cudaMemcpyDeviceToHost));
            for (int j = 0; j < m; ++j) h_lam[j] = h_eig[nb - 1 - j];   // host copy for lamnorm/output
            CHECK_CUDA(cudaMemcpy(Lam_d, h_lam.data(), (size_t)m * sizeof(double), cudaMemcpyHostToDevice));
        }
    };

    // AX = A X ; initial RR (block = X, nb = m) ; rotate X, AX to Ritz vectors
    CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, m, n, &one, A, n, X, n, &zero, AX, n));
    CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, m, m, n, &one, X, n, AX, n, &zero, Td, m));
    rr_topm(m);
    CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, m, m, &one, X,  n, Yd, m, &zero, Xnew,  n));
    CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, m, m, &one, AX, n, Yd, m, &zero, AXnew, n));
    CHECK_CUDA(cudaMemcpy(X,  Xnew,  (size_t)n * m * sizeof(double), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(AX, AXnew, (size_t)n * m * sizeof(double), cudaMemcpyDeviceToDevice));

    bool have_P = false;
    for (int iter = 1; iter <= maxiter; ++iter) {
        // R = AX - X diag(Lam)
        CHECK_CUDA(cudaMemcpy(Rr, AX, (size_t)n * m * sizeof(double), cudaMemcpyDeviceToDevice));
        CHECK_CUBLAS(cublasDdgmm(cublasH, CUBLAS_SIDE_RIGHT, n, m, X, n, Lam_d, 1, tmp, n));
        CHECK_CUBLAS(cublasDaxpy(cublasH, n * m, &neg1, tmp, 1, Rr, 1));

        double resnorm = 0.0;
        CHECK_CUBLAS(cublasDnrm2(cublasH, n * m, Rr, 1, &resnorm));   // host-pointer result: syncs
        double lamnorm = 0.0;
        for (int j = 0; j < m; ++j) lamnorm += h_lam[j] * h_lam[j];
        lamnorm = std::sqrt(lamnorm);
        if (verbose) std::printf("  [fast] iter %d  ||R||=%.3e  rel=%.3e\n",
                                 iter, resnorm, resnorm / std::max(lamnorm, 1.0));
        // Default is the ABSOLUTE test of the original interface (||R||_F <= tol);
        // relative_tol=true switches to ||R||_F <= tol * max(||D||_F, 1), which is
        // what the adaptive deflation path uses.
        const double res_thresh = relative_tol ? tol * std::max(lamnorm, 1.0) : tol;
        if (resnorm <= res_thresh) break;

        // build search block S = [X, R, (P)]
        int nb = 2 * m;
        CHECK_CUDA(cudaMemcpy(S,                  X,  (size_t)n * m * sizeof(double), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(S + (size_t)n * m,  Rr, (size_t)n * m * sizeof(double), cudaMemcpyDeviceToDevice));
        if (have_P) {
            CHECK_CUDA(cudaMemcpy(S + (size_t)2 * n * m, P, (size_t)n * m * sizeof(double), cudaMemcpyDeviceToDevice));
            nb = 3 * m;
        }

        // orthonormalise S -> Q  (no per-iteration info read)
        CHECK_CUSOLVER(cusolverDnDgeqrf(cusolverH, n, nb, S, n, tau, qrwork, lwork, devInfo));
        CHECK_CUSOLVER(cusolverDnDorgqr(cusolverH, n, nb, nb, S, n, tau, qrwork, lwork, devInfo));

        // (B) single A-multiply:  W = A Q ;  T = Q^T W
        CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, nb, n, &one, A, n, S, n, &zero, W, n));
        CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, nb, nb, n, &one, S, n, W, n, &zero, Td, nb));

        rr_topm(nb);   // host Jacobi (nb <= RR_GPU_NB) or GPU syevd -> Yd (nb x m), h_lam

        // X_{k+1} = Q Y ;  AX_{k+1} = W Y   (no second A-multiply)
        CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, m, nb, &one, S, n, Yd, nb, &zero, Xnew,  n));
        CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, m, nb, &one, W, n, Yd, nb, &zero, AXnew, n));

        // P = X_{k+1} - X_k
        CHECK_CUDA(cudaMemcpy(P, Xnew, (size_t)n * m * sizeof(double), cudaMemcpyDeviceToDevice));
        CHECK_CUBLAS(cublasDaxpy(cublasH, n * m, &neg1, X, 1, P, 1));
        have_P = true;

        CHECK_CUDA(cudaMemcpy(X,  Xnew,  (size_t)n * m * sizeof(double), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(AX, AXnew, (size_t)n * m * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    // outputs
    CHECK_CUBLAS(cublasDcopy(cublasH, n * m, X, 1, V, 1));
    CHECK_CUDA(cudaMemcpy(D, Lam_d, (size_t)m * sizeof(double), cudaMemcpyDeviceToDevice));

    // one deferred status read of the LAST cuSOLVER call (the QR, or the RR syevd
    // when nb > RR_GPU_NB); earlier iterations' statuses are overwritten, and a
    // failure is only printed under verbose -- never returned to the caller
    int hInfo = 0; CHECK_CUDA(cudaMemcpy(&hInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (hInfo != 0 && verbose) std::fprintf(stderr, "lobpcg: last QR devInfo=%d\n", hInfo);

    CHECK_CUDA(cudaFree(X)); CHECK_CUDA(cudaFree(AX)); CHECK_CUDA(cudaFree(Rr)); CHECK_CUDA(cudaFree(P));
    CHECK_CUDA(cudaFree(Xnew)); CHECK_CUDA(cudaFree(AXnew)); CHECK_CUDA(cudaFree(tmp));
    CHECK_CUDA(cudaFree(S)); CHECK_CUDA(cudaFree(W)); CHECK_CUDA(cudaFree(Td)); CHECK_CUDA(cudaFree(Yd));
    CHECK_CUDA(cudaFree(Lam_d)); CHECK_CUDA(cudaFree(tau)); CHECK_CUDA(cudaFree(qrwork));
    CHECK_CUDA(cudaFree(devInfo));
    if (syev_eigs) CHECK_CUDA(cudaFree(syev_eigs));
    if (syev_work) CHECK_CUDA(cudaFree(syev_work));
}

