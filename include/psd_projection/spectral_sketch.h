#ifndef PSD_PROJECTION_SPECTRAL_SKETCH_H
#define PSD_PROJECTION_SPECTRAL_SKETCH_H

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <cmath>


/// @brief Cheap spectral sketch of a symmetric matrix, obtained with
/// matrix-vector products only (Lanczos + stochastic Lanczos quadrature).
///
/// The sketch stores a Gaussian-quadrature representation of the spectral
/// density: a set of nodes (scaled Ritz values in [-1,1]) and weights such that
///
///     trace(g(A)) ~= (n / n_probes) * sum_j tau_j * g(scale * theta_j)
///     trace(g(A/scale)) ~= (n / n_probes) * sum_j tau_j * g(theta_j)
///
/// for any scalar function g.  From this we derive the quantities the adaptive
/// filter selector needs: near-zero mass, positive/negative mass, the scaled
/// spectral extremes, and asymmetry.
struct SpectralSketch {
    int    n         = 0;      ///< matrix dimension
    int    n_probes  = 0;      ///< number of stochastic probes
    double scale     = 1.0;    ///< a-posteriori upper bound on ||A||_2 used to scale
    double lam_min_scaled = 0; ///< signed smallest Ritz value / scale  (in [-1,1])
    double lam_max_scaled = 0; ///< signed largest  Ritz value / scale  (in [-1,1])
    double fro_sq    = 0.0;    ///< exact ||A||_F^2

    std::vector<double> theta; ///< scaled quadrature nodes (Ritz / scale), size n_nodes
    std::vector<double> tau;   ///< quadrature weights (sum to n_probes overall)

    /// Gaussian broadening used for the near-zero density (matches the Lanczos
    /// resolution ~ range/steps).  Sparse quadrature nodes cannot resolve the
    /// fine density near 0 with a hard cutoff, so near-zero queries convolve
    /// each node with N(theta_j, sigma^2).  Broad regions (positive/total mass)
    /// use exact sums.
    double broadening = 0.04;

    // ---- Gaussian helpers (standard normal cdf / pdf) ----
    static double Phi_(double x) { return 0.5 * std::erfc(-x * 0.7071067811865476); }
    static double phi_(double x) { return 0.3989422804014327 * std::exp(-0.5 * x * x); }

    // integral_{-eps}^{eps} y^2 N(y; mu, sigma^2) dy   (truncated 2nd moment)
    static double trunc_sq_(double eps, double mu, double sigma) {
        double a = (-eps - mu) / sigma, b = (eps - mu) / sigma;
        double Pa = Phi_(a), Pb = Phi_(b), pa = phi_(a), pb = phi_(b);
        double m = (mu * mu + sigma * sigma) * (Pb - Pa)
                 + 2.0 * mu * sigma * (pa - pb)
                 - sigma * sigma * (b * pb - a * pa);
        return m > 0.0 ? m : 0.0;
    }

    // ---- spectral-mass queries (all in the SCALED spectrum y = lambda/scale) ----

    /// sum_{|y_i| < eps} y_i^2, Gaussian-broadened (weighted dead-zone mass)
    double near_zero_frob_sq_scaled(double eps) const {
        double acc = 0.0;
        for (size_t j = 0; j < theta.size(); ++j)
            acc += tau[j] * trunc_sq_(eps, theta[j], broadening);
        return acc * (double)n / (double)n_probes;
    }

    /// sum_{y_i >= 0} y_i^2   (used to estimate ||Pi_+(A)||_F^2 = scale^2 * this)
    double positive_frob_sq_scaled() const {
        double acc = 0.0;
        for (size_t j = 0; j < theta.size(); ++j)
            if (theta[j] >= 0.0) acc += tau[j] * theta[j] * theta[j];
        return acc * (double)n / (double)n_probes;
    }

    /// sum_i y_i^2 estimated by quadrature (cross-checked against fro_sq/scale^2)
    double total_frob_sq_scaled() const {
        double acc = 0.0;
        for (size_t j = 0; j < theta.size(); ++j)
            acc += tau[j] * theta[j] * theta[j];
        return acc * (double)n / (double)n_probes;
    }

    /// estimated fraction of eigenvalues with |y| < eps, Gaussian-broadened
    double near_zero_count_frac(double eps) const {
        double acc = 0.0;
        for (size_t j = 0; j < theta.size(); ++j) {
            double a = (-eps - theta[j]) / broadening;
            double b = (eps - theta[j]) / broadening;
            acc += tau[j] * (Phi_(b) - Phi_(a));
        }
        return acc / (double)n_probes;
    }
};

/// @brief Compute a spectral sketch of symmetric ``A`` (n x n, device, column-major).
///
/// Runs ``n_probes`` m-step Lanczos passes with Rademacher start vectors (full
/// reorthogonalisation for stability; mathematically independent, but advanced in
/// lockstep as one block when ``use_block``) and an eigen-decomposition
/// of each tridiagonal to obtain the quadrature nodes/weights.  The scale is an
/// A-POSTERIORI (not certified) upper bound on ||A||_2: on the blocked path
/// (use_block=true -- what AdaptiveOptions::sketch_block selects, although this
/// declaration itself defaults use_block to false) it comes from the sketch's own
/// extremal Ritz value + Lanczos residual bound (no extra pass); the sequential path
/// (use_block=false) uses ``approximate_two_norm``.  Both paths pad the result and
/// clamp the scaled nodes into [-1,1].
///
/// @param cublasH   cuBLAS handle
/// @param cusolverH cuSOLVER handle
/// @param A         device pointer to the symmetric matrix (unchanged)
/// @param n         matrix dimension
/// @param m         Lanczos steps per probe (quadrature order), e.g. 30-50
/// @param n_probes  number of stochastic probes, e.g. 8-16
/// @param seed      RNG seed for the Rademacher vectors
/// @return the sketch (host-side)
/// @param use_fp32  (sequential path only, i.e. use_block=false) run the Lanczos
///                  matvec/reductions in single precision (the matvec then reads a
///                  FP32 copy of A, ~2x less HBM traffic).  Ignored when use_block
///                  is set -- the blocked path always runs in FP64.  The scale
///                  (||A||_2 bound) is always computed in FP64, so the composite
///                  filter's [-1,1] domain guarantee is unaffected.
/// @param use_block advance all probes in lockstep so each step's matvec is one
///                  BLAS-3 GEMM A*Q (A read once for all probes), reorthogonalise
///                  them with one strided-batched GEMV, and derive the scale from
///                  the sketch's own extremal Ritz value + its Lanczos residual
///                  bound (a tight a-posteriori upper bound on ||A||_2 for symmetric
///                  A, with no separate two-norm pass).  Runs in FP64.  Same
///                  quadrature math as the sequential path.
SpectralSketch compute_spectral_sketch(
    cublasHandle_t cublasH,
    cusolverDnHandle_t cusolverH,
    const double* A, int n,
    int m = 40, int n_probes = 8,
    unsigned long long seed = 1234ULL,
    bool use_fp32 = false,
    bool use_block = false
);

/// @brief Compute the `k` largest and `k` smallest Ritz pairs of symmetric `A`
/// from a single `m_lanczos`-step, fully reorthogonalised Lanczos run.  Used to
/// warm-start LOBPCG deflation (the extremal Ritz vectors already approximate
/// the dominant eigenvectors, so LOBPCG converges in a few iterations).
///
/// Outputs (all device, caller-allocated):
///   evecs_max (n x k, orthonormal), evals_max (k, descending)  -- largest
///   evecs_min (n x k, orthonormal), evals_min (k, ascending)   -- smallest
void compute_extremal_ritz(
    cublasHandle_t cublasH,
    cusolverDnHandle_t cusolverH,
    const double* A, int n, int k, int m_lanczos,
    double* evecs_max, double* evals_max,
    double* evecs_min, double* evals_min,
    unsigned long long seed = 1234ULL
);


#endif // PSD_PROJECTION_SPECTRAL_SKETCH_H
