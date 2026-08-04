#ifndef PSD_PROJECTION_LANCZOS_H
#define PSD_PROJECTION_LANCZOS_H

/// @brief Approximates the two-norm of a symmetric matrix using the Lanczos method.
/// @param A Dense square matrix stored on device
/// @param n Size of the matrix (n x n)
/// @param lo Lower bound on ‖A‖₂ (the largest Ritz value)
/// @param up A-posteriori upper bound on ‖A‖₂ (Ritz value + residual)
/// @param max_iter Maximum number of Lanczos iterations
/// @param tol Relative residual tolerance for convergence
void approximate_two_norm(
    cublasHandle_t cublasH,
    cusolverDnHandle_t cusolverH,
    const double* A, size_t n,
    double* lo, double* up,
    size_t max_iter = 20, double tol = 1e-10
);

/// @brief WIP: Compute the k eigenpairs of LARGEST MAGNITUDE of a symmetric matrix using the Lanczos method. It performs a two-step post-cleaning, dropping pairs whose residual >= ortho_tol and dropping Ritz vectors whose overlap with the already-accepted basis exceeds ortho_tol. NOTE: `tol` is accepted but never read; both cleaning steps use `ortho_tol`.
/// @param cublasH cuBLAS handle
/// @param cusolverH cuSOLVER handle
/// @param A symmetric matrix of size n x n
/// @param n size of A
/// @param k target number of extremal eigenpairs to compute
/// @param r actual number of eigenpairs computed (output parameter, r <= k)
/// @param eigenvalues rx1 vector of Ritz values after cleaning
/// @param eigenvectors n x r matrix of Ritz vectors after cleaning (column-major)
/// @param upper_bound_only if true, only computes an upper bound on the spectral norm (default false)
/// @param max_iter maximum number of Lanczos iterations (default n)
/// @param tol accepted but UNUSED by the implementation (default 1e-10)
/// @param ortho_tol tolerance used by BOTH cleaning steps: step 1 keeps Ritz pairs whose residual < ortho_tol, step 2 rejects candidates whose overlap with the accepted basis exceeds ortho_tol (default 1e-5)
/// @return A sign-robust upper bound on ‖A‖₂
double compute_eigenpairs(
    cublasHandle_t cublasH,
    cusolverDnHandle_t cusolverH,
    const double* A, size_t n,
    const size_t k,
    size_t *r,
    double* eigenvalues, double* eigenvectors,
    const bool upper_bound_only = false,
    const size_t max_iter = 0, const double tol = 1e-10, const double ortho_tol = 1e-5,
    const bool verbose = false
);

#endif // PSD_PROJECTION_LANCZOS_H