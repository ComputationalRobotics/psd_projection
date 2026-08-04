#ifndef PSD_PROJECTION_LOBPCG_H
#define PSD_PROJECTION_LOBPCG_H

#include <cublas_v2.h>
#include <cusolverDn.h>


/// @brief Block LOBPCG for the largest `m` eigenpairs of a symmetric matrix
/// `A`, warm-startable and performance-tuned:
///   (A) the QR / eigensolve info flag stays on the device and is read only once,
///       after the loop; the remaining per-iteration host transfers are the residual
///       norm and the small Rayleigh-Ritz block (or just its eigenvalues on the GPU
///       path), so an iteration does still synchronise, but only on O(nb^2) data;
///   (B) `A` is multiplied by the search block only ONCE per iteration
///       (W = A*Q); the Rayleigh-Ritz matrix and the next A*X are both derived
///       from W, halving the HBM traffic on `A`;
///   (C) the Rayleigh-Ritz eigenproblem is solved on the host with a cyclic
///       Jacobi sweep while the block is <= RR_GPU_NB (128); larger blocks fall
///       back to a per-iteration cuSOLVER syevd on the device;
///   (E) the first iteration uses the [X, R] block (no degenerate P); every later
///       iteration uses the full [X, R, P] block.
///
/// @note Convergence test.  `tol` keeps the ABSOLUTE meaning it had in the fixed
///       composite library -- the iteration stops when ||A X - X D||_F <= tol --
///       with the same 1e-8 default, so calls written against that interface behave
///       as before.  Pass `relative_tol = true` for the scale-invariant test
///       ||A X - X D||_F <= tol * max(||D||_F, 1); that is what the adaptive
///       deflation path uses internally (with tol = 1e-6), and it is the right
///       choice when ||A||_2 is not O(1).
///
/// @param cublasH cuBLAS handle
/// @param cusolverH cuSOLVER handle
/// @param A n x n symmetric matrix (device, column-major, unchanged)
/// @param V n x m eigenvectors (device; input initial guess if warmstart)
/// @param D m eigenvalues (device; OUTPUT only -- never read, even with warmstart),
///        descending
/// @param n matrix size
/// @param m number of eigenpairs (3m <= n)
/// @param warmstart if true, V (assumed orthonormal) seeds the iteration; D is
///        ignored on input and the initial Ritz values are recomputed
/// @param maxiter maximum iterations
/// @param tol residual tolerance, absolute unless `relative_tol` (see the note)
/// @param verbose print per-iteration residuals
/// @param relative_tol interpret `tol` as a relative residual instead of absolute
void lobpcg(
    cublasHandle_t cublasH,
    cusolverDnHandle_t cusolverH,
    const double* A,
    double* V,
    double* D,
    const int n,
    const int m,
    const bool warmstart = false,
    const int maxiter = 100,
    const double tol = 1e-8,
    const bool verbose = false,
    const bool relative_tol = false
);


#endif // PSD_PROJECTION_LOBPCG_H
