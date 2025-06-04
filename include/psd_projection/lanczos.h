#ifndef PSD_PROJECTION_LANCZOS_H
#define PSD_PROJECTION_LANCZOS_H

/// @brief Approximates the two-norm of a symmetric matrix using the Lanczos method.
/// @param A Dense square matrix stored on device
/// @param n Size of the matrix (n x n)
/// @param lo Certified lower bound on ‖A‖₂
/// @param up Certified upper bound on ‖A‖₂
/// @param max_iter Maximum number of Lanczos iterations
/// @param tol Relative residual tolerance for convergence
void approximate_two_norm(
    const double* A, size_t n,
    double* lo, double* up,
    size_t max_iter = 50, double tol = 1e-6
);

#endif // PSD_PROJECTION_LANCZOS_H