#ifndef PSD_PROJECTION_LOBPCG_H
#define PSD_PROJECTION_LOBPCG_H

/// @brief Computes the largest `m` eigenpairs of a symmetric matrix `A` using the LOBPCG algorithm.
/// @param cublasH cuBLAS handle
/// @param cusolverH cuSOLVER handle for QR and eigenvalue decompositions
/// @param A n x n symmetric matrix (device pointer)
/// @param V n x m matrix to store the eigenvectors (device pointer)
/// @param D m x m matrix to store the eigenvalues (device pointer)
/// @param n size of the matrix A
/// @param m number of eigenpairs to compute
/// @param warmstart if true, warm start the algorithm using the values in `V` and `D`; in this case, `V` is assumed to be orthonormal
/// @param maxiter maximum number of iterations
/// @param tol convergence tolerance
/// @param verbose if true, print verbose output
void lobpcg(
    cublasHandle_t cublasH,
    cusolverDnHandle_t cusolverH,
    const double* A, // n x n, device pointer
    double* V,       // n x m, device pointer (output eigenvectors)
    double* D,       // m x m, device pointer (output eigenvalues, diagonal)
    const int n,
    const int m ,    // number of eigenpairs
    const int warmstart = false,
    const int maxiter = 100, // maximum iterations
    const double tol = 1e-8,  // convergence tolerance
    const bool verbose = false // verbosity flag
);

#endif // PSD_PROJECTION_LOBPCG_H