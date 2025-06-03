#ifndef PSD_PROJECTION_LOPBCG_H
#define PSD_PROJECTION_LOPBCG_H

void lopbcg(
    const double* A, // n x n, device pointer
    double* V,       // n x m, device pointer (output eigenvectors)
    double* D,       // m x m, device pointer (output eigenvalues, diagonal)
    const int n,
    const int m ,    // number of eigenpairs
    const int maxiter = 100, // maximum iterations
    const double tol = 1e-8,  // convergence tolerance
    const bool verbose = false // verbosity flag
);

#endif // PSD_PROJECTION_LOPBCG_H