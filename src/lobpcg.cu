#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <cassert>
#include <cstdio>

#include "psd_projection/lobpcg.h"
#include "psd_projection/check.h"
#include "psd_projection/utils.h"

void lobpcg(
    cublasHandle_t cublasH,
    cusolverDnHandle_t cusolverH,
    const double* A, // n x n, device pointer
    double* V,       // n x m, device pointer (output eigenvectors)
    double* D,       // m x m, device pointer (output eigenvalues, diagonal)
    const int n,
    const int m ,      // number of eigenpairs
    const bool warmstart,
    const int maxiter, // maximum iterations
    const double tol,   // convergence tolerance
    const bool verbose
) {
    assert(m > 0);
    assert(n > 0);
    assert(3*m <= n);

    /* Allocations */
    // allocate the device memory
    double *X_k, *X_k_tmp, *Lam_k, *Lam_k_tmp, *T, *Delta_X_k, *T_tmp, *R_k;
    double *XRD, *Lam_all, *XRD_tmp, *T_XRD, *T_tmp_XRD;
    CHECK_CUDA(cudaMalloc(&X_k,            n * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&X_k_tmp,        n * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&Lam_k,              m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&Lam_k_tmp,          m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&Delta_X_k,      n * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&T_tmp,          n * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&T,              m * m * sizeof(double)));
    
    CHECK_CUDA(cudaMalloc(&XRD,          n * 3*m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&Lam_all,          3*m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&XRD_tmp,      n * 3*m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&T_XRD,      3*m * 3*m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&T_tmp_XRD,    n * 3*m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&R_k,          n * 3*m * sizeof(double)));

    double norm_R_k;

    // useful constants
    const double one = 1.0;
    const double zero = 0.0;
    const double half = 0.5;
    const double neg1 = -1.0;

    // workspace for eigenvalue decomposition
    int lwork_eig;
    double *d_work_eig;
    CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                                m, T, m, Lam_k, &lwork_eig));
    CHECK_CUDA(cudaMalloc(&d_work_eig, lwork_eig * sizeof(double)));

    // workspace for eigenvalue decomposition of XRD
    int lwork_eig_XRD;
    double *d_work_eig_XRD;
    CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                                3*m, T_XRD, 3*m, Lam_all, &lwork_eig_XRD));
    CHECK_CUDA(cudaMalloc(&d_work_eig_XRD, lwork_eig_XRD * sizeof(double)));

    int *devInfo;
    CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));

    // workspace for QR decomposition of XRD
    int lwork_xrd;
    double *d_work_xrd, *tau_xrd;
    CHECK_CUDA(cudaMalloc(&tau_xrd, 3*m * sizeof(double)));
    CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(cusolverH, n, 3*m, XRD, n, &lwork_xrd));
    CHECK_CUDA(cudaMalloc(&d_work_xrd, lwork_xrd * sizeof(double)));

    double *d_work, *tau;

    /* Initialization of X_k */
    if (warmstart) {
        CHECK_CUDA(cudaMemcpy(X_k, V, n * m * sizeof(double), D2D));
        CHECK_CUDA(cudaMemcpy(Lam_k, D, m * sizeof(double), D2D));
        // note: we assume V is orthonormal
    } else {
        // workspace for QR decomposition of X_k
        int lwork;
        CHECK_CUDA(cudaMalloc(&tau, m * sizeof(double)));
        CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(cusolverH, n, m, X_k, n, &lwork));
        CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(double)));

        fill_random(X_k, n * m, 0);

        // compute QR factorization (X_k overwritten with R, tau contains Householder scalars)
        CHECK_CUSOLVER(cusolverDnDgeqrf(cusolverH, n, m, X_k, n, tau, d_work, lwork, devInfo));

        // generate Q from the result (X_k overwritten with Q)
        CHECK_CUSOLVER(cusolverDnDorgqr(cusolverH, n, m, m, X_k, n, tau, d_work, lwork, devInfo));
    }

    /* Compute new X_k using T */
    // T = Q^T * A * Q
    // T_tmp = Q^T * A
    CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, m, n, n,
                             &one, X_k, n, A, n,
                             &zero, T_tmp, m));
    // T = T_tmp * Q
    CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, m, n,
                             &one, T_tmp, m, X_k, n,
                             &zero, T, m));
    // copy T to T_tmp
    CHECK_CUBLAS(cublasDcopy(cublasH, m * m, T, 1, T_tmp, 1));

    // T = 0.5 * (T + T^T)
    CHECK_CUBLAS(cublasDgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, m, m,
                             &half, T, m,
                             &half, T_tmp, m,
                             T, m));

    // compute eigenvalues and eigenvectors of T
    // both are in increasing order
    CHECK_CUSOLVER(cusolverDnDsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                    m, T, m, Lam_k_tmp, d_work_eig, lwork_eig, devInfo));
    // reverse Lam_k_tmp to get Lam_k in decreasing order
    reverse_vector(Lam_k_tmp, Lam_k, m);

    // X_k = Q * T
    CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, m, m,
                             &one, X_k, n, T, m,
                             &zero, X_k, n));

    // Delta_X_k = X_k
    CHECK_CUBLAS(cublasDcopy(cublasH, n * m, X_k, 1, Delta_X_k, 1));

    for (int iter = 1; iter <= maxiter; iter++) {
        // R_k = A * X_k - X_k * Lam_k
        CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, m, n,
                                 &one, A, n, X_k, n,
                                 &zero, R_k, n));
        // scale X_k by Lam_k
        CHECK_CUBLAS(cublasDdgmm(
            cublasH,
            CUBLAS_SIDE_RIGHT, // scale columns
            n,                 // number of rows
            m,                 // number of columns
            X_k, n,            // input matrix
            Lam_k, 1,         // vector (stride 1)
            X_k_tmp, n       // output matrix
        ));
        // substract it from R_k
        CHECK_CUBLAS(cublasDaxpy(cublasH, n * m, &neg1, X_k_tmp, 1, R_k, 1));

        CHECK_CUBLAS(cublasDnrm2(cublasH, n * m, R_k, 1, &norm_R_k));

        if (verbose) {
            std::cout << "LOBPCG iter: " << iter << "||R_k||_F = " << norm_R_k << std::endl;
        }

        // if the norm of R_k is less than tol, break
        if (norm_R_k < tol) {
            if (verbose) {
                std::cout << "Converged: ||R_k||_F < tol" << std::endl;
            }
            break;
        }

        // concatenate X_k, R_k, and Delta_X_k into XRD
        CHECK_CUDA(cudaMemcpy(XRD            ,       X_k, n * m * sizeof(double), D2D));
        CHECK_CUDA(cudaMemcpy(XRD +     n * m,       R_k, n * m * sizeof(double), D2D));
        CHECK_CUDA(cudaMemcpy(XRD + 2 * n * m, Delta_X_k, n * m * sizeof(double), D2D));

        // compute QR factorization of XRD
        CHECK_CUSOLVER(cusolverDnDgeqrf(cusolverH, n, 3*m, XRD, n, tau_xrd, d_work_xrd, lwork_xrd, devInfo));
        CHECK_CUSOLVER(cusolverDnDorgqr(cusolverH, n, 3*m, 3*m, XRD, n, tau_xrd, d_work_xrd, lwork_xrd, devInfo));

        // T = Q^T * A * Q
        // T_tmp = Q^T * A
        CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, 3*m, n, n,
                                &one, XRD, n, A, n,
                                &zero, T_tmp_XRD, 3*m));
        // T = T_tmp * Q
        CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, 3*m, 3*m, n,
                                &one, T_tmp_XRD, 3*m, XRD, n,
                                &zero, T_XRD, 3*m));
        // copy T to T_tmp
        CHECK_CUBLAS(cublasDcopy(cublasH, 3*m * 3*m, T_XRD, 1, T_tmp_XRD, 1));

        // T = 0.5 * (T + T^T)
        CHECK_CUBLAS(cublasDgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, 3*m, 3*m,
                                &half, T_XRD, 3*m,
                                &half, T_tmp_XRD, 3*m,
                                T_XRD, 3*m));

        // compute eigenvalues and eigenvectors of T
        // both are in increasing order
        CHECK_CUSOLVER(cusolverDnDsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                        3*m, T_XRD, 3*m, Lam_all, d_work_eig_XRD, lwork_eig_XRD, devInfo));
        // reverse columns of T_XRD
        reverse_columns(T_XRD, T_tmp_XRD, 3*m, 3*m);

        // XRD_tmp = Q * T
        CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, 3*m, 3*m,
                                &one, XRD, n, T_tmp_XRD, 3*m,
                                &zero, XRD_tmp, n));

        // Delta_X_k = - X_k
        CHECK_CUBLAS(cublasDcopy(cublasH, n * m, X_k, 1, Delta_X_k, 1));
        CHECK_CUBLAS(cublasDscal(cublasH, n * m, &neg1, Delta_X_k, 1));

        // X_k = XRD_tmp(1:m)
        for (int i = 0; i < m; ++i) {
            CHECK_CUBLAS(cublasDcopy(cublasH, n, XRD_tmp + i*n, 1, X_k + i*n, 1));
        }

        // Delta = X_kp1 - X_k
        CHECK_CUBLAS(cublasDaxpy(cublasH, n * m, &one, X_k, 1, Delta_X_k, 1));
        
        // Lam_k = Lam_all(2m:3m)
        CHECK_CUBLAS(cublasDcopy(cublasH, m, Lam_all + 2*m, 1, Lam_k_tmp, 1));
        reverse_vector(Lam_k_tmp, Lam_k, m);
    }

    /* Copy results to output */
    // V = X_k
    CHECK_CUBLAS(cublasDcopy(cublasH, n * m, X_k, 1, V, 1));
    // D = Lam_k
    CHECK_CUBLAS(cublasDcopy(cublasH, m, Lam_k, 1, D, 1));


    // Free device memory
    CHECK_CUDA(cudaFree(X_k));
    CHECK_CUDA(cudaFree(X_k_tmp));
    CHECK_CUDA(cudaFree(Lam_k));
    CHECK_CUDA(cudaFree(Lam_k_tmp));
    if (!warmstart) {
        CHECK_CUDA(cudaFree(d_work));
        CHECK_CUDA(cudaFree(tau));
    }
    CHECK_CUDA(cudaFree(d_work_xrd));
    CHECK_CUDA(cudaFree(tau_xrd));
    CHECK_CUDA(cudaFree(T_tmp));
    CHECK_CUDA(cudaFree(T_tmp_XRD));
    CHECK_CUDA(cudaFree(d_work_eig));
    CHECK_CUDA(cudaFree(devInfo));
    CHECK_CUDA(cudaFree(T));
    CHECK_CUDA(cudaFree(Delta_X_k));
    CHECK_CUDA(cudaFree(R_k));
    
    CHECK_CUDA(cudaFree(XRD));
    CHECK_CUDA(cudaFree(Lam_all));
    CHECK_CUDA(cudaFree(T_XRD));
    CHECK_CUDA(cudaFree(XRD_tmp));
    CHECK_CUDA(cudaFree(d_work_eig_XRD));
}