#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <cassert>
#include <cstdio>

#include "psd_projection/lopbcg.h"
#include "psd_projection/check.h"
#include "psd_projection/utils.h"

void lopbcg(
    const double* A, // n x n, device pointer
    double* V,       // n x m, device pointer (output eigenvectors)
    double* D,       // m x m, device pointer (output eigenvalues, diagonal)
    const int n,
    const int m ,      // number of eigenpairs
    const int maxiter, // maximum iterations
    const double tol   // convergence tolerance
) {
    assert(m > 0);
    assert(n > 0);
    assert(3*m <= n);

    /* Allocations */
    // cuBLAS/cuSOLVER handles
    cublasHandle_t cublasH = nullptr;
    cusolverDnHandle_t cusolverH = nullptr;
    CHECK_CUBLAS(cublasCreate(&cublasH));
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));

    // allocate the device memory
    double *X_k, *Lam_k, *T, *Tt, *Delta_X_k, *T_tmp, *R_k;
    double *XRD, *Lam_all, *XRD_tmp, *T_XRD, *Tt_XRD, *T_tmp_XRD;
    CHECK_CUDA(cudaMalloc(&X_k,            n * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&Lam_k,              m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&Delta_X_k,      n * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&T_tmp,          n * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&T,              m * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&Tt,             m * m * sizeof(double)));
    
    CHECK_CUDA(cudaMalloc(&XRD,          n * 3*m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&Lam_all,          3*m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&XRD_tmp,      n * 3*m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&T_XRD,      3*m * 3*m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&Tt_XRD,     3*m * 3*m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&T_tmp_XRD,    n * 3*m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&R_k,          n * 3*m * sizeof(double)));

    double norm_R_k;

    // workspace for QR decomposition of X_k
    int lwork, *devInfo;
    double *d_work, *tau;
    CHECK_CUDA(cudaMalloc(&tau, m * sizeof(double)));
    CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(cusolverH, n, m, X_k, n, &lwork));
    CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));

    // workspace for QR decomposition of XRD
    int lwork_xrd;
    double *d_work_xrd, *tau_xrd;
    CHECK_CUDA(cudaMalloc(&tau_xrd, 3*m * sizeof(double)));
    CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(cusolverH, n, 3*m, XRD, n, &lwork_xrd));
    CHECK_CUDA(cudaMalloc(&d_work_xrd, lwork_xrd * sizeof(double)));

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

    // useful constants
    const double one = 1.0;
    const double zero = 0.0;
    const double half = 0.5;
    const double neg1 = -1.0;

    /* Initialization of X_k */
    // initialize X_k with random values on host
    std::vector<double> h_Xk(n * m);
    for (int i = 0; i < n * m; ++i) {
        h_Xk[i] = 2.0 * (static_cast<double>(rand()) / RAND_MAX) - 1.0; // uniform on [-1, 1)
    }
    CHECK_CUDA(cudaMemcpy(X_k, h_Xk.data(), n * m * sizeof(double), H2D));

    // compute QR factorization (X_k overwritten with R, tau contains Householder scalars)
    CHECK_CUSOLVER(cusolverDnDgeqrf(cusolverH, n, m, X_k, n, tau, d_work, lwork, devInfo));

    // generate Q from the result (X_k overwritten with Q)
    CHECK_CUSOLVER(cusolverDnDorgqr(cusolverH, n, m, m, X_k, n, tau, d_work, lwork, devInfo));

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
    // copy T to Tt
    CHECK_CUBLAS(cublasDcopy(cublasH, m * m, T, 1, Tt, 1));

    // T = 0.5 * (T + T^T)
    CHECK_CUBLAS(cublasDgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, m, m,
                             &half, T, m,
                             &half, Tt, m,
                             T, m));

    // compute eigenvalues and eigenvectors of T
    // both are in increasing order
    CHECK_CUSOLVER(cusolverDnDsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                    m, T, m, Lam_k, d_work_eig, lwork_eig, devInfo));

    // X_k = Q * T
    CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, m, m,
                             &one, X_k, n, T, m,
                             &zero, X_k, n));

    // Delta_X_k = X_k
    CHECK_CUBLAS(cublasDcopy(cublasH, n * m, X_k, 1, Delta_X_k, 1));

    for (int iter = 1; iter <= maxiter; iter++) {
        // R_k = A * X_k - X_k * Lam_k
        CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, m, m,
                                 &one, A, n, X_k, n,
                                 &zero, R_k, n));
        CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, m, m,
                                 &neg1, X_k, n, Lam_k, m,
                                 &one, R_k, n));

        // if the norm of R_k is less than tol, break
        CHECK_CUBLAS(cublasDnrm2(cublasH, n * m, R_k, 1, &norm_R_k));
        if (norm_R_k < tol)
            break;

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
                                &one, T_tmp_XRD, 3*m, X_k, n,
                                &zero, T_XRD, 3*m));
        // copy T to Tt
        CHECK_CUBLAS(cublasDcopy(cublasH, 3*m * 3*m, T_XRD, 1, Tt_XRD, 1));

        // T = 0.5 * (T + T^T)
        CHECK_CUBLAS(cublasDgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, 3*m, 3*m,
                                &half, T_XRD, 3*m,
                                &half, Tt_XRD, 3*m,
                                T_XRD, 3*m));

        // compute eigenvalues and eigenvectors of T
        // both are in increasing order
        CHECK_CUSOLVER(cusolverDnDsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                        3*m, T_XRD, 3*m, Lam_all, d_work_eig_XRD, lwork_eig_XRD, devInfo));

        // XRD_tmp = Q * T
        CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, m, m,
                                &one, XRD_tmp, n, T, m,
                                &zero, XRD_tmp, n));
    }


    // Free device memory
    CHECK_CUDA(cudaFree(X_k));
    CHECK_CUDA(cudaFree(Lam_k));
    CHECK_CUDA(cudaFree(d_work));
    CHECK_CUDA(cudaFree(tau));
    CHECK_CUDA(cudaFree(d_work_xrd));
    CHECK_CUDA(cudaFree(tau_xrd));
    CHECK_CUDA(cudaFree(T_tmp));
    CHECK_CUDA(cudaFree(T_tmp_XRD));
    CHECK_CUDA(cudaFree(d_work_eig));
    CHECK_CUDA(cudaFree(devInfo));
    CHECK_CUDA(cudaFree(T));
    CHECK_CUDA(cudaFree(Tt));
    CHECK_CUDA(cudaFree(Delta_X_k));
    CHECK_CUDA(cudaFree(R_k));
    
    CHECK_CUDA(cudaFree(XRD));
    CHECK_CUDA(cudaFree(Lam_all));
    CHECK_CUDA(cudaFree(T_XRD));
    CHECK_CUDA(cudaFree(Tt_XRD));
    CHECK_CUDA(cudaFree(XRD_tmp));
    CHECK_CUDA(cudaFree(d_work_eig_XRD));

    CHECK_CUBLAS(cublasDestroy(cublasH));
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverH));
}