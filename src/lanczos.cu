#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <cassert>
#include <cstdio>

#include "psd_projection/lanczos.h"
#include "psd_projection/check.h"
#include "psd_projection/utils.h"

void approximate_two_norm(
    cublasHandle_t cublasH,
    cusolverDnHandle_t cusolverH,
    const double* A, size_t n,
    double* lo, double* up,
    size_t max_iter, double tol
) {
    /* Allocations */
    // constants
    const double zero = 0.0;
    const double one = 1.0;
    
    // storage
    double *V, *V_old, *alpha, *q, *w, *AtA;
    CHECK_CUDA(cudaMalloc(&V,     n * max_iter * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&V_old,            n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&alpha,     max_iter * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&q,                n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&w,                n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&AtA,          n * n * sizeof(double)));

    std::vector<double> beta(max_iter, 0.0);

    // precompute A^T * A
    CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
                             &one, A, n, A, n,
                             &zero, AtA, n));

    double minus_alpha, minus_beta_old;

    /* Initial vector */
    // q = randn(n, 1)
    std::vector<double> q_host(n);
    for (size_t i = 0; i < n; ++i) {
        q_host[i] = 2.0 * (static_cast<double>(rand()) / RAND_MAX) - 1.0; // Random values in [-1, 1)
    }
    CHECK_CUDA(cudaMemcpy(q, q_host.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    // q = q / norm(q)
    double norm_q;
    CHECK_CUBLAS(cublasDnrm2(cublasH, n, q, 1, &norm_q));
    if (norm_q != 0.0) {
        double norm_q_inv = 1.0 / norm_q;
        CHECK_CUBLAS(cublasDscal(cublasH, n, &norm_q_inv, q, 1));
    }

    // V(:, 1) = q
    CHECK_CUBLAS(cublasDcopy(cublasH, n, q, 1, V, 1));
    // fill V_old with zeros
    CHECK_CUDA(cudaMemset(V_old, 0, n * sizeof(double)));

    /* Lanczos loop */
    int nb_iter = 0;
    for (int k = 0; k < max_iter; k++) {
        // w = A^T * A * q
        CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, n,
                                 &one, AtA, n, q, n,
                                 &zero, w, n));

        // alpha(k) = q^T * w
        CHECK_CUBLAS(cublasDdot(cublasH, n, q, 1, w, 1, &alpha[k]));

        // minus_alpha = -alpha[k]
        CHECK_CUDA(cudaMemcpy(&minus_alpha, &alpha[k], sizeof(double), D2H));
        minus_alpha = -minus_alpha;
        
        // w = w - alpha(k) * q - beta_old * V_old
        CHECK_CUBLAS(cublasDaxpy(cublasH, n, &minus_alpha, q, 1, w, 1));
        CHECK_CUBLAS(cublasDaxpy(cublasH, n, &minus_beta_old, V_old, 1, w, 1));
        
        // beta(k) = norm(w)
        CHECK_CUBLAS(cublasDnrm2(cublasH, n, w, 1, &beta[k]));
        
        if (beta[k] <= tol * (-minus_alpha) && k > 1)
            break;
            
        // V_old = q
        CHECK_CUBLAS(cublasDcopy(cublasH, n, q, 1, V_old, 1));
        // q = w / beta(k)
        CHECK_CUDA(cudaMemcpy(q, w, n * sizeof(double), cudaMemcpyDeviceToDevice));
        if (beta[k] != 0.0) {
            double beta_inv = 1.0 / beta[k];
            CHECK_CUBLAS(cublasDscal(cublasH, n, &beta_inv, q, 1));
        } else {
            // If beta is zero, we cannot proceed further
            fprintf(stderr, "Lanczos iteration %d: beta is zero, stopping early.\n", k);
            break;
        }

        if (k < max_iter - 1) {
            // V(:, k+1) = q
            CHECK_CUBLAS(cublasDcopy(cublasH, n, q, 1, V + (k + 1) * n, 1));
        }
        // minus_beta_old = -beta[k]
        minus_beta_old = -beta[k];

        nb_iter++;
    }

    /* Tridiagonal T */
    // T = diag(alpha) + diag(beta(2:end),1) + diag(beta(2:end),-1);
    double *T;
    CHECK_CUDA(cudaMalloc(&T, nb_iter * nb_iter * sizeof(double)));
    std::vector<double> alpha_host(nb_iter, 0.0);
    CHECK_CUDA(cudaMemcpy(alpha_host.data(), alpha, nb_iter * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> T_host(nb_iter * nb_iter, 0.0);
    for (int i = 0; i < nb_iter; i++) {
        T_host[i * nb_iter + i] = alpha_host[i]; // diagonal
        if (i < nb_iter - 1) {
            T_host[i * nb_iter + (i + 1)] = beta[i + 1]; // upper diagonal
            T_host[(i + 1) * nb_iter + i] = beta[i + 1]; // lower diagonal
        }
    }
    CHECK_CUDA(cudaMemcpy(T, T_host.data(), nb_iter * nb_iter * sizeof(double), H2D));

    /* Largest Ritz pair */
    // allocate memory for eigenvalues
    double *d_eigenvalues;
    CHECK_CUDA(cudaMalloc(&d_eigenvalues, nb_iter * sizeof(double)));

    // allocate workspace for eigenvalue decomposition
    int lwork_eig, *devInfo;
    double *d_work_eig;
    CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                               nb_iter, T, nb_iter, d_eigenvalues, &lwork_eig));
    CHECK_CUDA(cudaMalloc(&d_work_eig, lwork_eig * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));

    // compute eigenvalues and eigenvectors
    CHECK_CUSOLVER(cusolverDnDsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                    nb_iter, T, nb_iter, d_eigenvalues,
                                    d_work_eig, lwork_eig, devInfo));

    // retrieve the max eigenvalue and corresponding eigenvector
    int idx_max;
    CHECK_CUBLAS(cublasIdamax(cublasH, nb_iter, d_eigenvalues, 1, &idx_max));
    idx_max--; // convert to 0-based index

    double theta;
    CHECK_CUDA(cudaMemcpy(&theta, d_eigenvalues + idx_max, sizeof(double), D2H));

    double *uk, *y;
    CHECK_CUDA(cudaMalloc(&uk, nb_iter * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y,        n * sizeof(double)));
    // uk = V(:, idx_max)
    CHECK_CUBLAS(cublasDcopy(cublasH, nb_iter, V + idx_max * n, 1, uk, 1));
    // y = V(:,1:nb_iter) * uk
    CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_N, n, nb_iter, &one, V, n, uk, 1, &zero, y, 1));

    // ry = A^T * A * y
    double *ry;
    CHECK_CUDA(cudaMalloc(&ry, n * sizeof(double)));
    CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_T, n, n, &one, AtA, n, y, 1, &zero, ry, 1));
    // ry = ry - theta * y
    double minus_theta = -theta;
    CHECK_CUBLAS(cublasDaxpy(cublasH, n, &minus_theta, y, 1, ry, 1));

    /* Output */
    // lo = sqrt(theta)
    *lo = std::sqrt(theta);

    // up = sqrt(theta + norm(ry))
    double norm_ry;
    CHECK_CUBLAS(cublasDnrm2(cublasH, n, ry, 1, &norm_ry));
    *up = std::sqrt(theta + norm_ry);

    /* Free memory */
    CHECK_CUDA(cudaFree(V));
    CHECK_CUDA(cudaFree(V_old));
    CHECK_CUDA(cudaFree(alpha));
    CHECK_CUDA(cudaFree(q));
    CHECK_CUDA(cudaFree(w));
    CHECK_CUDA(cudaFree(T));
    CHECK_CUDA(cudaFree(d_eigenvalues));
    CHECK_CUDA(cudaFree(d_work_eig));
    CHECK_CUDA(cudaFree(devInfo));
    CHECK_CUDA(cudaFree(AtA));
    CHECK_CUDA(cudaFree(uk));
    CHECK_CUDA(cudaFree(y));
    CHECK_CUDA(cudaFree(ry));
}