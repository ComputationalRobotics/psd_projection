#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <cassert>
#include <cstdio>
#include <limits>
#include <algorithm>

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
    double *V, *V_old, *alpha, *q, *w;
    max_iter = min(max_iter, n);
    CHECK_CUDA(cudaMalloc(&V,     n * max_iter * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&V_old,            n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&alpha,     max_iter * sizeof(double))); // TODO: on host
    CHECK_CUDA(cudaMalloc(&q,                n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&w,                n * sizeof(double)));

    std::vector<double> beta(max_iter, 0.0);

    double minus_alpha, minus_beta_old = 0.0;

    /* Initial vector */
    // q = randn(n, 1)
    fill_random(q, n, 0);

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
        // w = A * q
        CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_N, n, n,
                                 &one, A, n, q, 1,
                                 &zero, w, 1));
        // w = At * w
        CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_T, n, n,
                                 &one, A, n, w, 1,
                                 &zero, w, 1));
        // hence w = A^T * A * q

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
            // fprintf(stderr, "Lanczos iteration %d: beta is zero, stopping early.\n", k);
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

    if (nb_iter == 0) {
        // in this case, the matrix is an all-zero matrix
        *lo = 0.0;
        *up = 1.0;

        CHECK_CUDA(cudaFree(V));
        CHECK_CUDA(cudaFree(V_old));
        CHECK_CUDA(cudaFree(alpha));
        CHECK_CUDA(cudaFree(q));
        CHECK_CUDA(cudaFree(w));
        CHECK_CUDA(cudaDeviceSynchronize());

        return;
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
    // uk = T(:, idx_max)
    CHECK_CUBLAS(cublasDcopy(cublasH, nb_iter, T + idx_max * nb_iter, 1, uk, 1));
    // y = V(:,1:nb_iter) * uk
    CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_N, n, nb_iter, &one, V, n, uk, 1, &zero, y, 1));

    double *ry;
    CHECK_CUDA(cudaMalloc(&ry, n * sizeof(double)));
    // ry = A * q
    CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_N, n, n,
                                &one, A, n, q, 1,
                                &zero, ry, 1));
    // ry = At * ry
    CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_T, n, n,
                                &one, A, n, ry, 1,
                                &zero, ry, 1));
    // hence ry = A^T * A * q

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
    CHECK_CUDA(cudaFree(uk));
    CHECK_CUDA(cudaFree(y));
    CHECK_CUDA(cudaFree(ry));
    CHECK_CUDA(cudaDeviceSynchronize());

    return;
}

__global__ void compute_res_all_kernel(const double* Z, double beta_m, double* res_all, int m) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < m) {
        // Z[j*m] is the 1st row, j-th column (column-major)
        res_all[j] = fabs(beta_m * Z[j * m]);
    }
}

void compute_res_all(
    const double* Z, size_t n, size_t m,
    double beta_m, double* res_all
) {
    // Allocate device memory for res_all
    CHECK_CUDA(cudaMemset(res_all, 0, m * sizeof(double)));

    // Launch kernel to compute residuals for all Ritz pairs
    int blockSize = 1024;
    int numBlocks = (m + blockSize - 1) / blockSize;
    compute_res_all_kernel<<<numBlocks, blockSize>>>(Z, beta_m, res_all, m);
    
    // Check for errors in kernel launch
    CHECK_CUDA(cudaGetLastError());
}

__global__ void fill_random_kernel(double* vec, int n, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        vec[idx] = curand_uniform_double(&state); // random double in (0,1]
    }
}

void fill_random(double* vec, int n, unsigned long seed, const int threadsPerBlock) {
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    fill_random_kernel<<<blocks, threadsPerBlock>>>(vec, n, seed);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in fill_random: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void fill_tridiagonal_kernel(double* T, const double* alpha, const double* beta, int nb_iter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nb_iter) {
        // Set diagonal
        T[i * nb_iter + i] = alpha[i];
        // Set upper diagonal
        if (i < nb_iter - 1) {
            T[i * nb_iter + (i + 1)] = beta[i];
            T[(i + 1) * nb_iter + i] = beta[i];
        }
    }
}

void fill_tridiagonal(
    double* T, const double *d_alpha, const double *d_beta, int nb_iter, const int threadsPerBlock = 1024
) {
    // Launch kernel to fill the tridiagonal matrix T
    int blocks = (nb_iter + threadsPerBlock - 1) / threadsPerBlock;
    fill_tridiagonal_kernel<<<blocks, threadsPerBlock>>>(T, d_alpha, d_beta, nb_iter);
    
    // Check for errors in kernel launch
    CHECK_CUDA(cudaGetLastError());
}

double compute_eigenpairs(
    cublasHandle_t cublasH,
    cusolverDnHandle_t cusolverH,
    const double* A, size_t n,
    const size_t k,
    size_t *r,
    double* eigenvalues, double* eigenvectors,
    const bool upper_bound_only,
    size_t max_iter, const double tol, const double ortho_tol,
    const bool verbose
) {
    if (max_iter == 0)
        max_iter = n;

    /* Allocation */
    double *v0, *Q, *w;
    CHECK_CUDA(cudaMalloc(&v0,                     n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&Q,     n * (max_iter + 1) * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&w,                      n * sizeof(double)));

    double zero = 0.0;
    double one = 1.0;

    double minus_beta = 0.0;
    std::vector<double> alpha, beta;
    double minus_alpha;

    /* Initialize */
    // v0 = randn(n, 1)
    fill_random(v0, n);
    // v0 = v0 / norm(v0)
    double norm_v0;
    CHECK_CUBLAS(cublasDnrm2(cublasH, n, v0, 1, &norm_v0));
    if (norm_v0 != 0.0) {
        double norm_v0_inv = 1.0 / norm_v0;
        CHECK_CUBLAS(cublasDscal(cublasH, n, &norm_v0_inv, v0, 1));
    }

    // Q(:, 1) = v0
    CHECK_CUBLAS(cublasDcopy(cublasH, n, v0, 1, Q, 1));

    int nb_iter = 0;
    /* Lanczos reccurence */
    for (int m = 0; m < max_iter; m++) {
        nb_iter++;
        // w = A * Q(:, m)
        CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_N, n, n,
                                 &one, A, n, Q + m * n, 1,
                                 &zero, w, 1));
        
        // w = w - beta(m-1) * Q(:, m-1)
        CHECK_CUBLAS(cublasDaxpy(cublasH, n, &minus_beta, Q + (m - 1) * n, 1, w, 1));

        // alpha = Q(:, m)^T * w
        CHECK_CUBLAS(cublasDdot(cublasH, n, Q + m * n, 1, w, 1, &minus_alpha));
        alpha.push_back(minus_alpha);
        minus_alpha = -minus_alpha;
        
        // w = w - alpha * Q(:, m)
        CHECK_CUBLAS(cublasDaxpy(cublasH, n, &minus_alpha, Q + m * n, 1, w, 1));

        // beta(m) = norm(w)
        CHECK_CUBLAS(cublasDnrm2(cublasH, n, w, 1, &minus_beta));
        beta.push_back(minus_beta);
        minus_beta = -minus_beta;

        if (-minus_beta <= std::numeric_limits<double>::epsilon())
            break;
        
        // Q(:, m+1) = w / beta(m)
        double beta_inv = -1.0 / minus_beta;
        CHECK_CUBLAS(cublasDscal(cublasH, n, &beta_inv, w, 1));
        CHECK_CUBLAS(cublasDcopy(cublasH, n, w, 1, Q + (m + 1) * n, 1));
    }

    /* Tridiagonal T */
    double *T, *d_alpha, *d_beta;
    CHECK_CUDA(cudaMalloc(&T,       nb_iter * nb_iter * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_alpha,           nb_iter * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_beta,      (nb_iter - 1) * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_alpha, alpha.data(), nb_iter * sizeof(double), H2D));
    CHECK_CUDA(cudaMemcpy(d_beta,  beta.data(),  (nb_iter - 1) * sizeof(double), H2D));
    fill_tridiagonal(
        T, d_alpha, d_beta, nb_iter
    );

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
    // note that the eigenvalues are sorted in ascending order

    // compute the residuals for all Ritz pairs
    double *res_all;
    CHECK_CUDA(cudaMalloc(&res_all, nb_iter * sizeof(double)));
    double beta_m = minus_beta; // last beta value
    compute_res_all(T, nb_iter, nb_iter, beta_m, res_all);

    /* Compute the decreasing absolute value order */
    std::vector<int> idx(nb_iter);
    for (int i = 0; i < nb_iter; i++) {
        idx[i] = i;
    }
    std::vector<double> eigenvalues_host(nb_iter);
    CHECK_CUDA(cudaMemcpy(eigenvalues_host.data(), d_eigenvalues, nb_iter * sizeof(double), cudaMemcpyDeviceToHost));
    std::sort(idx.begin(), idx.end(), [&](size_t i1, size_t i2) {
        return std::abs(eigenvalues_host[i1]) > std::abs(eigenvalues_host[i2]);
    });

    size_t sel_count = 0;
    double *x_candidate, *overlap, *X_basis, *sel;
    if (!upper_bound_only) {
        /* Choose the first k */
        CHECK_CUDA(cudaMalloc(&x_candidate, n * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&overlap,     k * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&X_basis, k * n * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&sel,         k * sizeof(double)));
        bool accept;
        int j;

        for (int id = 0; id < nb_iter; id++) {
            j = idx[id]; // get the index of the j-th Ritz pair

            // Step 1 cleaning: residual filter
            double res_j;
            CHECK_CUDA(cudaMemcpy(&res_j, res_all + j, sizeof(double), D2H));
            if (res_j < ortho_tol) {
                // Step 2 cleaning: ghost orthogonality filter
                // x_candidate = Q(:,1:m) * Z(:,j)
                CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_N, n, nb_iter,
                                        &one, Q, n, T + j * nb_iter, 1,
                                        &zero, x_candidate, 1));            
                if (sel_count == 0) {
                    accept = true;
                } else {
                    // overlap = X_basis' * x_candidate
                    CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_T, n, sel_count,
                                            &one, X_basis, n, x_candidate, 1,
                                            &zero, overlap, 1));

                    // get overlap back to host
                    std::vector<double> overlap_host(sel_count);
                    CHECK_CUDA(cudaMemcpy(overlap_host.data(), overlap, sel_count * sizeof(double), cudaMemcpyDeviceToHost));

                    // check if overlap is below the tolerance
                    accept = true;
                    for (size_t i = 0; i < sel_count; i++) {
                        if (abs(overlap_host[i]) > ortho_tol) {
                            accept = false;
                            break;
                        }
                    }
                }

                if (accept) {
                    // add the new eigenvector to the basis
                    CHECK_CUBLAS(cublasDcopy(cublasH, n, x_candidate, 1, X_basis + sel_count * n, 1));
                    // store the eigenvalue
                    CHECK_CUDA(cudaMemcpy(sel + sel_count, d_eigenvalues + j, sizeof(double), cudaMemcpyDeviceToHost));
                    
                    sel_count++;

                    if (sel_count == k) {
                        // if we have enough eigenpairs, stop
                        break;
                    }
                }
            }
        }

        if (sel_count < k && verbose) {
            fprintf(stderr, "Warning: only %zu eigenpairs found, requested %zu.\n", sel_count, k);
        }
        *r = sel_count;
    }

    /* Spectral norm upper bound */
    // retrieve the max eigenvalue and corresponding eigenvector
    int idx_max = idx[0];

    double theta;
    CHECK_CUDA(cudaMemcpy(&theta, d_eigenvalues + idx_max, sizeof(double), D2H));

    double norm_upper;
    CHECK_CUDA(cudaMemcpy(&norm_upper, res_all + idx_max, sizeof(double), D2H));
    norm_upper += fabs(theta);

    /* Output */
    if (!upper_bound_only) {
        // eigenvectors = X_basis
        CHECK_CUBLAS(cublasDcopy(cublasH, sel_count * n, X_basis, 1, eigenvectors, 1));
        // eigenvalues = sel
        CHECK_CUDA(cudaMemcpy(eigenvalues, sel, sel_count * sizeof(double), D2D));

        CHECK_CUDA(cudaFree(x_candidate));
        CHECK_CUDA(cudaFree(overlap));
        CHECK_CUDA(cudaFree(X_basis));
        CHECK_CUDA(cudaFree(sel));
    }

    /* Free memory */
    CHECK_CUDA(cudaFree(v0));
    CHECK_CUDA(cudaFree(Q));
    CHECK_CUDA(cudaFree(w));
    CHECK_CUDA(cudaFree(T));
    CHECK_CUDA(cudaFree(d_alpha));
    CHECK_CUDA(cudaFree(d_beta));
    CHECK_CUDA(cudaFree(d_eigenvalues));
    CHECK_CUDA(cudaFree(d_work_eig));
    CHECK_CUDA(cudaFree(devInfo));
    CHECK_CUDA(cudaFree(res_all));

    CHECK_CUDA(cudaDeviceSynchronize());

    return norm_upper + 1e-3;
}