#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <cmath>
#include <iomanip> 
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <cusolverDn.h>

#include "psd_projection/iterative_TF16.h"
#include "psd_projection/check.h"
#include "psd_projection/utils.h"

// Generate and project PSD: in double precision
// Outputs A (n×n) and its PSD projection A_psd
void generateAndProject(int n,
                        double* dA,
                        double* dA_psd,
                        cusolverDnHandle_t solverH,
                        cublasHandle_t cublasH) {
    size_t nn = size_t(n)*n;
    // 1) Generate random Gaussian M_h on host
    std::vector<double> M_h(nn);
    srand(1);
    for(size_t i=0;i<nn;i++) M_h[i] = rand() / (double)RAND_MAX - 0.5;
    // Copy to device as dB
    double *dB; CHECK_CUDA(cudaMalloc(&dB, nn*sizeof(double)));
    CHECK_CUDA(cudaMemcpy(dB, M_h.data(), nn*sizeof(double), cudaMemcpyHostToDevice));

    // 2) QR factorization to get Q in dB
    int *devInfo; CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));
    int lwork_qr = 0;
    CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(solverH, n, n, dB, n, &lwork_qr));
    double *dWork; CHECK_CUDA(cudaMalloc(&dWork, lwork_qr*sizeof(double)));
    double *tau;   CHECK_CUDA(cudaMalloc(&tau, n*sizeof(double)));
    CHECK_CUSOLVER(cusolverDnDgeqrf(solverH, n, n, dB, n, tau, dWork, lwork_qr, devInfo));
    CHECK_CUSOLVER(cusolverDnDorgqr(solverH, n, n, n, dB, n, tau, dWork, lwork_qr, devInfo));
    // dB now contains Q

    // 3) Copy Q to dQ before scaling
    double *dQ; CHECK_CUDA(cudaMalloc(&dQ, nn*sizeof(double)));
    CHECK_CUDA(cudaMemcpy(dQ, dB, nn*sizeof(double), cudaMemcpyDeviceToDevice));

    // 4) Generate lambda on host and copy to device
    std::vector<double> lambda_h(n);

    // uniform eigenvalue
    for(int i=0;i<n;i++) lambda_h[i] = -1.0 + 2.0*i/(double)(n-1);

    // 1/k eigenvalue
    // for (int i = 0; i < n; i++) {
    //     double tmp = n/2.0 - abs(n/2.0 - i) + 1.0;
    //     if (i < n/2) lambda_h[i] = 1.0 / (tmp * tmp);
    //     else lambda_h[i] = -1.0 / (tmp * tmp);

    //     // if (i < n/2) lambda_h[i] = 0.0;
    //     // else lambda_h[i] = 1.0;
    //     // printf("i: %d, lam_i: %.10f \n", i, lambda_h[i]);
    // }

    double *dLambda; CHECK_CUDA(cudaMalloc(&dLambda, n*sizeof(double)));
    CHECK_CUDA(cudaMemcpy(dLambda, lambda_h.data(), n*sizeof(double), cudaMemcpyHostToDevice));

    // 5) Build A = Q * diag(lambda) * Q^T
    // scale Q columns in dB by lambda
    for(int i=0;i<n;i++){
        CHECK_CUBLAS(cublasDscal(cublasH, n, &lambda_h[i], dB + i*n, 1));
    }
    const double one_d=1.0, zero_d=0.0;
    // A = (Q*Lambda) * Q^T, using original Q in dQ
    CHECK_CUBLAS(cublasGemmEx(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
        n, n, n,
        &one_d,
        dB, CUDA_R_64F, n,
        dQ, CUDA_R_64F, n,
        &zero_d,
        dA, CUDA_R_64F, n,
        CUDA_R_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        
    // Symmetrize A = (A + A^T)/2 using a temporary buffer
    double half_d = 0.5;
    double *dSym; CHECK_CUDA(cudaMalloc(&dSym, nn*sizeof(double)));
    CHECK_CUBLAS(cublasDgeam(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
        n, n,
        &half_d, dA, n,
        &half_d, dA, n,
        dSym, n));
    CHECK_CUDA(cudaMemcpy(dA, dSym, nn*sizeof(double), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaFree(dSym));

    // save the original A for later
    double *dA_orig; CHECK_CUDA(cudaMalloc(&dA_orig, nn*sizeof(double)));
    CHECK_CUDA(cudaMemcpy(dA_orig, dA, nn*sizeof(double), cudaMemcpyDeviceToDevice));

    // 6) PSD projection: eigendecomp on dA
    double *dW; CHECK_CUDA(cudaMalloc(&dW, n*sizeof(double)));
    int lwork_ev = 0;
    CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(
        solverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
        n, dA, n, dW, &lwork_ev));
    double *dWork_ev; CHECK_CUDA(cudaMalloc(&dWork_ev, lwork_ev*sizeof(double)));
    CHECK_CUSOLVER(cusolverDnDsyevd(
        solverH,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        n, dA, n, dW,
        dWork_ev, lwork_ev, devInfo));
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<double> W_h(n);
    CHECK_CUDA(cudaMemcpy(W_h.data(), dW, n*sizeof(double), cudaMemcpyDeviceToHost));
    for(int i=0;i<n;i++) if(W_h[i]<0) W_h[i]=0;

    // Copy eigenvectors from dA to dV
    double *dV; CHECK_CUDA(cudaMalloc(&dV, nn*sizeof(double)));
    CHECK_CUDA(cudaMemcpy(dV, dA, nn*sizeof(double), cudaMemcpyDeviceToDevice));

    // Scale columns of dV by W_h
    for(int i=0;i<n;i++){
        CHECK_CUBLAS(cublasDscal(cublasH, n, &W_h[i], dV + i*n, 1));
    }

    // Reconstruct A_psd = V * V^T
    double *dTmp; CHECK_CUDA(cudaMalloc(&dTmp, nn*sizeof(double)));
    CHECK_CUBLAS(cublasGemmEx(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
        n, n, n,
        &one_d,
        dV, CUDA_R_64F, n,
        dA, CUDA_R_64F, n,
        &zero_d,
        dTmp, CUDA_R_64F, n,
        CUDA_R_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaMemcpy(dA_psd, dTmp, nn*sizeof(double), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaFree(dTmp));
    CHECK_CUDA(cudaFree(dV));

    // copy the dA_origin back to dA
    CHECK_CUDA(cudaMemcpy(dA, dA_orig, nn*sizeof(double), cudaMemcpyDeviceToDevice));

    // cleanup internals
    cudaFree(dB); cudaFree(dQ); cudaFree(devInfo);
    cudaFree(dWork); cudaFree(tau);
    cudaFree(dLambda); cudaFree(dW); cudaFree(dWork_ev);
}

TEST(IterativeTF16, Random)
{
    cusolverDnHandle_t solverH; cublasHandle_t cublasH;
    CHECK_CUSOLVER(cusolverDnCreate(&solverH));
    CHECK_CUBLAS(cublasCreate(&cublasH));
    CHECK_CUBLAS(cublasSetMathMode(cublasH, CUBLAS_TENSOR_OP_MATH));

    size_t n = 1024;
    size_t nn = n*n;

    double *dA, *dA_psd;
    CHECK_CUDA(cudaMalloc(&dA, nn*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dA_psd, nn*sizeof(double)));

    generateAndProject(n, dA, dA_psd, solverH, cublasH); // cuSOLVER
    projection_TF16(cublasH, dA, n, 0, 100); // iterative TF16 projection

    // check if dA and dA_psd are approximately equal
    double *dDiff; CHECK_CUDA(cudaMalloc(&dDiff, nn*sizeof(double)));
    double one = 1.0, neg1 = -1.0;
    CHECK_CUBLAS(cublasDgeam(
        cublasH,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n,
        &one,  dA_psd, n,
        &neg1, dA, n,
        dDiff,       n));
    double final_err = 0.0f;
    CHECK_CUBLAS(cublasDnrm2(cublasH, nn, dDiff, 1, &final_err));

    ASSERT_LE(final_err, 1e-2) << "Final error: " << final_err;
}