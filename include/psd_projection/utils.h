#ifndef PSD_PROJECTION_UTILS_H
#define PSD_PROJECTION_UTILS_H

#include <iostream>
#include <iomanip>

#define D2H cudaMemcpyDeviceToHost
#define H2D cudaMemcpyHostToDevice
#define D2D cudaMemcpyDeviceToDevice

// Print an n×n matrix of doubles
inline void printMatrixDouble(const double* dM, int n, int m = -1) {
    // If m is not specified, use n for a square matrix
    if (m == -1)
        m = n;

    size_t N = size_t(n) * m;
    std::vector<double> hM(N);
    CHECK_CUDA(cudaMemcpy(hM.data(), dM, N*sizeof(double), cudaMemcpyDeviceToHost));
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < m; ++j) {
            std::cout << std::fixed << std::setprecision(6)
                      << hM[j*n + i] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// Print an n×n matrix of floats
inline void printMatrixFloat(const float* dM, int n) {
    size_t N = size_t(n)*n;
    std::vector<float> hM(N);
    CHECK_CUDA(cudaMemcpy(hM.data(), dM, N*sizeof(float), cudaMemcpyDeviceToHost));
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            std::cout << std::fixed << std::setprecision(6)
                      << hM[i*n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// Print an n×n matrix of __half
inline void printMatrixHalf(const __half* dM, int n) {
    size_t N = size_t(n)*n;
    std::vector<__half> hM(N);
    CHECK_CUDA(cudaMemcpy(hM.data(), dM, N*sizeof(__half), cudaMemcpyDeviceToHost));
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            float v = __half2float(hM[i*n + j]);
            std::cout << std::fixed << std::setprecision(6)
                      << v << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

/// @brief Symmetrize a matrix M by replacing it by (M + M^T) / 2.
/// @param cublasH cuBLAS handle
/// @param M matrix to symmetrize of size n
/// @param n size of the matrix (n x n)
/// @param workspace device memory workspace of size n x n
inline void symmetrizeFloat(
    cublasHandle_t cublasH, float* M, int n, float* workspace
) {
    const float one = 1.0, half = 0.5, zero = 0.0;

    // workspace = M^T
    CHECK_CUBLAS(cublasSgeam(
        cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
        n, n,
        &one, M, n,
        &zero, M, n,
        workspace, n
    ));

    // M = M + workspace (which is M^T)
    CHECK_CUBLAS(cublasSgeam(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        n, n,
        &one, M, n,
        &one, workspace, n,
        M, n
    ));

    // M = 0.5 * M
    CHECK_CUBLAS(cublasSscal(cublasH, n * n, &half, M, 1));
}

#endif // PSD_PROJECTION_UTILS_H