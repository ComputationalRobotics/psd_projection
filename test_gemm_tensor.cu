#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size_M>" << std::endl;
        return -1;
    }

    int M = std::atoi(argv[1]);
    int N = M;
    int K = M;
    size_t size_A = static_cast<size_t>(M) * K;
    size_t size_B = static_cast<size_t>(K) * N;
    size_t size_C = static_cast<size_t>(M) * N;

    // Host allocations
    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    std::vector<float> h_C(size_C, 0.0f);

    // Random fill
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : h_A) v = dist(gen);
    for (auto &v : h_B) v = dist(gen);

    // Device allocations
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A * sizeof(float));
    cudaMalloc(&d_B, size_B * sizeof(float));
    cudaMalloc(&d_C, size_C * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C * sizeof(float));

    // Create cuBLAS handle and enable Tensor Core math
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    cublasSetEmulationStrategy(handle, CUBLAS_EMULATION_STRATEGY_EAGER);

    float alpha = 1.0f;
    float beta  = 0.0f;

    // Warm-up to exclude startup overhead
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, CUDA_R_32F, N,
        d_A, CUDA_R_32F, K,
        &beta,
        d_C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    cudaDeviceSynchronize();

    // Setup CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start
    cudaEventRecord(start);

    // Actual GEMM
    for(int i = 0; i < 100; ++i) {
        cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, CUDA_R_32F, N,
            d_A, CUDA_R_32F, K,
            &beta,
            d_C, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
    }

    // Record stop and synchronize
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    std::cout << "Tensor-Core GEMM took " << elapsed_ms << " ms (excluding warm-up)" << std::endl;

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Tensor-Core GEMM failed" << std::endl;
        return -1;
    }

    // Copy result back
    cudaMemcpy(h_C.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the first element as a sanity check
    std::cout << "C[0,0] = " << h_C[0] << std::endl;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}