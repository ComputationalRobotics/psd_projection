#include "psd_projection/utils.h"
#include "psd_projection/check.h"
#include <cublas_v2.h>

void symmetrizeFloat(
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

__global__ void convert_double_to_float_kernel(const double* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = static_cast<float>(in[idx]);
    }
}

__global__ void convert_float_to_double_kernel(const float* in, double* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = static_cast<double>(in[idx]);
    }
}

void convert_double_to_float(const double* d_in, float* d_out, int n, const int threadsPerBlock) {
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    convert_double_to_float_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
}

void convert_float_to_double(const float* d_in, double* d_out, int n, const int threadsPerBlock) {
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    convert_float_to_double_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
}

__global__ void build_identity_kernel(float* mat, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n)
        mat[idx] = (idx / n == idx % n) ? 1.0f : 0.0f;
}

void build_identity(
    cublasHandle_t cublasH,
    float* mat,
    int n,
    const int threadsPerBlock
) {
    const int blocksPerGrid = (n * n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel to build identity matrix
    build_identity_kernel<<<blocksPerGrid, threadsPerBlock>>>(mat, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void add_identity_kernel(float* mat, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx / n;
        int col = idx % n;
        if (row == col) {
            mat[idx] += 1.0f; // Add 1 to the diagonal elements
        }
    }
}

void add_identity(
    cublasHandle_t cublasH,
    float* mat,
    int n
) {
    const int threadsPerBlock = 1024;
    const int blocksPerGrid = (n * n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel to add identity matrix
    add_identity_kernel<<<blocksPerGrid, threadsPerBlock>>>(mat, n);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void float4_to_half_kernel(
    const float4* __restrict__ A4,
    __half2 * __restrict__ B2,
    size_t N4
) {
    size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= N4) return;

    // load 4 floats
    float4 v = A4[idx];

    // pack low two floats into half2
    B2[2*idx + 0] = __float22half2_rn(make_float2(v.x, v.y));
    // pack high two floats into half2
    B2[2*idx + 1] = __float22half2_rn(make_float2(v.z, v.w));
}

void convert_float_to_half4(const float* dA, __half* dB, size_t N) {
    size_t N4 = (N + 3)/4;  // how many float4’s
    auto A4 = reinterpret_cast<const float4*>(dA);
    auto B2 = reinterpret_cast<__half2*>(dB);

    const int blk = 1024;
    int grid = (N4 + blk - 1)/blk;
    float4_to_half_kernel<<<grid,blk>>>(A4, B2, N4);
}

// Kernel to replace A by I - A
__global__ void identity_minus_kernel(const float* A_in, float* A_out, const int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx / n;
        int col = idx % n;
        if (row == col) {
            A_out[idx] = 1.0f - A_in[idx]; // diagonal elements
        } else {
            A_out[idx] = -A_in[idx]; // off-diagonal elements
        }
    }
}

void identity_minus(
    const float* A_in,
    float* A_out,
    const int n
) {
    const int nn = n * n;
    const int threads = 1024;
    const int blocks = (nn + threads - 1) / threads;

    // Launch kernel to compute I - A
    identity_minus_kernel<<<blocks, threads>>>(A_in, A_out, n);
    CHECK_CUDA(cudaGetLastError());
}

// Kernel to replace A by I + A
__global__ void identity_plus_kernel(const float* A_in, float* A_out, const int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx / n;
        int col = idx % n;
        if (row == col)
            A_out[idx] = 1.0f + A_in[idx]; // diagonal elements
        else
            A_out[idx] = A_in[idx]; // off-diagonal elements
    }
}

void identity_plus(
    const float* A_in, // device pointer to matrix A
    float* A_out, // device pointer to matrix A
    const int n
) {
    const int nn = n * n;
    const int threads = 1024;
    const int blocks = (nn + threads - 1) / threads;

    // Launch kernel to compute I + A
    identity_plus_kernel<<<blocks, threads>>>(A_in, A_out, n);
    CHECK_CUDA(cudaGetLastError());
}