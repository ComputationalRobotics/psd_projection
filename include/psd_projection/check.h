/*

    check.h

    Defines CHECK macros for CUDA, cuBLAS, cuSOLVER, and cuSPARSE.
    Each tests the return status of an API call and printf's a diagnostic when it
    fails.  NOTE: they do NOT abort, throw, or propagate the status -- execution
    continues after a failed call, so a failed allocation leaves a null pointer in
    play.  Callers that need hard failure must check the status themselves.

*/

#ifndef CUADMM_CHECK_H
#define CUADMM_CHECK_H

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <iostream>

// Check if the function returns a CUDA error
#define CHECK_CUDA(func)                                                       \
do {                                                                           \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at %s:%d with error: %s (%d)",                 \
               __FILE__, __LINE__, cudaGetErrorString(status), status);        \
        std::cout << std::endl;                                                \
    }                                                                          \
} while (0) // wrap it in a do-while loop to be called with a semicolon

// Check if the function returns a cuBLAS error
#define CHECK_CUBLAS(func)                                                     \
do {                                                                           \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("cuBLAS error %d at %s:%d", status, __FILE__, __LINE__);        \
        std::cout << std::endl;                                                \
    }                                                                          \
} while (0)

// Check if the function returns a cuSOLVER error
#define CHECK_CUSOLVER(func)                                                   \
do {                                                                           \
    cusolverStatus_t status = (func);                                          \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
        printf("cuSOLVER error %d at %s:%d", status, __FILE__, __LINE__);      \
        std::cout << std::endl;                                                \
    }                                                                          \
} while (0)

// Check if the function returns a cuSPARSE error
#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("cuSPARSE error %s (%d) at %s:%d",                              \
              cusparseGetErrorString(status), status, __FILE__, __LINE__);     \
        std::cout << std::endl;                                                \
    }                                                                          \
}

#endif // CUADMM_CHECK_H