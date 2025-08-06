/*

    psd_projection_MATLAB.cu

    This file is part of psd_projection. It defines MATLAB interface functions for the psd_projection library.

*/

#include <memory>
#include <vector>
#include <string>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cassert>

#include "mex.h"
#include "matrix.h"
#include "mat.h"

#include "psd_projection/check.h"
#include "psd_projection/utils.h"
#include "psd_projection/composite_FP32.h"
#include "psd_projection/composite_FP32_emulated.h"
#include "psd_projection/composite_FP16.h"
#include "psd_projection/lanczos.h"
#include "psd_projection/eig_FP64_psd.h"
#include "psd_projection/eig_FP32_psd.h"

void get_dnmat_from_matlab(
    const mxArray* mx_dnmat,
    size_t* n,
    std::vector<double>& cpu_dnmat_vals
) {
    // read the matrix size from MATLAB and check that it is square
    int cpu_dnmat_row_size = static_cast<int>( mxGetM(mx_dnmat) );
    int cpu_dnmat_col_size = static_cast<int>( mxGetN(mx_dnmat) );
    assert(cpu_dnmat_row_size == cpu_dnmat_col_size);
    *n = static_cast<size_t>(cpu_dnmat_row_size);

    double* cpu_dnmat_vals_pointer = mxGetPr(mx_dnmat);
    cpu_dnmat_vals.clear();
    cpu_dnmat_vals.resize(*n * *n, 0);
    memcpy(cpu_dnmat_vals.data(), cpu_dnmat_vals_pointer, sizeof(double) * *n * *n);
    return;
}

// input order
class INPUT_ID_factory {
    public:
        int mat;
        int method;

        INPUT_ID_factory(int offset = 0) {
            this->mat = offset + 0;
            this->method = offset + 1;
        }
};

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    /* Input */
    INPUT_ID_factory INPUT_ID(0);
    if (nrhs != 2) {
        mexErrMsgTxt("Wrong number of input arguments. Expected 2 inputs: mat, method.");
    }
    
    // get the matrix
    size_t n;
    std::vector<double> cpu_At_csc_vals;
    get_dnmat_from_matlab(prhs[INPUT_ID.mat], &n, cpu_At_csc_vals);

    // get the method
    const mxArray* mx_method = prhs[INPUT_ID.method];
    if (!mxIsChar(mx_method)) {
        mexErrMsgTxt("The 'method' input must be a string.");
    }
    char* method_cstr = mxArrayToString(mx_method);
    if (!method_cstr) {
        mexErrMsgTxt("Failed to convert 'method' input to string.");
    }
    std::string method(method_cstr);
    mxFree(method_cstr);

    /* Project the matrix */
    // create the handles
    cusolverDnHandle_t solverH;
    CHECK_CUSOLVER(cusolverDnCreate(&solverH));

    cublasHandle_t cublasH;
    CHECK_CUBLAS(cublasCreate(&cublasH));
    if (method == "composite_FP16" || method == "eig_FP64" || method == "eig_FP32") {
        CHECK_CUBLAS(cublasSetMathMode(cublasH, CUBLAS_TENSOR_OP_MATH));
    }
    #if defined(CUDA_VERSION) && (CUDA_VERSION >= 12090)
    else if (method == "composite_FP32_emulated") {
        CHECK_CUBLAS(cublasSetMathMode(cublasH, CUBLAS_TENSOR_OP_MATH));
        CHECK_CUBLAS(cublasSetEmulationStrategy(cublasH, CUBLAS_EMULATION_STRATEGY_EAGER));
    }
    #endif
    
    // create the host matrix
    double *dA_psd;
    CHECK_CUDA(cudaMalloc(&dA_psd, n * n * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(dA_psd, cpu_At_csc_vals.data(), n * n * sizeof(double), H2D));

    // if method is 'eig_FP64' or 'eig_FP32', also output the eigenvalues
    double *eigenvals;

    // call the appropriate method
    if (method == "composite_FP16") {
        // approximate the spectral norm
        double lo, up;
        approximate_two_norm(
            cublasH, solverH, dA_psd, n, &lo, &up
        );
        // scale to have eigenvalues in [-1, 1]
        const double scale = up > 0.0 ? up : 1.0;
        const double inv_scale = 1.0/scale;
        CHECK_CUBLAS( cublasDscal(cublasH, n*n, &inv_scale, dA_psd, 1) );
        composite_FP16(cublasH, dA_psd, n);
        // unscale
        CHECK_CUBLAS( cublasDscal(cublasH, n*n, &scale, dA_psd, 1) );
    } else if (method == "composite_FP32") {
        // approximate the spectral norm
        double lo, up;
        approximate_two_norm(
            cublasH, solverH, dA_psd, n, &lo, &up
        );
        // scale to have eigenvalues in [-1, 1]
        const double scale = up > 0.0 ? up : 1.0;
        const double inv_scale = 1.0/scale;
        CHECK_CUBLAS( cublasDscal(cublasH, n*n, &inv_scale, dA_psd, 1) );
        composite_FP32(cublasH, dA_psd, n);
        // unscale
        CHECK_CUBLAS( cublasDscal(cublasH, n*n, &scale, dA_psd, 1) );
    }
    else if (method == "composite_FP32_emulated") {
        #if defined(CUDA_VERSION) && (CUDA_VERSION >= 12090)
        // approximate the spectral norm
        double lo, up;
        approximate_two_norm(
            cublasH, solverH, dA_psd, n, &lo, &up
        );
        // scale to have eigenvalues in [-1, 1]
        const double scale = up > 0.0 ? up : 1.0;
        const double inv_scale = 1.0/scale;
        CHECK_CUBLAS( cublasDscal(cublasH, n*n, &inv_scale, dA_psd, 1) );
        composite_FP32_emulated(cublasH, dA_psd, n);
        // unscale
        CHECK_CUBLAS( cublasDscal(cublasH, n*n, &scale, dA_psd, 1) );
        #else
        mexErrMsgTxt("composite_FP32_emulated is only supported with CUDA 12.9 or later.");
        return;
        #endif
    }
    else if (method == "eig_FP64") {
        eigenvals = eig_FP64_psd(solverH, cublasH, dA_psd, n, true);
    }
    else if (method == "eig_FP32") {
        eigenvals = eig_FP32_psd(solverH, cublasH, dA_psd, n, true);
    } else {
        mexErrMsgTxt("Unknown method. Supported methods: 'composite_FP16', 'composite_FP32', 'composite_FP32_emulated', 'eig_FP64', and 'eig_FP32'.");
        return;
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    /* Output the result */
    plhs[0] = mxCreateDoubleMatrix(n, n, mxREAL);
    double* cpu_At_psd_vals = mxGetPr(plhs[0]);
    CHECK_CUDA(cudaMemcpy(cpu_At_psd_vals, dA_psd, n * n * sizeof(double), D2H));

    if (method == "eig_FP64" || method == "eig_FP32") {
        plhs[1] = mxCreateDoubleMatrix(n, 1, mxREAL);
        double* cpu_eigenvals = mxGetPr(plhs[1]);
        CHECK_CUDA(cudaMemcpy(cpu_eigenvals, eigenvals, n * sizeof(double), D2H));
        CHECK_CUDA(cudaFree(eigenvals));
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    // free
    CHECK_CUDA(cudaFree(dA_psd));
    CHECK_CUBLAS(cublasDestroy(cublasH));
    CHECK_CUSOLVER(cusolverDnDestroy(solverH));
}