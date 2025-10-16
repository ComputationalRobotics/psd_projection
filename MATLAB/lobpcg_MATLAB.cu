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

#include "psd_projection/lobpcg.h"
#include "psd_projection/utils.h"

void get_dnmat_from_matlab(
    const mxArray* mx_dnmat,
    int* n,
    std::vector<double>& cpu_dnmat_vals
) {
    // read the matrix size from MATLAB and check that it is square
    int cpu_dnmat_row_size = static_cast<int>( mxGetM(mx_dnmat) );
    int cpu_dnmat_col_size = static_cast<int>( mxGetN(mx_dnmat) );
    assert(cpu_dnmat_row_size == cpu_dnmat_col_size);
    *n = static_cast<int>(cpu_dnmat_row_size);

    double* cpu_dnmat_vals_pointer = mxGetPr(mx_dnmat);
    cpu_dnmat_vals.clear();
    cpu_dnmat_vals.resize(*n * *n, 0);
    memcpy(cpu_dnmat_vals.data(), cpu_dnmat_vals_pointer, sizeof(double) * *n * *n);
    return;
}

void get_dnmat_from_matlab(
    const mxArray* mx_dnmat,
    int* nb_rows,
    int* nb_cols,
    std::vector<double>& cpu_dnmat_vals
) {
    int cpu_nb_rows = static_cast<int>( mxGetM(mx_dnmat) );
    int cpu_nb_cols = static_cast<int>( mxGetN(mx_dnmat) );
    *nb_rows = static_cast<int>(cpu_nb_rows);
    *nb_cols = static_cast<int>(cpu_nb_cols);

    double* cpu_dnmat_vals_pointer = mxGetPr(mx_dnmat);
    cpu_dnmat_vals.clear();
    cpu_dnmat_vals.resize(*nb_rows * *nb_cols, 0);
    memcpy(cpu_dnmat_vals.data(), cpu_dnmat_vals_pointer, sizeof(double) * (*nb_rows) * (*nb_cols));
    return;
}

void get_dnvec_from_matlab(
    const mxArray* mx_dnvec,
    int& cpu_dnvec_size, 
    std::vector<double>& cpu_dnvec_vals
) {
    // matlab should pass a column vector, so col_size should always be 1
    int col_size = static_cast<int>( mxGetN(mx_dnvec) );
    assert(col_size == 1);
    cpu_dnvec_size = static_cast<int>( mxGetM(mx_dnvec) );
    double* cpu_dnvec_vals_pointer = mxGetPr(mx_dnvec);
    cpu_dnvec_vals.clear();
    cpu_dnvec_vals.resize(cpu_dnvec_size, 0);
    memcpy(cpu_dnvec_vals.data(), cpu_dnvec_vals_pointer, sizeof(double) * cpu_dnvec_size);
    return;
}


// input order
class INPUT_ID_factory {
    public:
        int A;
        int V;
        int D;
        int m;
        int warmstart;
        int maxiter;
        int tol;
        int verbose;

        INPUT_ID_factory(int offset = 0) {
            this->A = offset + 0;
            this->V = offset + 1;
            this->D = offset + 2;
            this->m = offset + 3;
            this->warmstart = offset + 4;
            this->maxiter = offset + 5;
            this->tol = offset + 6;
            this->verbose = offset + 7;
        }
};

class OUTPUT_ID_factory {
    public:
        int V;
        int D;

        OUTPUT_ID_factory(int offset = 0) {
            this->V = offset + 0;
            this->D = offset + 1;
        }
};

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    /* Input */
    INPUT_ID_factory INPUT_ID(0);
    if (nrhs != 8) {
        mexErrMsgTxt("Wrong number of input arguments. Expected 8 inputs.");
    }
    
    // get A
    int n;
    std::vector<double> cpu_A;
    get_dnmat_from_matlab(prhs[INPUT_ID.A], &n, cpu_A);

    // get m
    int m  = static_cast<int>(mxGetScalar(prhs[INPUT_ID.m]));

    // get V
    int m_v, n_v;
    std::vector<double> cpu_V;
    get_dnmat_from_matlab(prhs[INPUT_ID.V], &n_v, &m_v, cpu_V);
    assert (m_v == m);
    assert (n_v == n);

    // get D
    int m_d;
    std::vector<double> cpu_D;
    get_dnvec_from_matlab(prhs[INPUT_ID.D], m_d, cpu_D);
    assert (m_d == m);

    // get the rest
    bool warmstart = static_cast<bool>(mxGetScalar(prhs[INPUT_ID.warmstart]));
    int maxiter = static_cast<int>(mxGetScalar(prhs[INPUT_ID.maxiter]));
    double tol = static_cast<double>(mxGetScalar(prhs[INPUT_ID.tol]));
    bool verbose = static_cast<bool>(mxGetScalar(prhs[INPUT_ID.verbose]));

    /* Project the matrix */
    // create the handles
    cusolverDnHandle_t solverH;
    CHECK_CUSOLVER(cusolverDnCreate(&solverH));
    cublasHandle_t cublasH;
    CHECK_CUBLAS(cublasCreate(&cublasH));
    
    // create the host matrices
    double *A, *V, *D;
    CHECK_CUDA(cudaMalloc(&A, n * n * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(A, cpu_A.data(), n * n * sizeof(double), H2D));
    CHECK_CUDA(cudaMalloc(&V, m * n * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(V, cpu_V.data(), m * n * sizeof(double), H2D));
    CHECK_CUDA(cudaMalloc(&D, m * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(D, cpu_D.data(), m * sizeof(double), H2D));

    // launch LOBPCG
    lobpcg(
        cublasH, solverH,
        A, V, D,
        n, m,
        warmstart, maxiter, tol, verbose
    );    

    CHECK_CUDA(cudaDeviceSynchronize());

    /* Output the result */
    OUTPUT_ID_factory OUTPUT_ID(0);
    plhs[OUTPUT_ID.V] = mxCreateDoubleMatrix(n, m, mxREAL);
    double *cpu_V_out = mxGetPr(plhs[OUTPUT_ID.V]);
    CHECK_CUDA(cudaMemcpy(cpu_V_out, V, n * m * sizeof(double), D2H));

    plhs[OUTPUT_ID.D] = mxCreateDoubleMatrix(m, 1, mxREAL);
    double *cpu_D_out = mxGetPr(plhs[OUTPUT_ID.D]);
    CHECK_CUDA(cudaMemcpy(cpu_D_out, D, m * sizeof(double), D2H));

    CHECK_CUDA(cudaDeviceSynchronize());

    // free
    CHECK_CUDA(cudaFree(A));
    CHECK_CUDA(cudaFree(V));
    CHECK_CUDA(cudaFree(D));
    CHECK_CUBLAS(cublasDestroy(cublasH));
    CHECK_CUSOLVER(cusolverDnDestroy(solverH));
}