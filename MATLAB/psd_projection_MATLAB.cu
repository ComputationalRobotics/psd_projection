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
#include "psd_projection/adaptive_filter.h"
#include "psd_projection/eig_FP64_psd.h"

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
        int tol;

        INPUT_ID_factory(int offset = 0) {
            this->mat = offset + 0;
            this->method = offset + 1;
            this->tol = offset + 2;
        }
};

// Build the MATLAB struct describing what the adaptive method chose.
static mxArray* make_report_struct(const AdaptiveReport& rep) {
    const char* fields[] = {"T", "gemms", "eps", "scale", "predicted_rel_err",
                            "deflated", "deflation_ms", "qualified"};
    mxArray* s = mxCreateStructMatrix(1, 1, 8, fields);
    mxSetField(s, 0, "T",                 mxCreateDoubleScalar((double)rep.T));
    mxSetField(s, 0, "gemms",             mxCreateDoubleScalar((double)rep.gemms));
    mxSetField(s, 0, "eps",               mxCreateDoubleScalar(rep.eps));
    mxSetField(s, 0, "scale",             mxCreateDoubleScalar(rep.scale));
    mxSetField(s, 0, "predicted_rel_err", mxCreateDoubleScalar(rep.predicted_rel_err));
    mxSetField(s, 0, "deflated",          mxCreateDoubleScalar((double)rep.deflated));
    mxSetField(s, 0, "deflation_ms",      mxCreateDoubleScalar(rep.deflation_ms));
    mxSetField(s, 0, "qualified",         mxCreateLogicalScalar(rep.qualified));
    return s;
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    /* Input */
    INPUT_ID_factory INPUT_ID(0);
    if (nrhs != 2 && nrhs != 3) {
        mexErrMsgTxt("Wrong number of input arguments. Expected 2 or 3 inputs: mat, method, [tol].");
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

    // get the optional target relative tolerance (adaptive methods only)
    double tol = 1e-3;
    if (nrhs == 3) {
        const mxArray* mx_tol = prhs[INPUT_ID.tol];
        if (!mxIsDouble(mx_tol) || mxIsComplex(mx_tol) || mxGetNumberOfElements(mx_tol) != 1) {
            mexErrMsgTxt("The 'tol' input must be a real scalar.");
        }
        tol = mxGetScalar(mx_tol);
        if (!(tol > 0.0)) {
            mexErrMsgTxt("The 'tol' input must be positive.");
        }
    }

    /* Project the matrix */
    // create the handles
    cusolverDnHandle_t solverH;
    CHECK_CUSOLVER(cusolverDnCreate(&solverH));

    cublasHandle_t cublasH;
    CHECK_CUBLAS(cublasCreate(&cublasH));
    if (method == "adaptive_FP16" || method == "composite_FP16" || method == "eig_FP64") {
        CHECK_CUBLAS(cublasSetMathMode(cublasH, CUBLAS_TENSOR_OP_MATH));
    }

    // create the device matrix
    double *dA_psd;
    CHECK_CUDA(cudaMalloc(&dA_psd, n * n * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(dA_psd, cpu_At_csc_vals.data(), n * n * sizeof(double), H2D));

    // if method is 'eig_FP64', also output the eigenvalues
    double *eigenvals = nullptr;
    // if method is adaptive, also output the report
    AdaptiveReport rep;
    bool is_adaptive = false;

    // call the appropriate method.  The adaptive method needs no pre-scaling: it
    // derives an a-posteriori bound on ||A||_2 from its own spectral sketch.
    //
    // 'composite_FP32' / 'composite_FP16' are DEPRECATED ALIASES kept so scripts
    // written against the fixed-composite version keep running.  They now run the
    // adaptive method at the same precision, which is both faster and more accurate
    // (see README); the fixed-T filters themselves are no longer built.  The
    // original routines required the caller to pre-scale into [-1,1] and always
    // spent T = 10 (FP32) / T = 7 (FP16) stages -- the alias does neither, so the
    // numerical result differs (it is closer to the exact projection, not further).
    if (method == "adaptive" || method == "adaptive_FP32" || method == "composite_FP32") {
        AdaptiveOptions opts;
        opts.tol = tol;
        opts.precision = Precision::FP32;
        rep = psd_projection_adaptive(cublasH, solverH, dA_psd, (int)n, opts);
        is_adaptive = true;
    }
    else if (method == "adaptive_FP16" || method == "composite_FP16") {
        AdaptiveOptions opts;
        opts.tol = tol;
        opts.precision = Precision::FP16;
        rep = psd_projection_adaptive(cublasH, solverH, dA_psd, (int)n, opts);
        is_adaptive = true;
    }
    else if (method == "eig_FP64") {
        eigenvals = eig_FP64_psd(solverH, cublasH, dA_psd, n, true);
    }
    else if (method == "composite_FP32_emulated") {
        mexErrMsgTxt("'composite_FP32_emulated' has been removed (the BF16x9 "
                     "emulated fixed composite is no longer built). Use 'adaptive' "
                     "for FP32 accuracy at fewer GEMMs, or 'adaptive_FP16' for "
                     "tensor-core throughput. See MIGRATION.md.");
        return;
    }
    else if (method == "eig_FP32") {
        mexErrMsgTxt("'eig_FP32' has been removed (cuSOLVER single-precision "
                     "eigendecomposition). Use 'eig_FP64' for the exact reference "
                     "projection, or 'adaptive' for the fast approximate one. "
                     "See MIGRATION.md.");
        return;
    } else {
        mexErrMsgTxt("Unknown method. Supported methods: 'adaptive' (= 'adaptive_FP32'), "
                     "'adaptive_FP16', and 'eig_FP64'. The names 'composite_FP32' and "
                     "'composite_FP16' are accepted as deprecated aliases of the "
                     "adaptive methods.");
        return;
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    /* Output the result */
    plhs[0] = mxCreateDoubleMatrix(n, n, mxREAL);
    double* cpu_At_psd_vals = mxGetPr(plhs[0]);
    CHECK_CUDA(cudaMemcpy(cpu_At_psd_vals, dA_psd, n * n * sizeof(double), D2H));

    if (nlhs > 1) {
        if (method == "eig_FP64") {
            plhs[1] = mxCreateDoubleMatrix(n, 1, mxREAL);
            double* cpu_eigenvals = mxGetPr(plhs[1]);
            CHECK_CUDA(cudaMemcpy(cpu_eigenvals, eigenvals, n * sizeof(double), D2H));
        } else if (is_adaptive) {
            plhs[1] = make_report_struct(rep);
        }
    }
    if (eigenvals) CHECK_CUDA(cudaFree(eigenvals));

    CHECK_CUDA(cudaDeviceSynchronize());

    // free
    CHECK_CUDA(cudaFree(dA_psd));
    CHECK_CUBLAS(cublasDestroy(cublasH));
    CHECK_CUSOLVER(cusolverDnDestroy(solverH));
}
