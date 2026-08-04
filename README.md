# PSD Projection Toolbox
Implementation of different algorithms of projection onto the PSD cone.

This library focuses on orthogonally projecting a symmetric matrix $X$ onto the Positive Semidefinite (PSD) cone, that is:
```math
\Pi_{\mathbb{S}_+^n}(X) \;:=\; \text{argmin}_{Y \in \mathbb{S}_+^n} \,\frac{1}{2}\,\|Y - X\|_{\text{F}}^2,
```
where $\mathbb{S}^n$ is the set of $n \times n$ symmetric matrices and $\|\cdot\|_{\text{F}}$ is the Frobenius norm.

With memory and time efficiency in mind, the library is implemented in C++ and CUDA, and supports multiple data types (half, single and double precision). A MATLAB interface is also provided.

The polynomial filter used by the factorization-free method is selected per matrix at run time rather than fixed in advance, from a cheap estimate of where the eigenvalues of the input actually lie, so that the projection costs the fewest matrix products that still meet a caller-specified relative accuracy. On the 33 matrices of the MatrixDepot benchmark at $n = 20000$ this reduces the mean number of matrix products from 31 to 15.9 in single precision and from 22 to 13.2 in half precision, at a lower mean error. Callers of the previous fixed-composite interface should consult [`MIGRATION.md`](MIGRATION.md), which maps every removed symbol to its replacement.

## Features
### Factorization-based methods
The standard approach to compute the projection is to use the eigenvalue decomposition (EVD) of $X$. If the spectral decomposition of $X$ is given by
```math
X = Q \Lambda Q^\top,
```
the projection can be computed as:
```math
\Pi_{\mathbb{S}_+^n}(X) = Q \max(\Lambda, 0) Q^\top,
```
where $\max(\Lambda, 0)$ is the matrix obtained by replacing all negative eigenvalues in $\Lambda$ with zero.

This approach is implemented in double precision in [`src/eig_FP64_psd.cu`](src/eig_FP64_psd.cu), and uses CUDA's cuSOLVER library for efficient computation of the EVD. It is exposed as `eig_FP64_psd`, and is the exact reference the factorization-free methods are measured against.

### Factorization-free solution via polynomial filtering
An alternative approach to compute the projection is to use polynomial filtering, which avoids the need for factorization, as described in [the associated paper](https://arxiv.org/abs/2507.09165). The matrix is rescaled by an upper bound of its spectral norm, and a polynomial approximation of the ReLU function is applied to its eigenvalues.

The upper bound is obtained by a Lanczos iteration, exposed as `approximate_two_norm` in [`lanczos.cu`](src/lanczos.cu), and composite polynomial filtering is implemented in [`src/composite_generic.cu`](src/composite_generic.cu). A composite filter of $T$ stages costs $3T + 1$ matrix products, and the fixed filter of the paper uses $T = 10$ in single precision and $T = 7$ in half precision, each designed for a worst-case dead zone of $10^{-3}$ around the origin.

### Spectrum-aware adaptive filtering
A filter designed for a worst-case dead zone is longer than necessary whenever the eigenvalues of the input do not in fact crowd the origin. The adaptive method therefore estimates where the eigenvalues of $X$ lie and then selects the shortest filter that still meets a prescribed accuracy. It is implemented in [`src/adaptive_filter.cu`](src/adaptive_filter.cu) and exposed as `psd_projection_adaptive`.

The estimate is a spectral sketch, computed in [`src/spectral_sketch.cu`](src/spectral_sketch.cu) from matrix-vector products alone. It provides an a-posteriori upper bound on $\|X\|_2$, which is used to scale the spectrum into $[-1, 1]$, the extremal Ritz values of $X$ and a stochastic Lanczos quadrature approximation of the spectral density, from which the mass of eigenvalues near the origin is read off for any candidate dead zone. The bound is padded slightly, since the composite filter diverges for eigenvalues of magnitude greater than one.

Spectral outliers are then optionally removed by a warm-started block LOBPCG, implemented in [`src/lobpcg.cu`](src/lobpcg.cu), with the two ends of the spectrum treated independently. A side is deflated when the sketch places at least one but fewer than $0.01\,n$ of its scaled eigenvalues beyond $0.05$ in magnitude, which keeps the removal quadratic in $n$, and the number removed is clamped to $n/3$. The remainder is re-sketched and filtered, and the positive part of each removed eigenpair is added back exactly. A matrix whose positive part is empty after this step is projected to zero directly, with no filter applied, and is reported as qualified with a predicted error of zero.

Selection is driven by a model of the relative Frobenius error of each candidate filter: `select_filter` returns the entry with the fewest matrix products whose predicted error, multiplied by a safety factor of 1.5, meets the requested tolerance. The model includes a per-precision arithmetic floor, $10^{-6}$ in single and $5 \times 10^{-4}$ in half precision, below which no polynomial can improve the result, so a half-precision tolerance below roughly $7.5 \times 10^{-4}$ cannot be met by any filter in the bank; on a matrix that has a positive part the selector then falls back to its most accurate entry and reports `qualified = false`. The prediction is an estimate rather than a bound, and is in practice conservative.

The filters themselves are designed offline, indexed by dead zone and stage count, using the two-stage construction of the paper. The bank is distributed as the generated header [`include/psd_projection/filter_bank.h`](include/psd_projection/filter_bank.h), which is not meant to be edited by hand; the generator is not part of this repository, so building the library never requires running it. Its entries span one to seven stages, or four to twenty-two matrix products.

### Measured performance
The adaptive method and the fixed composite filter were compared on the 33 matrices of the MatrixDepot benchmark, at $n = 5000$, $10000$ and $20000$, in single and half precision, against `eig_FP64_psd` as the reference. The adaptive runs used a tolerance of $10^{-4}$ in single precision and $10^{-3}$ in half precision. Errors are relative Frobenius errors against the reference, and both columns are means over the 33 matrices; per-matrix times are medians of three runs, except for the cuSOLVER reference, which is run once. The benchmark binary was compiled for `sm_90`, so the times carry no JIT compilation. The harness that produced these numbers, and the copy of the fixed composite filter it compares against, are not part of this repository.

| $n$ | method | mean error | mean time |
|---|---|--:|--:|
| 5000 | cuSOLVER FP64 (reference) | — | 0.259 s |
| | composite FP32 (fixed) | 6.45e-05 | 0.178 s |
| | adaptive FP32 | 6.62e-06 | 0.105 s |
| | composite FP16 (fixed) | 9.07e-04 | 0.016 s |
| | adaptive FP16 | 1.63e-04 | 0.023 s |
| 10000 | cuSOLVER FP64 (reference) | — | 1.016 s |
| | composite FP32 (fixed) | 6.22e-05 | 1.297 s |
| | adaptive FP32 | 7.47e-06 | 0.693 s |
| | composite FP16 (fixed) | 3.45e-03 | 0.080 s |
| | adaptive FP16 | 1.81e-04 | 0.072 s |
| 20000 | cuSOLVER FP64 (reference) | — | 5.899 s |
| | composite FP32 (fixed) | 3.40e-04 | 10.024 s |
| | adaptive FP32 | 8.71e-06 | 5.201 s |
| | composite FP16 (fixed) | 8.53e-03 | 0.539 s |
| | adaptive FP16 | 1.84e-04 | 0.405 s |

The mean error improves by a factor of 8.3 to 39 in single precision and 5.6 to 46 in half precision, depending on the size. The improvement is not uniform across matrices: at $n = 20000$ the adaptive method is the more accurate of the two on 25 of the 33 matrices in single precision and on 27 of 33 in half precision. Single precision is faster than the fixed filter at every size, and its mean time is below that of the exact FP64 factorization at every size as well, although at $n = 20000$ that second comparison holds only in the mean: cuSOLVER is faster there on 22 of the 33 matrices, and the mean is pulled down by the few inputs that deflation resolves entirely. Half precision is slower than the fixed filter at $n = 5000$, where the cost of the sketch is not yet amortized, and faster from $n = 10000$ up. These runs share a GPU node.

## Compilation
### Execution
Build the project using CMake:
```bash
mkdir build
cmake -S . -B build && cmake --build build
```
This builds the shared library `libpsd_lib.so`. C++17 is required, since the generated filter bank is declared as `inline constexpr` aggregates; the requirement is exported through `target_compile_features`, so a consumer that links `psd_lib` through CMake inherits it. This project has been tested with CMake 4.2, CUDA 12.4 and GCC 12.2 on Linux (NVIDIA H200).

> [!NOTE]
> `CUDA_ARCHITECTURES` is set to `OFF` on the library target, so nvcc emits its own default cubin together with PTX and the library stays forward-portable. Because the property is set on the target, it overrides `-DCMAKE_CUDA_ARCHITECTURES` given on the command line; to emit native code for a specific GPU, edit the property in [`CMakeLists.txt`](CMakeLists.txt) or remove that line. A process that loads `libpsd_lib.so` on a GPU the default cubin does not cover pays a one-time JIT compilation on its first kernel launch.

### Usage
The matrix does not need to be rescaled by the caller: the spectral sketch derives the upper bound on $\|X\|_2$ and applies the scaling internally. The matrix is overwritten in place with the projection, and the returned report describes what was selected:
```cpp
#include "psd_projection/adaptive_filter.h"

AdaptiveOptions opts;
opts.tol       = 1e-3;              // target relative Frobenius error
opts.precision = Precision::FP32;   // or Precision::FP16

// dA: n x n symmetric matrix, double, device, column-major
AdaptiveReport rep = psd_projection_adaptive(cublasH, cusolverH, dA, n, opts);

printf("T=%d gemms=%d predicted_rel_err=%.2e deflated=%d\n",
       rep.T, rep.gemms, rep.predicted_rel_err, rep.deflated);
```
Element counts passed to cuBLAS are 32-bit, so a single call is limited to $n \le 46340$.

### MATLAB interface
We provide in the [`MATLAB`](MATLAB) directory a MATLAB interface to the library. It can be built using CMake:
```bash
cd MATLAB
mkdir build
cmake -S . -B build && cmake --build build
```

> [!NOTE]
> The MEX file must be built with GCC 12 or older. It is loaded into MATLAB's own process, and MATLAB ships its own `libstdc++` that provides at most `GLIBCXX_3.4.30` as of R2024b, so a MEX built with GCC 13 links but fails to load. The core library has no such constraint.

You can then call the PSD projection routines from MATLAB:
```matlab
addpath('build');
A = ...
[A_psd, report] = psd_projection_MATLAB(A, 'adaptive', 1e-3);
```
where the second argument specifies the method to use (the supported methods are `adaptive`, `adaptive_FP16` and `eig_FP64`, with `adaptive_FP32` accepted as a synonym of `adaptive`) and the optional third argument is the target relative Frobenius error, which defaults to `1e-3` and applies to the adaptive methods only. The optional second output is the selection report for the adaptive methods, carrying the fields `T`, `gemms`, `eps`, `scale`, `predicted_rel_err`, `deflated`, `deflation_ms` and `qualified`, or the eigenvalues for `eig_FP64`. The names `composite_FP32` and `composite_FP16` are also accepted as deprecated aliases of the adaptive methods, so that existing scripts keep running; see [`MIGRATION.md`](MIGRATION.md) for the full mapping. A minimal working example is provided in [`MATLAB/example.m`](MATLAB/example.m).

### Testing
After building with the option `PSD_PROJECTION_BUILD_TESTS` in the CMake file, you can execute the unit tests. The option is off by default and enabling it fetches GoogleTest at configure time, which requires network access. The tests are host-only, exercising `select_filter` and the error model on synthetic sketches, and cover no CUDA code path:
```bash
cmake -S . -B build -DPSD_PROJECTION_BUILD_TESTS=ON && cmake --build build
cd build && ctest
```

## Citing
To cite the method, or if you used this library in your work, please use the following BibTeX entry:
```bibtex
@misc{kang2025psdprojection,
    title={Factorization-free Orthogonal Projection onto the Positive Semidefinite Cone with Composite Polynomial Filtering}, 
    author={Shucheng Kang and Haoyu Han and Antoine Groudiev and Heng Yang},
    year={2025},
    eprint={2507.09165},
    archivePrefix={arXiv},
    primaryClass={math.OC},
    url={https://arxiv.org/abs/2507.09165}
}
```
