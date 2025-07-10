# PSD Projection Toolbox
Implementation of different algorithms of projection onto the PSD cone.

This library focuses on orthogonally projecting a symmetric matrix $X$ onto the Positive Semidefinite (PSD) cone, that is:
```math
\Pi_{\mathbb{S}^n}(X) \;:=\; \text{argmin}_{Y \in \mathbb{S}^n} \,\frac{1}{2}\,||Y - X||_{\text{F}}^2,
```
where $\mathbb{S}^n$ is the set of $n \times n$ symmetric matrices and $||\cdot||_{\text{F}}$ is the Frobenius norm.

With memory and time efficiency in mind, the library is implemented in C++ and CUDA, and supports multiple data types (half, single and double precision). A MATLAB interface is also provided.

## Features
### Factorization-based methods
The standard approach to compute the projection is to use the eigenvalue decomposition (EVD) of $X$. If the spectral decomposition of $X$ is given by
```math
X = Q \Lambda Q^\top,
```
the projection can be computed as:
```math
\Pi_{\mathbb{S}^n}(X) = Q \max(\Lambda, 0) Q^\top,
```
where $\max(\Lambda, 0)$ is the matrix obtained by replacing all negative eigenvalues in $\Lambda$ with zero.

This approach is implemented in double precision in [`src/eig_FP64_psd.cu`](src/eig_FP64_psd.cu), and uses CUDA's cuSOLVER library for efficient computation of the EVD.

### Factorization-free solution via polynomial filtering
An alternative approach to compute the projection is to use polynomial filtering, which avoids the need for factorization. The matrix is rescaled by an upper bound of its spectral norm, and a polynomial approximation of the ReLU function is applied to its eigenvalues.

The upper bound is computed using [`lanczos.cu`](src/lanczos.cu), and composite polynomial filtering is implemented in [`src/composite_FP32.cu`](src/composite_FP32.cu), [`src/composite_TF16.cu`](src/composite_TF16.cu).

> [!NOTE]
> For GPUs with Blackwell architecture and CUDA 12.9 or later, we implement in [`src/composite_FP32_emulated.cu`](src/composite_FP32_emulated.cu) a version of the standard FP32 composite method that uses the BF16x9 algorithm to emulate FP32 operations with multiple half precision operations, which is more efficient without any drop in accuracy.

## Compilation
### Execution
Build the project using CMake:
```bash
mkdir build
cmake -S . -B build && cmake --build build
```

### MATLAB interface
We provide in the [`MATLAB`](MATLAB) directory a MATLAB interface to the library. It can be built using CMake:
```bash
cd MATLAB
mkdir build
cmake -S . -B build && cmake --build build
```
You can then call the PSD projection routines from MATLAB:
```matlab
addpath('build');
A = ...
A_psd = psd_projection_MATLAB(A, 'composite_FP32');
```
where the second argument specifies the method to use (the supported methods are `composite_TF16`, `composite_FP32`, `composite_FP32_emulated`, `haoyu_TF16`, `eig_FP64`.)
A minimal working example is provided in [`MATLAB/example.m`](MATLAB/example.m).

### Testing
After building, you can execute the unit tests:
```bash
cd build && ctest
```