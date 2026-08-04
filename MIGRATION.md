# Migration guide — fixed composite → spectrum-aware adaptive

This release replaces the **fixed composite polynomial filter** with the
**spectrum-aware adaptive** method. One routine, `psd_projection_adaptive`,
supersedes the whole `composite_*` family: it sketches the input's actual spectrum
with matrix–vector products, deflates outliers when that pays, and then applies the
cheapest filter from an offline `(ε, T)` bank that meets a caller-specified relative
Frobenius tolerance. Typical result: **1.4×–3.1× fewer GEMMs at equal or better
accuracy** (see the benchmark tables in [`README.md`](README.md)).

This is a **breaking change** to the C++ API. The MATLAB interface is kept
backward-compatible via deprecated aliases. Nothing below is a silent behaviour
change: every removal is a compile error or a `mexErrMsgTxt`, and the one function
whose semantics were reworked (`lobpcg`) keeps its original defaults.

## Quick start

The adaptive routine does **not** require the caller to pre-scale the matrix — it
derives an a-posteriori upper bound on `‖X‖₂` from its own sketch and scales
internally. That removes the `approximate_two_norm` + `cublasDscal` dance that every
old `composite_*` call site had to perform.

```cpp
// BEFORE — fixed composite, caller scales into [-1,1]
double lo, up;
approximate_two_norm(cublasH, solverH, dA, n, &lo, &up);
const double scale = up > 0.0 ? up : 1.0, inv = 1.0 / scale;
cublasDscal(cublasH, n * n, &inv,   dA, 1);
composite_FP32(cublasH, dA, n);
cublasDscal(cublasH, n * n, &scale, dA, 1);

// AFTER — adaptive, no pre-scaling
#include "psd_projection/adaptive_filter.h"

AdaptiveOptions opts;
opts.tol       = 1e-3;              // target RELATIVE Frobenius error
opts.precision = Precision::FP32;   // or Precision::FP16
AdaptiveReport rep = psd_projection_adaptive(cublasH, solverH, dA, n, opts);
```

`AdaptiveReport` tells you what was chosen: `T`, `gemms`, `eps`, `scale`,
`predicted_rel_err`, `deflated`, `deflation_ms`, `qualified`. `qualified == false`
means no filter in the bank met `tol · safety` and the most accurate available one
was used as a best effort — worth checking if you depend on the tolerance.

## C++ API: removed symbols

| Removed | Header (old) | Replacement |
|---|---|---|
| `composite_FP32` | `composite_FP32.h` | `psd_projection_adaptive` with `Precision::FP32` |
| `composite_FP32_auto_scale` | `composite_FP32.h` | same (auto-scaling is now built in) |
| `composite_FP32_auto_scale_deflate` | `composite_FP32.h` | same (deflation is now automatic; `AdaptiveOptions::enable_deflation`, `deflation_frac`, `outlier_thresh`) |
| `composite_FP16` | `composite_FP16.h` | `psd_projection_adaptive` with `Precision::FP16` |
| `composite_FP16_auto_scale` | `composite_FP16.h` | same |
| `composite_FP16_auto_scale_deflate` | `composite_FP16.h` | same |
| `composite_FP32_emulated` | `composite_FP32_emulated.h` | `psd_projection_adaptive` (FP32 or FP16). The BF16x9 emulated path is no longer built. |
| `eig_FP32_psd` | `eig_FP32_psd.h` | `eig_FP64_psd` for the exact reference, `psd_projection_adaptive` for the fast approximation |
| `eig_FP64_deflate` | `eig_FP64_psd.h` | deflation is internal to the adaptive pipeline; there is no standalone replacement |

The four headers `composite_FP32.h`, `composite_FP16.h`, `composite_FP32_emulated.h`
and `eig_FP32_psd.h` no longer exist, so `#include`s of them fail at compile time
rather than silently resolving. No compatibility shims are provided for the
`composite_*` functions on purpose: their contract required a pre-scaled input and a
fixed stage count, neither of which the adaptive routine honours, so a shim would
have to either misreport its semantics or ignore the workspace pointers you passed.

### Manual workspaces

The old routines took optional `float*` / `__half*` workspaces so callers could
preallocate. `psd_projection_adaptive` manages its own scratch, because the amount
it needs depends on the filter and deflation size it picks at runtime. If you were
pooling that memory, the allocation now happens inside the call.

## C++ API: unchanged

These keep their exact signatures — recompiling is enough:

* `eig_FP64_psd` (`eig_FP64_psd.h`) — the exact cuSOLVER FP64 projection, still the
  reference the adaptive method is measured against;
* `approximate_two_norm`, `compute_eigenpairs` (`lanczos.h`);
* everything in `utils.h` and `check.h`. Note that the `CHECK_CUDA` / `CHECK_CUBLAS` /
  `CHECK_CUSOLVER` macros only *print* on failure — they do not abort, throw, or
  propagate the status, so a failed allocation continues with a null pointer. This
  was true of the fixed-composite version too; callers that need hard failure must
  test the statuses themselves.

Doc comments on several of these were corrected (they described behaviour the code
did not have — e.g. `compute_eigenpairs` ignores its `tol` argument and uses
`ortho_tol` for both cleaning steps, and `approximate_two_norm`'s bounds are
a-posteriori, not certified). No code changed.

## `lobpcg`: convergence test

`lobpcg`'s internals were reworked (one `A`-multiply per iteration, host cyclic-Jacobi
Rayleigh–Ritz below a 128 block size, device-side info flag) and it gained a
**scale-invariant relative** convergence test for the adaptive deflation path.

To keep old call sites behaving as they did, `tol` **retains its original absolute
meaning** (stop when `‖AX − XD‖_F ≤ tol`) and its original `1e-8` default. The
relative test is an explicit opt-in via a new trailing parameter:

```cpp
// unchanged behaviour: absolute residual, 1e-8 default
lobpcg(cublasH, cusolverH, A, V, D, n, m);

// scale-invariant: stop when ||AX - XD||_F <= tol * max(||D||_F, 1)
lobpcg(cublasH, cusolverH, A, V, D, n, m, warmstart, maxiter, tol, verbose,
       /*relative_tol=*/true);
```

Prefer `relative_tol = true` when `‖A‖₂` is not `O(1)`; an absolute `1e-8` on a
matrix with norm `1e6` is effectively a demand for full double precision. The
adaptive pipeline uses `relative_tol = true` with `tol = 1e-6`.

Two pre-existing contract details, now documented rather than changed: `D` is
**output only** — it is never read, even with `warmstart = true`, so a warm start
seeds from `V` alone — and `lobpcg` reports **no convergence status**, so a caller
cannot tell whether it hit `maxiter` with unconverged pairs.

## MATLAB interface

`psd_projection_MATLAB(A, method)` keeps its shape, and `tol` was added as an
optional third argument, so two-argument calls are unaffected.

| Old `method` | Status |
|---|---|
| `'composite_FP32'` | **deprecated alias** → runs `'adaptive'` (FP32) |
| `'composite_FP16'` | **deprecated alias** → runs `'adaptive_FP16'` |
| `'eig_FP64'` | unchanged |
| `'composite_FP32_emulated'` | removed — errors with a message naming the replacement |
| `'eig_FP32'` | removed — errors with a message naming the replacement |

The two aliases mean existing scripts keep running, but they now run the adaptive
method: no pre-scaling requirement, and a stage count chosen per matrix instead of a
fixed `T = 10` (FP32) / `T = 7` (FP16). **The numbers you get back will differ** —
they are closer to the exact projection, not further, but they are not bit-identical
to the old fixed filter. Switch to `'adaptive'` / `'adaptive_FP16'` when convenient;
the aliases exist for compatibility, not permanence.

New names and the optional arguments:

```matlab
[A_psd, report] = psd_projection_MATLAB(A, 'adaptive', 1e-3);   % or 'adaptive_FP16'
fprintf('T=%d gemms=%d deflated=%d predicted=%.2e\n', ...
        report.T, report.gemms, report.deflated, report.predicted_rel_err);
```

The second output is the selection report for adaptive methods (`T`, `gemms`, `eps`,
`scale`, `predicted_rel_err`, `deflated`, `deflation_ms`, `qualified`) and the
eigenvalues for `'eig_FP64'`. It is now produced only when actually requested
(`nargout > 1`).

**`lobpcg_MATLAB` was removed**, along with `MATLAB/example_lobpcg.m`. There is no
MEX entry point for the eigensolver; `lobpcg` remains available from C++.

## Build

* **C++17 is now required** (was C++14) — the generated filter bank declares
  `inline constexpr` aggregates. The library exports this as
  `target_compile_features(psd_lib PUBLIC cxx_std_17)`, so CMake consumers linking
  `psd_lib` inherit it; a hand-rolled build must pass `-std=c++17` to both the host
  and CUDA compilers.
* `set(CMAKE_INSTALL_PREFIX ../)` was **removed** from `CMakeLists.txt`. `install()`
  destinations (`lib/`, `bin/`) are unchanged, but they are now resolved against
  CMake's default prefix or your `-DCMAKE_INSTALL_PREFIX=...`. If you relied on
  headers and the library landing one level above the build tree, pass the prefix
  explicitly.
* GoogleTest is now fetched **inside** the `PSD_PROJECTION_BUILD_TESTS` guard, so a
  default configure no longer needs network access.
* The library target name (`psd_lib` → `libpsd_lib.so`), the include layout
  (`psd_projection/*.h`), and the MEX target name (`psd_projection_MATLAB`) are all
  unchanged.
