# Matrix exponential benchmarks

`bench_exponential.jl` times and accuracy-tests every applicable `MatrixAlgebraKit.exponential`
algorithm (`MatrixFunctionViaTaylor`, `MatrixFunctionViaLA`, `MatrixFunctionViaEig`,
`MatrixFunctionViaEigh`) for `Float64`, `ComplexF64`, `BigFloat` and `Complex{BigFloat}`.

Accuracy inputs are built from a *known* spectrum as `A = V · Diag(λ) · V⁻¹`, so the exact
`exp(A) = V · Diag(exp λ) · V⁻¹` is available analytically (computed at high precision).
Eigenvalue families (`small`, `wide`, `imaginary`, `stiff`, `illconditioned`, `hermitian`)
and eigenvector families (`unitary`, `general`, `illconditioned`) are chosen to stress or
favor particular algorithms.

## Setup (once)

From the repository root:

```sh
julia --project=benchmark -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
```

This `dev`s the in-tree `MatrixAlgebraKit` into the benchmark's own environment.

## Run

```sh
julia --project=benchmark benchmark/bench_exponential.jl            # full run
julia --project=benchmark benchmark/bench_exponential.jl --quick    # fast smoke run
julia --project=benchmark benchmark/bench_exponential.jl --csv out.csv
```

Results are printed as per-type accuracy and timing tables and written to two CSVs derived
from the `--csv` path (default base `benchmark/results.csv`): `*-accuracy.csv` and
`*-timing.csv`. They are separate files because the accuracy and timing passes report
disjoint quantities — a single combined table would leave the timing columns empty on the
accuracy rows and vice-versa.

## Accuracy columns

- `analytic_err` — relative error vs the analytic `V·Diag(exp λ)·V⁻¹` reference (primary).
- `recomp_err` — relative error vs an independent high-precision `exp` of the exact input.
- `expexp_err` — `‖exp(A)·exp(−A) − I‖₁` (reference-free consistency check).
- `det_err` — relative error of `det(exp A)` against `exp(tr A)`.
- `condV` — designed condition number of the eigenvector basis (context for non-normal cases).
