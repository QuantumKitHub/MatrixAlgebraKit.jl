# MatrixAlgebraKit.jl test suite

Tests are driven by [ParallelTestRunner.jl](https://github.com/JuliaTesting/ParallelTestRunner.jl).
Every `.jl` file under `test/` is auto-discovered and executed in its own worker process, so test
files share no state and must be self-contained.

The test environment is a [Pkg workspace](https://pkgdocs.julialang.org/dev/workspaces/) member:
`test/Project.toml` declares the test dependencies and resolves the parent package through
`[sources]`, while the whole workspace shares a single `Manifest.toml` at the repository root.

## Running tests

```julia
# Standard — works on all supported Julia versions
using Pkg; Pkg.test()

# With arguments (see "Selecting tests" below)
using Pkg; Pkg.test("MatrixAlgebraKit"; test_args = ["decompositions", "--fast"])
```

```bash
# Direct invocation — requires Julia 1.12+ (workspace support)
julia --project=test test/runtests.jl
```

The direct form is the convenient one for local iteration: it reuses the already-instantiated
workspace environment instead of building a fresh temporary one on every run.

## Selecting tests

Each discovered test is keyed by its path relative to `test/`, with the `.jl` extension stripped —
for example `decompositions/svd`, `common/truncate`, `mooncake/qr`. Positional arguments are matched
against those keys with `startswith`, so a directory name selects a whole group and a longer path
selects a single file:

```bash
julia --project=test test/runtests.jl --list             # print all discovered test keys

julia --project=test test/runtests.jl decompositions     # one group
julia --project=test test/runtests.jl decompositions/svd # one file
julia --project=test test/runtests.jl mooncake enzyme    # union of several selectors
```

Available flags:

| Flag | Effect |
|------|--------|
| `--list` | List all discovered tests and exit |
| `--fast` | Reduced test matrix (see below) |
| `--jobs=N` | Use `N` worker processes (default: derived from CPU count and available memory) |
| `--verbose` | More detailed progress output |
| `--quickfail` | Abort the whole run as soon as one test errors |
| `--help` | Usage information |

Note that passing an explicit selector **disables** the automatic OS- and CI-based filtering in
`runtests.jl` — `filter_tests!` returns `false` once positional arguments are present, so you get
exactly the tests you asked for.

### Fast mode (`--fast`)

Sets a `fast_tests = true` constant in every test sandbox. Files branch on
`@isdefined(fast_tests) && fast_tests` to shrink their element-type matrix, typically from
`(Float32, Float64, ComplexF32, ComplexF64)` plus generic floats down to `(Float64, ComplexF64)` and
one generic float. CI uses this for draft pull requests.

## Test groups

The top-level directories under `test/` are the test groups, and they are exactly the matrix jobs of
the CI workflow — `.github/workflows/Tests.yml` delegates to
`QuantumKitHub/QuantumKitHubActions/.github/workflows/TestGroups.yml`, which discovers groups by
listing those directories. Adding a new directory therefore adds a new CI job automatically; a test
file placed directly in `test/` would belong to no group and never run in CI.

| Group | Contents |
|-------|----------|
| `common` | Algorithm selection and defaults, truncation strategies, projections, matrix exponential, Aqua code-quality checks |
| `decompositions` | `qr`, `lq`, `svd`, `eig`, `eigh`, `gen_eig`, `schur`, `polar`, `orthnull` on CPU and GPU array types |
| `chainrules` | ChainRulesCore rules, exercised through ChainRulesTestUtils and Zygote |
| `mooncake` | Mooncake AD rules |
| `enzyme` | Enzyme AD rules, exercised through EnzymeTestUtils |
| `genericlinearalgebra` | `MatrixAlgebraKitGenericLinearAlgebraExt` |
| `genericschur` | `MatrixAlgebraKitGenericSchurExt` |

Two directories are *not* groups: `testsuite/` holds the shared implementation described below and
is excluded both from discovery (in `runtests.jl`) and from CI (`exclude: '["testsuite"]'`).

## `testsuite/` — the shared test suite

`test/testsuite/TestSuite.jl` defines the `TestSuite` module, a backend-parametric suite modelled on
GPUArrays.jl and intended to be reusable by packages that build on MatrixAlgebraKit. It contains the
actual assertions; the files under the group directories are thin drivers that pick element types
and array types and call into it.

Because it is excluded from discovery, it is loaded explicitly by each test file that needs it:

```julia
@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite
```

The suite exposes `test_*` entry points, one per operation, taking a *matrix type* and a size:

```julia
TestSuite.test_qr(Matrix{Float64}, (54, 37))
TestSuite.test_qr(CuMatrix{ComplexF64}, (54, 37); test_pivoted = false, test_blocksize = false)
TestSuite.test_qr(Diagonal{Float64, CuVector{Float64}}, 54)
TestSuite.test_qr_algs(Matrix{Float64}, (54, 37), (Householder(; positive = true),))
```

The AD entry points (`test_chainrules`, `test_mooncake_*`, `test_enzyme_*`) instead take a scalar
element type and a size, plus `atol`/`rtol` keywords:

```julia
TestSuite.test_chainrules(ComplexF64, (19, 17); atol = tol, rtol = tol)
```

Supporting infrastructure in the module:

- `instantiate_matrix(AT, size)` — builds a random matrix of the requested type, with methods for
  `Array`, `CuArray`, `ROCArray`, `Diagonal`, and `Diagonal` wrapping GPU vectors. This is the single
  hook a new backend needs to implement.
- `rng` / `seed_rng!(seed)` — a shared `StableRNG(123)`. Drivers call `TestSuite.seed_rng!(123)`
  before each parameter combination so results are reproducible independent of test ordering.
- `precision(T)` — the default tolerance, `sqrt(eps(real(T)))`.
- Predicates used throughout the assertions: `isleftnull`, `isrightnull`, `isleftcomplete`,
  `isrightcomplete`, `has_positive_diagonal`.
- `instantiate_unitary`, `instantiate_rank_deficient_matrix` — inputs with prescribed structure.

`test/linearmap.jl` is a further helper, defining a `LinearMap` wrapper that is deliberately *not* an
`AbstractMatrix`, used to check the generic code paths. It is excluded from discovery and included
directly where needed.

## Hardware and platform gating

GPU tests are guarded at runtime rather than by dependency availability: CUDA and AMDGPU are
unconditional dependencies of the test environment, and the tests themselves are wrapped in
`CUDA.functional()` / `AMDGPU.functional()` blocks. Running the suite on a machine without a GPU
therefore skips them silently.

`runtests.jl` applies two further filters, both only when no explicit selector was given:

- **`BUILDKITE=true`** (the GPU CI runners) drops the files that carry no GPU coverage —
  `common/algorithms`, `common/codequality`, `common/truncate`, `decompositions/gen_eig`, and the
  `chainrules` group — so GPU runners spend their time on GPU code paths.
- **macOS CI** drops the `mooncake` and `chainrules` groups; **macOS and Windows CI** drop the
  `enzyme` group. These reflect AD backends that are unsupported or unreliable on those platforms.

## Adding a test

Create a `.jl` file inside one of the group directories. It is picked up automatically, runs in a
fresh worker process, and must therefore load everything it needs itself:

```julia
using MatrixAlgebraKit
using Test
using CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

if @isdefined(fast_tests) && fast_tests
    eltypes = (Float64, ComplexF64)
else
    eltypes = (Float32, Float64, ComplexF32, ComplexF64)
end

for T in eltypes
    TestSuite.seed_rng!(123)
    TestSuite.test_myop(Matrix{T}, (54, 37))
    CUDA.functional() && TestSuite.test_myop(CuMatrix{T}, (54, 37))
end
```

Add assertions that should be shared with downstream packages to `test/testsuite/` and call them
from the driver, rather than writing them inline.
