```@meta
CurrentModule = MatrixAlgebraKit
CollapsedDocStrings = true
```

# Sketching

*Sketching* methods project a large matrix onto a low-dimensional subspace before any expensive dense factorization is performed.
The result is an approximate low-rank decomposition that can be substantially cheaper than the corresponding full decomposition, depending on the quality of the sketch and the spectrum of the original matrix.

This page first describes how to compute a sketch as a standalone operation, then how to plug a sketch into a partial decomposition such as a truncated SVD.

## Standalone Sketches

The basic sketching primitives are [`left_sketch`](@ref) and [`right_sketch`](@ref) (with their bang counterparts).
They return an isometric range/co-range factor together with a corresponding core matrix, independent of any subsequent decomposition.

[`left_sketch`](@ref) computes an isometric matrix `Q` whose column span approximates the range of `A`, together with the core factor `B = Q' * A`:

```jldoctest sketching; output=false
using MatrixAlgebraKit
using MatrixAlgebraKit: diagview
using LinearAlgebra: norm
using Random: MersenneTwister

# A rank-3 matrix
A = randn(MersenneTwister(0), 8, 3) * randn(MersenneTwister(1), 3, 6);

Q, B = left_sketch(A; howmany = 3, rng = MersenneTwister(42));
isisometric(Q) && A ≈ Q * B

# output
true
```

[`right_sketch`](@ref) is the dual operation, returning a right-isometric matrix `Pᴴ` (orthonormal rows) and core factor `B = A * Pᴴ'`:

```jldoctest sketching; output=false
B, Pᴴ = right_sketch(A; howmany = 3, rng = MersenneTwister(42));
isisometric(Pᴴ') && A ≈ B * Pᴴ

# output
true
```

These functions follow the same conventions as other decompositions: `left_sketch!` / `right_sketch!` may destroy the input, and the bang forms optionally accept pre-allocated output tuples.

```@docs; canonical=false
left_sketch
left_sketch!
right_sketch
right_sketch!
```

## Available Sketching Strategies

The keyword arguments accepted by `left_sketch` / `right_sketch` are forwarded to the default sketching strategy for the input type (currently [`GaussianSketching`](@ref) for all `AbstractMatrix`).
For full control, construct a strategy directly and pass it as the second positional argument:

```jldoctest sketching; output=false
Q, B = left_sketch(A, GaussianSketching(3; numiter = 4, rng = MersenneTwister(42)));
A ≈ Q * B

# output
true
```

```@docs; canonical=false
GaussianSketching
SketchingStrategy
```

The `numiter` keyword controls the number of power iterations.
The first iteration is the initial sketch; additional iterations apply `A * A'` to improve accuracy on slowly-decaying spectra at the cost of two extra matrix multiplications per iteration.
The default `numiter = 2` is a conservative choice; values of 4–5 often improve accuracy significantly when the singular values decay slowly.

!!! note "Additional strategies"
    [`GaussianSketching`](@ref) is currently the only built-in [`SketchingStrategy`](@ref).
    Additional strategies (for example, structured or subsampled sketches) may be added in the future; the interface is deliberately written against the abstract [`SketchingStrategy`](@ref) supertype so that new strategies plug in without changes to the downstream decomposition code.

## Sketched Partial Decompositions

A sketch can be combined with a small dense decomposition of the core factor to obtain an approximate partial decomposition of the original matrix.
At present, this is supported for the truncated SVD via [`svd_trunc`](@ref) / [`svd_trunc!`](@ref) / [`svd_trunc_no_error`](@ref).

!!! note "Not yet supported"
    Sketched variants of [`eigh_trunc`](@ref) and [`eig_trunc`](@ref) are natural extensions of the same machinery but are not implemented yet.
    The [`SketchedAlgorithm`](@ref) wrapper and the `sketch =` keyword are currently only accepted by the truncated SVD functions.

There are two equivalent ways to request a sketched truncated SVD, paralleling the two-form syntax used for [Truncations](@ref).

### 1. Using the `sketch` keyword with a `NamedTuple`

The simplest form is to pass a `NamedTuple` of sketch parameters together with the desired truncation:

```jldoctest sketching; output=false
U, S, Vᴴ, ϵ = svd_trunc(A;
    sketch = (; howmany = 3, rng = MersenneTwister(42)),
    trunc = truncrank(3),
);
size(diagview(S), 1) == 3 && A ≈ U * S * Vᴴ

# output
true
```

The `NamedTuple` keywords are forwarded to the default sketching strategy for the input type, exactly as for `left_sketch` above.

### 2. Using an explicit `SketchedAlgorithm`

For full control, construct a [`SketchedAlgorithm`](@ref) value directly and pass it as the `alg` argument:

```jldoctest sketching; output=false
alg = SketchedAlgorithm(;
    sketch = GaussianSketching(3; rng = MersenneTwister(42)),
    trunc = truncrank(3),
);
U, S, Vᴴ, ϵ = svd_trunc(A, alg);
A ≈ U * S * Vᴴ

# output
true
```

When an `alg::SketchedAlgorithm` is supplied, the `sketch` and `trunc` keywords cannot also be specified at the call site; doing so raises `ArgumentError`.
All configuration must instead live inside the algorithm constructor.

### The `SketchedAlgorithm` Wrapper

```@docs; canonical=false
SketchedAlgorithm
```

`SketchedAlgorithm` differs from [`TruncatedAlgorithm`](@ref) in that it is *self-truncating*: the sketch step itself produces a small dense problem of size `sketch.howmany`, and any further `trunc` is applied to the result of the inner factorization rather than to a full dense decomposition.

The `driver` field selects the backend implementing the sketched pipeline:

- `Native()` (the default for CPU array types) runs the generic *sketch-then-decompose* pipeline using the standard [`left_sketch!`](@ref) / [`right_sketch!`](@ref) building blocks followed by the inner `alg`.
- `CUSOLVER()` (the default for CUDA array types, with the appropriate extension loaded) dispatches to cuSOLVER's fused `gesvdr` kernel, which performs the sketch and the small SVD in a single device call.

To force a particular driver, set it explicitly:

```julia
using MatrixAlgebraKit: CUSOLVER  # driver types are not exported by default

alg = SketchedAlgorithm(;
    sketch = GaussianSketching(k; numiter = 4),
    trunc = truncrank(k),
    driver = CUSOLVER(),
)
U, S, Vᴴ, ϵ = svd_trunc(A_cuda, alg)
```
