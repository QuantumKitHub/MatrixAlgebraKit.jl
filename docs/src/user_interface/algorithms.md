```@meta
CurrentModule = MatrixAlgebraKit
CollapsedDocStrings = true
```

# [Algorithm Selection](@id sec_algorithmselection)

All factorization functions in MatrixAlgebraKit accept an optional `alg` keyword argument that controls which algorithm is used and how it is configured.
By default, an appropriate algorithm is selected automatically based on the function and the array type.
This page explains how to override that default, what algorithm types are available, and how to configure them.

## The `alg` Keyword

The `alg` keyword is interpreted by [`MatrixAlgebraKit.select_algorithm`](@ref), which accepts five forms, e.g.:

```julia
# Form 1: No alg — algorithm selected automatically based on function and array type.
Q, R = qr_compact(A);

# Form 2: Symbol — creates Algorithm{:Householder}(; positive=false).
Q, R = qr_compact(A; alg = :Householder, positive = false);

# Form 3: Algorithm type — calls Householder(; positive=false).
Q, R = qr_compact(A; alg = Householder, positive = false);

# Form 4: Algorithm instance — used as-is; no additional kwargs are allowed.
Q, R = qr_compact(A; alg = Householder(; positive = false));

# Form 5: NamedTuple — equivalent to qr_compact(A; positive=false).
Q, R = qr_compact(A; alg = (; positive = false));
```

!!! note
    When passing an already-constructed algorithm instance (form 4), additional keyword arguments at the call site are not permitted and will throw an `ArgumentError`.
    All configuration must go into the constructor in that case.

```@docs; canonical=false
MatrixAlgebraKit.select_algorithm
```

## Discovering the Default Algorithm

To check which algorithm is used by default for a given function and array type, call [`MatrixAlgebraKit.default_algorithm`](@ref).
The available keyword arguments depend on the algorithm type; refer to the docstrings listed in [Available Algorithm Types](@ref) below.

```@docs; canonical=false
MatrixAlgebraKit.default_algorithm
```

## Configuring Algorithms

Each algorithm accepts keyword arguments that control its behavior.
These can be provided either at the call site (forms 1–3) or inside the algorithm constructor:

```julia
# The following four calls are all equivalent:
U, S, Vᴴ = svd_compact(A; fixgauge = false)
U, S, Vᴴ = svd_compact(A; alg = :SafeDivideAndConquer, fixgauge = false)
U, S, Vᴴ = svd_compact(A; alg = SafeDivideAndConquer, fixgauge = false)
U, S, Vᴴ = svd_compact(A; alg = SafeDivideAndConquer(; fixgauge = false))
```

## The `DefaultAlgorithm` Sentinel

Package developers who want to store algorithm configuration without committing to a specific algorithm can use `DefaultAlgorithm`.
It defers algorithm selection to call time, forwarding its stored keyword arguments to [`MatrixAlgebraKit.select_algorithm`](@ref):

```julia
# Store configuration without picking a specific algorithm:
alg = DefaultAlgorithm(; positive = true)

# Equivalent to qr_compact(A; positive = true):
Q, R = qr_compact(A; alg)
```

```@docs; canonical=false
DefaultAlgorithm
```

## Available Algorithm Types

The following high-level algorithm types are available.
They all accept an optional `driver` keyword to select the computational backend; see [Driver Selection](@ref sec_driverselection) for details.

| Algorithm | Applicable decompositions | Key keyword arguments |
|:----------|:--------------------------|:----------------------|
| [`Householder`](@ref) | QR, LQ | `positive`, `pivoted`, `blocksize` |
| [`DivideAndConquer`](@ref) | SVD, eigh | `fixgauge` |
| [`SafeDivideAndConquer`](@ref) | SVD, eigh | `fixgauge` |
| [`QRIteration`](@ref) | SVD, eigh, eig, Schur | `fixgauge`, `expert`, `permute`, `scale` |
| [`Bisection`](@ref) | eigh, SVD | `fixgauge` |
| [`Jacobi`](@ref) | eigh, SVD | `fixgauge` |
| [`RobustRepresentations`](@ref) | eigh | `fixgauge` |
| [`SVDViaPolar`](@ref) | SVD | `fixgauge`, `tol` |
| [`PolarViaSVD`](@ref) | polar | positional `svd_alg` argument |
| [`PolarNewton`](@ref) | polar | `maxiter`, `tol` |

For full docstring details on each algorithm type, see the corresponding section in [Decompositions](@ref).

## [Driver Selection](@id sec_driverselection)

!!! note "Expert use case"
    Selecting a specific driver is an advanced feature intended for users who need to target a specific computational backend, such as a GPU.
    For most use cases, the default driver selection is sufficient.

Each algorithm in MatrixAlgebraKit can optionally accept a `driver` keyword argument to explicitly select the computational backend.
By default, the driver is set to `DefaultDriver()`, which automatically selects the most appropriate backend based on the input matrix type.
The available drivers are:

```@autodocs; canonical=false
Modules = [MatrixAlgebraKit]
Filter = t -> t isa Type && t <: MatrixAlgebraKit.Driver
```

For example, to force LAPACK for a generic matrix type, or to use a GPU backend:

```julia
using MatrixAlgebraKit
using MatrixAlgebraKit: LAPACK, CUSOLVER  # driver types are not exported by default

# Default: driver is selected automatically based on the input type
U, S, Vᴴ = svd_compact(A)
U, S, Vᴴ = svd_compact(A; alg = SafeDivideAndConquer())

# Expert: explicitly select LAPACK
U, S, Vᴴ = svd_compact(A; alg = SafeDivideAndConquer(; driver = LAPACK()))

# Expert: use a GPU backend (requires loading the appropriate extension)
U, S, Vᴴ = svd_compact(A; alg = QRIteration(; driver = CUSOLVER()))
```

Similarly, for QR decompositions:

```julia
using MatrixAlgebraKit: LAPACK  # driver types are not exported by default

# Default: driver is selected automatically
Q, R = qr_compact(A)
Q, R = qr_compact(A; alg = Householder())

# Expert: explicitly select a driver
Q, R = qr_compact(A; alg = Householder(; driver = LAPACK()))
```
