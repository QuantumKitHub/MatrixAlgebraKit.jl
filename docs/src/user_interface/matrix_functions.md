```@meta
CurrentModule = MatrixAlgebraKit
CollapsedDocStrings = true
```

# Matrix functions

Another class of matrix algebra methods consists of calculating some function of a single input `A`.
In order to streamline these functions, they all follow a similar common code pattern.
For a given function `f`, this consists of the following methods:

```julia
f(A; kwargs...) -> F...
f!(A, [F]; kwargs...) -> F...
```

Here, the input matrix is always the first argument, and optionally the output can be provided as well.
The keywords are algorithm-specific, and can be used to influence the behavior of the algorithms.
For a full description of how to select and configure algorithms, see [Algorithm Selection](@ref sec_algorithmselection).
Importantly, for generic code patterns it is recommended to always use the output `F` explicitly, rather than relying on the in-place functionality, since some implementations may not be able to reuse the provided memory.
Additionally, the `f!` method typically assumes that it is allowed to destroy the input `A`, and making use of the contents of `A` afterwards is undefined behavior.

## Algorithms

The matrix functions share a common set of algorithms, which differ in how they reduce the problem to a scalar function of the eigenvalues, along with more specialized implementations for specific functions:

- [`MatrixFunctionViaSchur`](@ref) applies to [`squareroot`](@ref) only, and evaluates the function on the (quasi-)triangular factor of a Schur decomposition computed through `schur_full`. It is backward stable, independent of the conditioning of the eigenbasis, and applies to generic data types at arbitrary precision, but it indexes individual entries and is therefore unsuited to GPU arrays.
- [`MatrixFunctionViaEig`](@ref) and [`MatrixFunctionViaEigh`](@ref) first compute an eigenvalue decomposition, through `eig_full` and `eigh_full` respectively, and then apply the scalar function to the eigenvalues. The latter requires a hermitian input, and in return its result is hermitian by construction.
- [`MatrixFunctionViaTaylor`](@ref) applies to [`exponential`](@ref) only, and evaluates its Taylor series through scaling and squaring. As it requires no LAPACK support, it also applies to generic data types at arbitrary precision.
- [`MatrixFunctionViaLA`](@ref) defers to the implementation of `LinearAlgebra`, which is a Padé approximation for [`exponential`](@ref). For [`squareroot`](@ref) it dispatches on the structure of the input rather than on a fixed strategy, and is Schur-based only for a general matrix: a hermitian one is routed through an eigenvalue decomposition instead, with a tolerance of its own that `domain_atol` cannot reach, and a real matrix at other than BLAS precision is promoted to a complex Schur form, so that a real result is not recovered even when it exists.
- [`DiagonalAlgorithm`](@ref) is the fast path for a `Diagonal` input, and simply maps the scalar function over the diagonal.

```@docs; canonical=false
MatrixAlgebraKit.MatrixFunctionViaTaylor
MatrixAlgebraKit.MatrixFunctionViaSchur
MatrixAlgebraKit.MatrixFunctionViaLA
MatrixAlgebraKit.MatrixFunctionViaEig
MatrixAlgebraKit.MatrixFunctionViaEigh
```

## Exponential

The [exponential](https://en.wikipedia.org/wiki/Matrix_exponential) of a square matrix `A` is used in many scientific applications, as it arises in the solution of an autonomous linear differential equation.
It is defined for every square matrix, so the [domain considerations](@ref sec_matrixfunction_domain) below do not apply to it.
The default algorithm is [`MatrixFunctionViaTaylor`](@ref), which is the only one that also covers generic data types at arbitrary precision.
Additionally, in order to calculate `exp(τ * A)`, the function `exponential` can be called with `(τ, A)`, using the same algorithms.

```@docs; canonical=false
exponential
```

## Square root

The principal [square root](https://en.wikipedia.org/wiki/Square_root_of_a_matrix) of a square matrix `A` is the unique square root whose eigenvalues have nonnegative real part.
It is computed by the function [`squareroot`](@ref), with [`MatrixFunctionViaSchur`](@ref) as the default algorithm, and is subject to the [domain considerations](@ref sec_matrixfunction_domain) below.

```@docs; canonical=false
squareroot
```

## [Domain considerations](@id sec_matrixfunction_domain)

Not every matrix function is defined for every square matrix, for example a real [`squareroot`](@ref) requires the eigenvalues to avoid the negative real axis, and its principal value is complex whenever eigenvalues on that axis are present.
In MatrixAlgebraKit, we aim to keep type stability, and thus the scalar type of the output always matches that of the input.
As such, a real matrix with eigenvalues on the negative real axis leads to a `DomainError`, and a complex matrix should be passed instead.

The hard part is that eigenvalues are *computed*, and thus contain some inaccuracy from the method used to compute them.
Typically it can be beneficial to introduce some tolerance to compare the domain with, which is controlled by the `domain_atol` keyword.
Clamping these values does come at a cost, as e.g. an eigenvalue at `-δ` perturbs the square root by `O(√δ)`, so an accepted result computed at the default tolerance can differ from the exact principal value by considerably more than the tolerance itself.

`domain_atol` defaults to [`default_domain_atol`](@ref), i.e. `n * eps * maximum(abs, λ)`, which is the accumulated roundoff of a spectrum computed to hermitian accuracy.
This is the same rule as `LinearAlgebra.sqrt(::Hermitian; rtol = eps(T) * size(A, 1))`, so for hermitian input MatrixAlgebraKit and `LinearAlgebra` accept and reject the same matrices.
The eigenvalues of [`MatrixFunctionViaEig`](@ref) are additionally limited by the conditioning of the eigenvectors, so for a poorly conditioned eigenbasis a larger `domain_atol` may have to be set explicitly.
The same default is used for the never user-settable tolerance with which a complex eigenvalue of a real matrix is decided to lie *on* the negative real axis, as that is a question about the accuracy of the eigensolver rather than about the domain.

[`MatrixFunctionViaSchur`](@ref) needs no such tolerance at all: a real eigenvalue of a real matrix is a `1×1` block of its real Schur form, whereas a `2×2` block is a genuine complex-conjugate pair, which lies off the negative real axis and always has a real square root.
The block structure of the decomposition therefore answers exactly the question that the other algorithms have to settle numerically.

Additionally, not all algorithms have acces to the spectrum, so not all methods are suitable for eigenvalues close to the domain edges.
For example, [`MatrixFunctionViaLA`](@ref) defers to `LinearAlgebra`, which decides internally whether a real result exists and hands back a complex matrix when it does not.
There are no eigenvalues to compare against anything, so it rejects a complex result for a real input outright, and passing it `domain_atol` is an error rather than a silent no-op.

!!! warning "`MatrixFunctionViaEig` and defective matrices"
    The eigenvalues of a Jordan block of size `k` are resolved only to `eps^(1/k)`, which exceeds every tolerance on this page.
    A real matrix with a defective negative eigenvalue can therefore have its spectrum reported as a complex-conjugate pair well off the axis, be judged in domain, and yield a result whose imaginary part is silently discarded.
    This is not specific to the domain test: `MatrixFunctionViaEig` reconstructs `f(A)` by inverting the eigenvector matrix, so for a defective or nearly defective matrix its result is unreliable whether the input is real or complex.
    Use the Schur-based [`MatrixFunctionViaSchur`](@ref), the default, for such matrices.
    Note that a defective eigenvalue sitting *on* the negative real axis is ill-conditioned for every algorithm, since the computed copies of it straddle the branch cut of the scalar square root.

```@docs; canonical=false
MatrixAlgebraKit.default_domain_atol
```
