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
Importantly, for generic code patterns it is recommended to always use the output `F` explicitly, since some implementations may not be able to reuse the provided memory.
Additionally, the `f!` method typically assumes that it is allowed to destroy the input `A`, and making use of the contents of `A` afterwards should be deemed as undefined behavior.

## Exponential

The [exponential](https://en.wikipedia.org/wiki/Matrix_exponential) of a square matrix `A` is used in many scientific applications, as it arises in the solution of an autonomous linear differential equation.
An implementation for the matrix exponential based on a Padé approximation is available in `LinearAlgebra`, and can be accessed by the algorithm [`MatrixFunctionViaLA`](@ref).
For more generic data types, the exponential can be calculated by first calculating the (hermitian) eigenvalue decomposition, and then computing
the scalar exponential of the diagonal elements.
This strategy is implemented via the algorithms [`MatrixFunctionViaEig`](@ref) and [`MatrixFunctionViaEigh`](@ref), and call `eig_full` and `eigh_full`, respectively.
Additionally, in order to calculate `exp(τ * A)`, the function `exponential` can be called with `(τ, A)`, using the same algorithms as before.

```@docs; canonical=false
exponential
MatrixAlgebraKit.MatrixFunctionViaLA
MatrixAlgebraKit.MatrixFunctionViaEig
MatrixAlgebraKit.MatrixFunctionViaEigh
```

## Domain considerations

The functions below ([`squareroot`](@ref), [`logarithm`](@ref) and [`power`](@ref) with fractional powers) are only defined for matrices whose eigenvalues avoid (part of) the negative real axis, and their principal values are complex whenever eigenvalues on that axis are present.
In MatrixAlgebraKit, the scalar type of the output always matches that of the input, and a real matrix with eigenvalues on the negative real axis therefore leads to a `DomainError`; pass a complex matrix instead to obtain the complex principal value.
To avoid spurious errors for eigenvalues that lie on the negative real axis only because of rounding errors (e.g. a positive semidefinite matrix with a tiny negative eigenvalue), eigenvalues within an absolute tolerance `domain_atol` of the domain boundary are clamped onto it.
This tolerance defaults to [`default_domain_atol`](@ref) and can be specified explicitly for the algorithms that support it, e.g. `MatrixFunctionViaEigh(eigh_alg; domain_atol=...)`.

```@docs; canonical=false
MatrixAlgebraKit.default_domain_atol
```

## Square root

The principal [square root](https://en.wikipedia.org/wiki/Square_root_of_a_matrix) of a square matrix `A` is the unique square root whose eigenvalues have nonnegative real part.
It is computed by the function [`squareroot`](@ref), where the default algorithm [`MatrixFunctionViaLA`](@ref) wraps the Schur-based implementation of `LinearAlgebra`, and the eigenvalue-decomposition-based algorithms [`MatrixFunctionViaEig`](@ref) and [`MatrixFunctionViaEigh`](@ref) are available as well.

```@docs; canonical=false
squareroot
```

## Logarithm

The principal [logarithm](https://en.wikipedia.org/wiki/Logarithm_of_a_matrix) of a square matrix `A` is the unique logarithm whose eigenvalues have imaginary part in `(-π, π]`, and exists for matrices without (numerically) zero eigenvalues that satisfy the domain considerations above.
It is computed by the function [`logarithm`](@ref), with the same algorithm choices as [`squareroot`](@ref).

```@docs; canonical=false
logarithm
```

## Power

Matrix powers `A^p` for real `p` are computed by the function [`power`](@ref), which takes the exponent as a second positional argument.
Integer powers are defined for any square matrix (invertible for negative powers) and reduce to repeated multiplication, while fractional powers are principal powers subject to the domain considerations above.

```@docs; canonical=false
power
```
