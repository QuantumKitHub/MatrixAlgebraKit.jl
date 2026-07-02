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
The default algorithm [`MatrixFunctionViaTaylor`](@ref) is a pure-Julia scaling-and-squaring evaluation of the Taylor series.
As it requires no LAPACK support, it also applies to generic data types at arbitrary precision.
Alternatively, an implementation based on a Padé approximation is available in `LinearAlgebra`, and can be accessed by the algorithm [`MatrixFunctionViaLA`](@ref).
The exponential can also be calculated by first calculating the (hermitian) eigenvalue decomposition, and then computing the scalar exponential of the diagonal elements.
This strategy is implemented via the algorithms [`MatrixFunctionViaEig`](@ref) and [`MatrixFunctionViaEigh`](@ref), and call `eig_full` and `eigh_full`, respectively.
Additionally, in order to calculate `exp(τ * A)`, the function `exponential` can be called with `(τ, A)`, using the same algorithms as before.

```@docs; canonical=false
exponential
MatrixAlgebraKit.MatrixFunctionViaTaylor
MatrixAlgebraKit.MatrixFunctionViaLA
MatrixAlgebraKit.MatrixFunctionViaEig
MatrixAlgebraKit.MatrixFunctionViaEigh
```
