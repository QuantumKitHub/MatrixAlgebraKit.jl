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

The [exponential](https://en.wikipedia.org/wiki/Matrix_exponential) of a rectangular matrix `A` is used in many quantum many-body problems.
This is implemented in `LinearAlgebra`, which can be accessed by the algorithm [`MatrixFunctionViaLA`](@ref).
For more generic data types, the exponential can be calculated by first calculating the (hermitian) eigenvalue decomposition.
This is done by `eig_full` and `eigh_full` via the algorithms [`MatrixFunctionViaEig`](@ref) and [`MatrixFunctionViaEigh`](@ref), respectively.
 Additionally, in order to calculate `exp(τ * A)`, the function `exponential` can be called with `(τ, A)`, using the same algorithms as before.

```@docs; canonical=false
exponential
```

```@autodocs; canonical=false
Modules = [MatrixAlgebraKit]
Filter = t -> t isa Type && (t <: MatrixAlgebraKit.MatrixFunctionViaLA || t <: MatrixAlgebraKit.MatrixFunctionViaEig || t <: MatrixAlgebraKit.MatrixFunctionViaEigh)
```
