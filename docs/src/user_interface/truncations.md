```@meta
CurrentModule = MatrixAlgebraKit
CollapsedDocStrings = true
```

# Truncations

Currently, truncations are supported through the following different methods:

```@docs; canonical=false
notrunc
truncrank
trunctol
truncfilter
truncerror
```

It is additionally possible to combine truncation strategies by making use of the `&` operator.
For example, truncating to a maximal dimension `10`, and discarding all values below `1e-6` would be achieved by:

```julia
maxdim = 10
atol = 1e-6
combined_trunc = truncrank(maxdim) & trunctol(; atol)
```

## Truncation Error

When using truncated decompositions such as [`svd_trunc`](@ref), [`eig_trunc`](@ref), or [`eigh_trunc`](@ref),
an additional truncation error value is returned. This error is defined as the 2-norm of the discarded 
singular values or eigenvalues, providing a measure of the approximation quality.

For example:
```julia
U, S, Vᴴ, ϵ = svd_trunc(A; trunc=truncrank(10))
# ϵ is the 2-norm of the discarded singular values
```
