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
