```@meta
CurrentModule = MatrixAlgebraKit
CollapsedDocStrings = true
```

# Truncations

Truncation strategies allow you to control which eigenvalues or singular values to keep when computing partial or truncated decompositions. These strategies are used in the functions [`eigh_trunc`](@ref), [`eig_trunc`](@ref), and [`svd_trunc`](@ref) to reduce the size of the decomposition while retaining the most important information.

## Using Truncations in Decompositions

Truncation strategies can be used with truncated decomposition functions in two ways, as illustrated below.
For concreteness, we use the following matrix as an example:

```jldoctest truncations
using MatrixAlgebraKit
using MatrixAlgebraKit: diagview

A = [2 1 0; 1 3 1; 0 1 4];
D, V = eigh_full(A);

diagview(D) ≈ [3 - √3, 3, 3 + √3]

# output

true
```

### 1. Using the `trunc` keyword with a `NamedTuple`

The simplest approach is to pass a `NamedTuple` with the truncation parameters.
For example, keeping only the largest 2 eigenvalues:

```jldoctest truncations
Dtrunc, Vtrunc = eigh_trunc(A; trunc = (maxrank = 2,));
size(Dtrunc, 1) <= 2

# output

true
```

Note however that there are no guarantees on the order of the output values:

```jldoctest truncations
diagview(Dtrunc) ≈ diagview(D)[[3, 2]]

# output

true
```

You can also use tolerance-based truncation or combine multiple criteria:

```jldoctest truncations
Dtrunc, Vtrunc = eigh_trunc(A; trunc = (atol = 2.9,));
all(>(2.9), diagview(Dtrunc))

# output

true
```

```jldoctest truncations
Dtrunc, Vtrunc = eigh_trunc(A; trunc = (maxrank = 2, atol = 2.9));
size(Dtrunc, 1) <= 2 && all(>(2.9), diagview(Dtrunc))

# output
true
```

In general, the keyword arguments that are supported can be found in the `TruncationStrategy` docstring:

```@docs; canonical = false
TruncationStrategy
```


### 2. Using explicit `TruncationStrategy` objects

For more control, you can construct [`TruncationStrategy`](@ref) objects directly.
This is also what the previous syntax will end up calling.

```jldoctest truncations
Dtrunc, Vtrunc = eigh_trunc(A; trunc = truncrank(2))
size(Dtrunc, 1) <= 2

# output

true
```

```jldoctest truncations
Dtrunc, Vtrunc = eigh_trunc(A; trunc = truncrank(2) & trunctol(; atol = 2.9))
size(Dtrunc, 1) <= 2 && all(>(2.9), diagview(Dtrunc))

# output
true
```

## Truncation with SVD vs Eigenvalue Decompositions

When using truncations with different decomposition types, keep in mind:

- **`svd_trunc`**: Singular values are always real and non-negative, sorted in descending order. Truncation by value typically keeps the largest singular values.

- **`eigh_trunc`**: Eigenvalues are real but can be negative for symmetric matrices. By default, `truncrank` sorts by absolute value, so `truncrank(k)` keeps the `k` eigenvalues with largest magnitude (positive or negative).

- **`eig_trunc`**: For general (non-symmetric) matrices, eigenvalues can be complex. Truncation by absolute value considers the complex magnitude.

## Truncation Strategies

MatrixAlgebraKit provides several built-in truncation strategies:

```@docs; canonical=false
notrunc
truncrank
trunctol
truncfilter
truncerror
```

Truncation strategies can be combined using the `&` operator to create intersection-based truncation.
When strategies are combined, only the values that satisfy all conditions are kept.

```julia
combined_trunc = truncrank(10) & trunctol(; atol = 1e-6);
```

