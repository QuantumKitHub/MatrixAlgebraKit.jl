```@meta
CurrentModule = MatrixAlgebraKit
CollapsedDocStrings = true
```

# Truncations

Truncation strategies allow you to control which eigenvalues or singular values to keep when computing partial or truncated decompositions. These strategies are used in functions like [`eigh_trunc`](@ref), [`eig_trunc`](@ref), and [`svd_trunc`](@ref) to reduce the size of the decomposition while retaining the most important information.

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

```jldoctest
julia> using MatrixAlgebraKit

julia> combined_trunc = truncrank(10) & trunctol(; atol = 1e-6);

julia> typeof(combined_trunc)
MatrixAlgebraKit.TruncationIntersection{Tuple{MatrixAlgebraKit.TruncationByOrder{typeof(abs)}, MatrixAlgebraKit.TruncationByValue{Float64, Int64, typeof(abs)}}}
```

## Using Truncations in Decompositions

Truncation strategies can be used with truncated decomposition functions in two ways:

### 1. Using the `trunc` keyword with a `NamedTuple`

The simplest approach is to pass a `NamedTuple` with the truncation parameters:

```jldoctest truncations
julia> using MatrixAlgebraKit

julia> # Create a symmetric matrix with known values
       A = [4.0 2.0 1.0; 2.0 5.0 3.0; 1.0 3.0 6.0];

julia> # Keep only the 2 largest eigenvalues
       D, V = eigh_trunc(A; trunc = (maxrank = 2,));

julia> size(D)
(2, 2)

julia> size(V)
(3, 2)
```

You can also use tolerance-based truncation or combine multiple criteria:

```jldoctest truncations
julia> # Keep eigenvalues with absolute value above tolerance
       D, V = eigh_trunc(A; trunc = (atol = 1e-6,));

julia> size(D, 1)  # All eigenvalues are above 1e-6
3

julia> # Combine multiple criteria
       D, V = eigh_trunc(A; trunc = (maxrank = 2, atol = 1e-10));

julia> size(D)
(2, 2)
```

### 2. Using explicit `TruncationStrategy` objects

For more control, you can construct `TruncationStrategy` objects directly:

```jldoctest truncations
julia> # Keep the 2 largest eigenvalues
       strategy = truncrank(2);

julia> D, V = eigh_trunc(A; trunc = strategy);

julia> size(D)
(2, 2)

julia> # Combine strategies: keep at most 2 eigenvalues, all above 1e-8
       strategy = truncrank(2) & trunctol(; atol = 1e-8);

julia> D, V = eigh_trunc(A; trunc = strategy);

julia> size(D)
(2, 2)
```

## Complete Example

Here's a complete example demonstrating different truncation approaches:

```jldoctest complete_example
julia> using MatrixAlgebraKit, LinearAlgebra

julia> # Create a symmetric test matrix with known spectrum
       A = [10.0  2.0  1.0  0.5;
             2.0  8.0  1.5  0.3;
             1.0  1.5  6.0  0.2;
             0.5  0.3  0.2  4.0];

julia> # 1. No truncation - keep all eigenvalues
       D_full, V_full = eigh_trunc(A; trunc = nothing);

julia> size(D_full)
(4, 4)

julia> # 2. Keep only the 2 largest eigenvalues
       D_rank, V_rank = eigh_trunc(A; trunc = (maxrank = 2,));

julia> size(D_rank)
(2, 2)

julia> size(V_rank)
(4, 2)

julia> # 3. Keep eigenvalues with absolute value above a threshold
       D_tol, V_tol = eigh_trunc(A; trunc = (atol = 5.0,));

julia> size(D_tol, 1) >= 2  # At least 2 eigenvalues are above 5.0
true

julia> # 4. Combine rank and tolerance truncation
       strategy = truncrank(3) & trunctol(; atol = 1e-8);

julia> D_combined, V_combined = eigh_trunc(A; trunc = strategy);

julia> size(D_combined, 1) <= 3
true

julia> # 5. Truncated SVD example
       B = [3.0 2.0 1.0; 1.0 4.0 2.0; 2.0 1.0 5.0; 0.5 1.0 2.0];

julia> U, S, Vh = svd_trunc(B; trunc = (maxrank = 2,));

julia> size(S)
(2, 2)

julia> size(U)
(4, 2)

julia> size(Vh)
(2, 3)

julia> # Verify the truncated decomposition error equals the discarded singular values
       norm(B - U * S * Vh) ≈ norm(svd_vals(B)[3:end])
true
```

## Truncation with SVD vs Eigenvalue Decompositions

When using truncations with different decomposition types, keep in mind:

- **`svd_trunc`**: Singular values are always real and non-negative, sorted in descending order. Truncation by value typically keeps the largest singular values.

- **`eigh_trunc`**: Eigenvalues are real but can be negative for symmetric matrices. By default, `truncrank` sorts by absolute value, so `truncrank(k)` keeps the `k` eigenvalues with largest magnitude (positive or negative).

- **`eig_trunc`**: For general (non-symmetric) matrices, eigenvalues can be complex. Truncation by absolute value considers the complex magnitude.

## Advanced: Custom Truncation Filters

For specialized needs, you can use [`truncfilter`](@ref) to define custom selection criteria:

```jldoctest custom_filters
julia> using MatrixAlgebraKit

julia> A = [4.0 -1.0 2.0; -1.0 3.0 1.0; 2.0 1.0 -2.0];

julia> # Keep only positive eigenvalues
       strategy = truncfilter(x -> x > 0);

julia> D_positive, V_positive = eigh_trunc(A; trunc = strategy);

julia> size(D_positive, 1) >= 2  # At least 2 positive eigenvalues
true

julia> # Keep eigenvalues in a specific range
       strategy = truncfilter(x -> 1.0 < abs(x) < 5.0);

julia> D_range, V_range = eigh_trunc(A; trunc = strategy);

julia> size(D_range, 1) >= 1  # At least 1 eigenvalue in range
true
```
