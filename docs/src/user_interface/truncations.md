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

## Combining Strategies

Truncation strategies can be combined using the `&` operator to create intersection-based truncation.
When strategies are combined, only the values that satisfy all conditions are kept.

For example, to keep at most 10 eigenvalues while also discarding all values below `1e-6`:

```julia
combined_trunc = truncrank(10) & trunctol(; atol = 1e-6)
```

## Using Truncations in Decompositions

Truncation strategies can be used with truncated decomposition functions in two ways:

### 1. Using the `trunc` keyword with a `NamedTuple`

The simplest approach is to pass a `NamedTuple` with the truncation parameters:

```julia
using MatrixAlgebraKit

# Create a symmetric matrix
A = randn(100, 100)
A = A + A'  # Make symmetric

# Keep only the 10 largest eigenvalues
D, V = eigh_trunc(A; trunc = (maxrank = 10,))

# Keep eigenvalues with absolute value above tolerance
D, V = eigh_trunc(A; trunc = (atol = 1e-6,))

# Combine multiple criteria
D, V = eigh_trunc(A; trunc = (maxrank = 20, atol = 1e-10, rtol = 1e-8))
```

### 2. Using explicit `TruncationStrategy` objects

For more control, you can construct `TruncationStrategy` objects directly:

```julia
# Keep the 5 largest eigenvalues
strategy = truncrank(5)
D, V = eigh_trunc(A; trunc = strategy)

# Keep eigenvalues above an absolute tolerance
strategy = trunctol(; atol = 1e-6)
D, V = eigh_trunc(A; trunc = strategy)

# Combine strategies: keep at most 10 eigenvalues, all above 1e-8
strategy = truncrank(10) & trunctol(; atol = 1e-8)
D, V = eigh_trunc(A; trunc = strategy)
```

## Complete Example

Here's a complete example demonstrating different truncation approaches:

```julia
using MatrixAlgebraKit
using LinearAlgebra

# Generate a test matrix with known spectrum
n = 50
A = randn(n, n)
A = A + A'  # Make symmetric

# 1. No truncation - keep all eigenvalues
D_full, V_full = eigh_trunc(A; trunc = nothing)
@assert size(D_full) == (n, n)

# 2. Keep only the 10 largest eigenvalues
D_rank, V_rank = eigh_trunc(A; trunc = (maxrank = 10,))
@assert size(D_rank) == (10, 10)
@assert size(V_rank) == (n, 10)

# 3. Keep eigenvalues with absolute value above a threshold
D_tol, V_tol = eigh_trunc(A; trunc = (atol = 1e-6,))
println("Kept $(size(D_tol, 1)) eigenvalues above tolerance")

# 4. Combine rank and tolerance truncation
strategy = truncrank(15) & trunctol(; atol = 1e-8)
D_combined, V_combined = eigh_trunc(A; trunc = strategy)
println("Kept $(size(D_combined, 1)) eigenvalues (max 15, all above 1e-8)")

# 5. Truncated SVD example
B = randn(100, 80)
U, S, Vh = svd_trunc(B; trunc = (maxrank = 20,))
@assert size(S) == (20, 20)
@assert size(U) == (100, 20)
@assert size(Vh) == (20, 80)

# Verify the truncated decomposition is accurate
@assert norm(B - U * S * Vh) ≈ norm(svd(B).S[21:end])
```

## Truncation with SVD vs Eigenvalue Decompositions

When using truncations with different decomposition types, keep in mind:

- **`svd_trunc`**: Singular values are always real and non-negative, sorted in descending order. Truncation by value typically keeps the largest singular values.

- **`eigh_trunc`**: Eigenvalues are real but can be negative for symmetric matrices. By default, `truncrank` sorts by absolute value, so `truncrank(k)` keeps the `k` eigenvalues with largest magnitude (positive or negative).

- **`eig_trunc`**: For general (non-symmetric) matrices, eigenvalues can be complex. Truncation by absolute value considers the complex magnitude.

## Advanced: Custom Truncation Filters

For specialized needs, you can use [`truncfilter`](@ref) to define custom selection criteria:

```julia
# Keep only positive eigenvalues
strategy = truncfilter(x -> x > 0)
D_positive, V_positive = eigh_trunc(A; trunc = strategy)

# Keep eigenvalues in a specific range
strategy = truncfilter(x -> 0.1 < abs(x) < 10.0)
D_range, V_range = eigh_trunc(A; trunc = strategy)
```
