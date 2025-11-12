```@meta
CurrentModule = MatrixAlgebraKit
CollapsedDocStrings = true
```

# Decompositions

A rather large class of matrix algebra methods consists of taking a single input `A`, and determining some factorization of that input.
In order to streamline these functions, they all follow a similar common code pattern.
For a given factorization `f`, this consists of the following methods:

```julia
f(A; kwargs...) -> F...
f!(A, [F]; kwargs...) -> F...
```

Here, the input matrix is always the first argument, and optionally the output can be provided as well.
The keywords are algorithm-specific, and can be used to influence the behavior of the algorithms.
Importantly, for generic code patterns it is recommended to always use the output `F` explicitly, since some implementations may not be able to reuse the provided memory.
Additionally, the `f!` method typically assumes that it is allowed to destroy the input `A`, and making use of the contents of `A` afterwards should be deemed as undefined behavior.

## QR and LQ Decompositions

The [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition) transforms a matrix `A` into a product `Q * R`, where `Q` is orthonormal and `R` upper triangular.
This is often used to solve linear least squares problems, or construct orthogonal bases, since it is typically less expensive than the [Singular Value Decomposition](@ref).
If the input `A` is invertible, `Q` and `R` are unique if we require the diagonal elements of `R` to be positive.

For rectangular matrices `A` of size `(m, n)`, there are two modes of operation, [`qr_full`](@ref) and [`qr_compact`](@ref).
The former ensures that the resulting `Q` is a square unitary matrix of size `(m, m)`, while the latter creates an isometric `Q` of size `(m, min(m, n))`.

Similarly, the [LQ decomposition](https://en.wikipedia.org/wiki/LQ_decomposition) transforms a matrix `A` into a product `L * Q`, where `L` is lower triangular and `Q` orthonormal.
This is equivalent to the *transpose* of the QR decomposition of the *transpose* matrix, but can be computed directly.
Again there are two modes of operation, [`lq_full`](@ref) and [`lq_compact`](@ref), with the same behavior as the QR decomposition.

```@docs; canonical=false
qr_full
qr_compact
lq_full
lq_compact
```

Alongside these functions, we provide a LAPACK-based implementation for dense arrays, as provided by the following algorithm:

```@docs; canonical=false
LAPACK_HouseholderQR
LAPACK_HouseholderLQ
```

## Eigenvalue Decomposition

The [Eigenvalue Decomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix) transforms a square matrix `A` into a product `V * D * V⁻¹`.
Equivalently, it finds `V` and `D` that satisfy `A * V = V * D`.

Not all matrices can be diagonalized, and some real matrices can only be diagonalized using complex arithmetic.
In particular, the resulting decomposition can only guaranteed to be real for real symmetric inputs `A`.
Therefore, we provide `eig_` and `eigh_` variants, where `eig` always results in complex-valued `V` and `D`, while `eigh` requires symmetric inputs but retains the scalartype of the input.

The full set of eigenvalues and eigenvectors can be computed using the [`eig_full`](@ref) and [`eigh_full`](@ref) functions.
If only the eigenvalues are required, the [`eig_vals`](@ref) and [`eigh_vals`](@ref) functions can be used.
These functions return the diagonal elements of `D` in a vector.

Finally, it is also possible to compute a partial or truncated eigenvalue decomposition, using the [`eig_trunc`](@ref) and [`eigh_trunc`](@ref) functions.
To control the behavior of the truncation, we refer to [Truncations](@ref) for more information.

### Symmetric Eigenvalue Decomposition

For symmetric matrices, we provide the following functions:

```@docs; canonical=false
eigh_full
eigh_trunc
eigh_vals
```

!!! note "Gauge Degrees of Freedom"
    The eigenvectors returned by these functions have residual phase degrees of freedom.
    By default, MatrixAlgebraKit applies a gauge fixing convention to ensure reproducible results.
    See [Gauge choices](@ref sec_gaugefix) for more details.

Alongside these functions, we provide a LAPACK-based implementation for dense arrays, as provided by the following algorithms:

```@autodocs; canonical=false
Modules = [MatrixAlgebraKit]
Filter = t -> t isa Type && t <: MatrixAlgebraKit.LAPACK_EighAlgorithm
```

### Eigenvalue Decomposition

For general matrices, we provide the following functions:

```@docs; canonical=false
eig_full
eig_trunc
eig_vals
```

!!! note "Gauge Degrees of Freedom"
    The eigenvectors returned by these functions have residual phase degrees of freedom.
    By default, MatrixAlgebraKit applies a gauge fixing convention to ensure reproducible results.
    See [Gauge choices](@ref sec_gaugefix) for more details.

Alongside these functions, we provide a LAPACK-based implementation for dense arrays, as provided by the following algorithms:

```@autodocs; canonical=false
Modules = [MatrixAlgebraKit]
Filter = t -> t isa Type && t <: MatrixAlgebraKit.LAPACK_EigAlgorithm
```

## Schur Decomposition

The [Schur decomposition](https://en.wikipedia.org/wiki/Schur_decomposition) transforms a complex square matrix `A` into a product `Q * T * Qᴴ`, where `Q` is unitary and `T` is upper triangular.
It rewrites an arbitrary complex square matrix as unitarily similar to an upper triangular matrix whose diagonal elements are the eigenvalues of `A`.
For real matrices, the same decomposition can be achieved in real arithmetic by allowing `T` to be quasi-upper triangular, i.e. triangular with blocks of size `(1, 1)` and `(2, 2)` on the diagonal.

This decomposition is also useful for computing the eigenvalues of a matrix, which is exposed through the [`schur_vals`](@ref) function.

```@docs; canonical=false
schur_full
schur_vals
```

The LAPACK-based implementation for dense arrays is provided by the following algorithms:

```@autodocs; canonical=false
Modules = [MatrixAlgebraKit]
Filter = t -> t isa Type && t <: MatrixAlgebraKit.LAPACK_EigAlgorithm
```

## Singular Value Decomposition

The [Singular Value Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) transforms a matrix `A` into a product `U * Σ * Vᴴ`, where `U` and `Vᴴ` are unitary, and `Σ` is diagonal, real and non-negative.
For a square matrix `A`, both `U` and `Vᴴ` are unitary, and if the singular values are distinct, the decomposition is unique.

For rectangular matrices `A` of size `(m, n)`, there are two modes of operation, [`svd_full`](@ref) and [`svd_compact`](@ref).
The former ensures that the resulting `U`, and `Vᴴ` remain square unitary matrices, of size `(m, m)` and `(n, n)`, with rectangular `Σ` of size `(m, n)`.
The latter creates an isometric `U` of size `(m, min(m, n))`, and `V = (Vᴴ)'` of size `(n, min(m, n))`, with a square `Σ` of size `(min(m, n), min(m, n))`.

It is also possible to compute the singular values only, using the [`svd_vals`](@ref) function.
This then returns a vector of the values on the diagonal of `Σ`.

Finally, we also support computing a partial or truncated SVD, using the [`svd_trunc`](@ref) function.

```@docs; canonical=false
svd_full
svd_compact
svd_vals
svd_trunc
```

!!! note "Gauge Degrees of Freedom"
    The singular vectors returned by these functions have residual phase degrees of freedom.
    By default, MatrixAlgebraKit applies a gauge fixing convention to ensure reproducible results.
    See [Gauge choices](@ref sec_gaugefix) for more details.

MatrixAlgebraKit again ships with LAPACK-based implementations for dense arrays:

```@autodocs; canonical=false
Modules = [MatrixAlgebraKit]
Filter = t -> t isa Type && t <: MatrixAlgebraKit.LAPACK_SVDAlgorithm
```

## Polar Decomposition

The [Polar Decomposition](https://en.wikipedia.org/wiki/Polar_decomposition) of a matrix `A` is a factorization `A = W * P`, where `W` is unitary and `P` is positive semi-definite.
If `A` is invertible (and therefore square), the polar decomposition always exists and is unique.
For non-square matrices `A` of size `(m, n)`, the decomposition `A = W * P` with `P` positive semi-definite of size `(n, n)` and `W` isometric of size `(m, n)` exists only if `m >= n`, and is unique if `A` and thus `P` is full rank.
For `m <= n`, we can analoguously decompose `A` as `A = P * Wᴴ` with `P` positive semi-definite of size `(m, m)` and `Wᴴ` of size `(m, n)` such that `W = (Wᴴ)'` is isometric. Only in the case `m = n` do both decompositions exist.

The decompositions `A = W * P` or `A = P * Wᴴ` can be computed with the [`left_polar`](@ref) and [`right_polar`](@ref) functions, respectively.

```@docs; canonical=false
left_polar
right_polar
```

These functions can be implemented by first computing a singular value decomposition, and then constructing the polar decomposition from the singular values and vectors. Alternatively, the polar decomposition can be computed using an 
iterative method based on Newton's method, that can be more efficient for large matrices, especially if they are
close to being isometric already.

```@docs; canonical=false
PolarViaSVD
PolarNewton
```

## Orthogonal Subspaces

Often it is useful to compute orthogonal bases for particular subspaces defined by a matrix.
Given a matrix `A`, we can compute an orthonormal basis for its image or coimage, and factorize the matrix accordingly.
These bases are accessible through [`left_orth`](@ref) and [`right_orth`](@ref) respectively.

### Overview

The [`left_orth`](@ref) function computes an orthonormal basis `V` for the image (column space) of `A`, along with a corestriction matrix `C` such that `A = V * C`.
The resulting `V` has orthonormal columns (`V' * V ≈ I` or `isisometric(V)`).

Similarly, [`right_orth`](@ref) computes an orthonormal basis for the coimage (row space) of `A`, i.e., the image of `A'`.
It returns matrices `C` and `Vᴴ` such that `A = C * Vᴴ`, where `V = (Vᴴ)'` has orthonormal columns (`isisometric(Vᴴ; side = :right)`).

These functions serve as high-level interfaces that automatically select the most appropriate decomposition based on the specified options, making them convenient for users who want orthonormalization without worrying about the underlying implementation details.

```@docs; canonical=false
left_orth
right_orth
```

### Algorithm Selection

Both functions support multiple decomposition drivers, which can be selected through the `alg` keyword argument:

**For `left_orth`:**
- `alg = :qr` (default without truncation): Uses QR decomposition via [`qr_compact`](@ref)
- `alg = :polar`: Uses polar decomposition via [`left_polar`](@ref)
- `alg = :svd` (default with truncation): Uses SVD via [`svd_compact`](@ref) or [`svd_trunc`](@ref)

**For `right_orth`:**
- `alg = :lq` (default without truncation): Uses LQ decomposition via [`lq_compact`](@ref)
- `alg = :polar`: Uses polar decomposition via [`right_polar`](@ref)
- `alg = :svd` (default with truncation): Uses SVD via [`svd_compact`](@ref) or [`svd_trunc`](@ref)

When `alg` is not specified, the function automatically selects `:qr`/`:lq` for exact orthogonalization, or `:svd` when a truncation strategy is provided.

### Extending with Custom Algorithms

To register a custom algorithm type for use with these functions, you need to define the appropriate conversion function, for example:

```julia
# For left_orth
MatrixAlgebraKit.left_orth_alg(alg::MyCustomAlgorithm) = LeftOrthAlgorithm{:qr}(alg)

# For right_orth
MatrixAlgebraKit.right_orth_alg(alg::MyCustomAlgorithm) = RightOrthAlgorithm{:lq}(alg)
```

The type parameter (`:qr`, `:lq`, `:polar`, or `:svd`) indicates which factorization backend will be used.
The wrapper algorithm types handle the dispatch to the appropriate implementation:

```@docs; canonical=false
left_orth_alg
right_orth_alg
LeftOrthAlgorithm
RightOrthAlgorithm
```

### Examples

Basic orthogonalization:

```jldoctest orthnull; output=false
using MatrixAlgebraKit
using LinearAlgebra

A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
V, C = left_orth(A)
(V' * V) ≈ I && A ≈ V * C

# output
true
```

Using different algorithms:

```jldoctest orthnull; output=false
A = randn(4, 3)
V1, C1 = left_orth(A; alg = :qr)
V2, C2 = left_orth(A; alg = :polar)
V3, C3 = left_orth(A; alg = :svd)
A ≈ V1 * C1 ≈ V2 * C2 ≈ V3 * C3

# output
true
```

With truncation:

```jldoctest orthnull; output=false
A = [1.0 0.0; 0.0 1e-10; 0.0 0.0]
V, C = left_orth(A; trunc = (atol = 1e-8,))
size(V, 2) == 1  # Only one column retained

# output
true
```


## Null Spaces

Similarly, it can be convenient to obtain an orthogonal basis for the kernel or cokernel of a matrix.
These are the orthogonal complements of the coimage and image, respectively, and can be computed using the [`left_null`](@ref) and [`right_null`](@ref) functions.

### Overview

The [`left_null`](@ref) function computes an orthonormal basis `N` for the cokernel (left nullspace) of `A`, which is the nullspace of `A'`.
This means `A' * N ≈ 0` and `N' * N ≈ I`.

Similarly, [`right_null`](@ref) computes an orthonormal basis for the kernel (right nullspace) of `A`.
It returns `Nᴴ` such that `A * Nᴴ' ≈ 0` and `Nᴴ * Nᴴ' ≈ I`, where `N = (Nᴴ)'` has orthonormal columns.

These functions automatically handle rank determination and provide convenient access to nullspace computation without requiring detailed knowledge of the underlying decomposition methods.

```@docs; canonical=false
left_null
right_null
```

### Algorithm Selection

Both functions support multiple decomposition drivers, which can be selected through the `alg` keyword argument:

**For `left_null`:**
- `alg = :qr` (default without truncation): Uses QR-based nullspace computation via [`qr_null`](@ref)
- `alg = :svd` (default with truncation): Uses SVD via [`svd_full`](@ref) with appropriate truncation

**For `right_null`:**
- `alg = :lq` (default without truncation): Uses LQ-based nullspace computation via [`lq_null`](@ref)
- `alg = :svd` (default with truncation): Uses SVD via [`svd_full`](@ref) with appropriate truncation

When `alg` is not specified, the function automatically selects `:qr`/`:lq` for exact nullspace computation, or `:svd` when a truncation strategy is provided to handle numerical rank determination.

!!! note
    For nullspace functions, [`notrunc`](@ref) has special meaning when used with the default QR/LQ algorithms.
    It indicates that the nullspace should be computed from the exact zeros determined by the additional rows/columns of the extended matrix, without any tolerance-based truncation.

### Extending with Custom Algorithms

To register a custom algorithm type for use with these functions, you need to define the appropriate conversion function:

```julia
# For left_null
MatrixAlgebraKit.left_null_alg(alg::MyCustomAlgorithm) = LeftNullAlgorithm{:qr}(alg)

# For right_null
MatrixAlgebraKit.right_null_alg(alg::MyCustomAlgorithm) = RightNullAlgorithm{:lq}(alg)
```

The type parameter (`:qr`, `:lq`, or `:svd`) indicates which factorization backend will be used.
The wrapper algorithm types handle the dispatch to the appropriate implementation:

```@docs; canonical=false
LeftNullAlgorithm
RightNullAlgorithm
left_null_alg
right_null_alg
```

### Examples

Basic nullspace computation:

```jldoctest orthnull; output=false
A = [1.0 2.0 3.0; 4.0 5.0 6.0]  # Rank 2 matrix
N = left_null(A)
size(N) == (2, 0)

# output
true
```

```jldoctest orthnull; output=false
Nᴴ = right_null(A)
size(Nᴴ) == (1, 3) && norm(A * Nᴴ') < 1e-14 && isisometric(Nᴴ; side = :right)

# output
true
```

Computing nullspace with rank detection:

```jldoctest orthnull; output=false
A = [1.0 2.0; 2.0 4.0; 3.0 6.0]  # Rank 1 matrix (second column = 2*first)
N = left_null(A; alg = :svd, trunc = (atol = 1e-10,))
size(N) == (3, 2) && norm(A' * N) < 1e-12 && isisometric(N)

# output
true
```

Using different algorithms:

```jldoctest orthnull; output=false
A = [1.0 0.0 0.0; 0.0 1.0 0.0]
N1 = right_null(A; alg = :lq)
N2 = right_null(A; alg = :svd)
norm(A * N1') < 1e-14 && norm(A * N2') < 1e-14 &&
    isisometric(N1; side = :right) && isisometric(N2; side = :right)

# output
true
```

## [Gauge choices](@id sec_gaugefix)

Both eigenvalue and singular value decompositions have residual gauge degrees of freedom even when the eigenvalues or singular values are unique.
These arise from the fact that even after normalization, the eigenvectors and singular vectors are only determined up to a phase factor.

### Phase Ambiguity in Decompositions

For the eigenvalue decomposition `A * V = V * D`, if `v` is an eigenvector with eigenvalue `λ` and `|v| = 1`, then so is `e^(iθ) * v` for any real phase `θ`.
When `λ` is non-degenerate (i.e., has multiplicity 1), the eigenvector is unique up to this phase.

Similarly, for the singular value decomposition `A = U * Σ * Vᴴ`, the singular vectors `u` and `v` corresponding to a non-degenerate singular value `σ` are unique only up to a common phase.
We can replace `u → e^(iθ) * u` and `vᴴ → e^(-iθ) * vᴴ` simultaneously.

### Gauge Fixing Convention

To remove this phase ambiguity and ensure reproducible results, MatrixAlgebraKit implements a gauge fixing convention by default.
The convention ensures that **the entry with the largest magnitude in each eigenvector or left singular vector is real and positive**.

For eigenvectors, this means that for each column `v` of `V`, we multiply by `conj(sign(v[i]))` where `i` is the index of the entry with largest absolute value.

For singular vectors, we apply the phase factor to both `u` and `v` based on the entry with largest magnitude in `u`.
This preserves the decomposition `A = U * Σ * Vᴴ` while fixing the gauge.

### Disabling Gauge Fixing

Gauge fixing is enabled by default for all eigenvalue and singular value decompositions.
If you prefer to obtain the raw results from the underlying LAPACK routines without gauge fixing, you can disable it using the `gaugefix` keyword argument:

```julia
# With gauge fixing (default)
D, V = eigh_full(A)

# Without gauge fixing
D, V = eigh_full(A; gaugefix = false)
```

The same keyword is available for `eig_full`, `eig_trunc`, `svd_full`, `svd_compact`, and `svd_trunc` functions.

```@docs; canonical=false
MatrixAlgebraKit.gaugefix!
```


