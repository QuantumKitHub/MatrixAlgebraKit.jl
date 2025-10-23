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

Often it is useful to compute orthogonal bases for a particular subspace defined by a matrix.
Given a matrix `A` we can compute an orthonormal basis for its image or coimage, and factorize the matrix accordingly.
These bases are accessible through [`left_orth`](@ref) and [`right_orth`](@ref) respectively.
This is implemented through a combination of the decompositions mentioned above, and serves as a convenient interface to these operations.

```@docs; canonical=false
left_orth
right_orth
```

In order to dispatch to the underlying factorizations, the following wrapper algorithms are used:

```@docs; canonical=false
LeftOrthAlgorithm
RightOrthAlgorithm
```


## Null Spaces

Similarly, it can be convenient to obtain an orthogonal basis for the kernel or cokernel of a matrix.
These are the compliments of the coimage and image, respectively, and can be computed using the [`left_null`](@ref) and [`right_null`](@ref) functions.
Again, this is typically implemented through a combination of the decompositions mentioned above, and serves as a convenient interface to these operations.

```@docs; canonical=false
left_null
right_null
```

Again, dispatching happens through the following wrapper algorithm types:

```@docs; canonical=false
LeftNullAlgorithm
RightNullAlgorithm
```
