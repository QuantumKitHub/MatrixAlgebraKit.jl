# ================================
# MATRIX FUNCTION ALGORITHMS
# ================================
"""
    MatrixFunctionViaLA()

Algorithm type to denote computing a function of a matrix `A` via the implementation of `LinearAlgebra`.
In order to retain type stability, complex results for real inputs are rejected with a `DomainError`.
Use [`MatrixFunctionViaEig`](@ref) or [`MatrixFunctionViaEigh`](@ref) to check the spectrum itself against a tolerance.
See also [Domain considerations](@ref sec_matrixfunction_domain).
"""
@algdef MatrixFunctionViaLA

"""
    MatrixFunctionViaTaylor(; tol=eps, balance=true, estimate_order=4)

Algorithm type to denote finding the exponential of `A` through a pure-Julia scaling-and-squaring
evaluation of its Taylor series, following Fasi & Higham (2018).
The truncation order and the number of squarings are chosen to reach a relative accuracy `tol`,
and the Taylor polynomial is evaluated with the Paterson–Stockmeyer scheme.
When `balance` is `true`, `A` is first balanced by a diagonal similarity.
`estimate_order` sets how many powers of `A` are formed up front to sharpen the norm estimate via the
Al-Mohy–Higham quantities `‖Aᵖ‖^(1/p)` (Al-Mohy & Higham, 2009); these powers are reused by the
Paterson–Stockmeyer evaluation.
As this algorithm requires no LAPACK support, it also applies at arbitrary precision.

## References

- A. H. Al-Mohy and N. J. Higham, "A New Scaling and Squaring Algorithm for the Matrix
  Exponential", SIAM J. Matrix Anal. Appl., 31(3), 970–989, 2009.
"""
@algdef MatrixFunctionViaTaylor

"""
    MatrixFunctionViaEigh(eigh_alg; domain_atol = default_domain_atol(λ))
    MatrixFunctionViaEigh(; eigh_alg, domain_atol = default_domain_atol(λ))

Algorithm type for computing a function of a matrix by computing its hermitian eigenvalue decomposition and applying the function to the eigenvalues.
The optional `eigh_alg` specifies which hermitian eigendecomposition implementation to use, either
positionally or as a keyword, and defaults to the one selected for the input.
`domain_atol` applies to [`squareroot`](@ref): it is the absolute tolerance within which negative
eigenvalues are treated as rounding artifacts and clamped onto zero, and defaults to
[`default_domain_atol`](@ref). Raising it accepts more matrices;
see [Domain considerations](@ref sec_matrixfunction_domain).
"""
@algdef MatrixFunctionViaEigh

@deprecate(
    MatrixFunctionViaEigh(eigh_alg::AbstractAlgorithm; kwargs...),
    MatrixFunctionViaEigh(; eigh_alg, kwargs...)
)

"""
    MatrixFunctionViaEig(eig_alg; domain_atol = default_domain_atol(λ))
    MatrixFunctionViaEig(; eig_alg, domain_atol = default_domain_atol(λ))

Algorithm type for computing a function of a matrix by computing its eigenvalue decomposition and applying the function to the eigenvalues.
The optional `eig_alg` specifies which eigendecomposition implementation to use, either positionally
or as a keyword, and defaults to the one selected for the input.
`domain_atol` applies to [`squareroot`](@ref): it is the absolute tolerance within which eigenvalues
on the negative real axis are treated as rounding artifacts and clamped onto zero, and defaults to
[`default_domain_atol`](@ref). Raising it accepts more matrices;
see [Domain considerations](@ref sec_matrixfunction_domain).

!!! warning
    This algorithm presumes a well-conditioned eigenbasis. For a defective or nearly defective
    matrix both its result and its domain verdict are unreliable, since the eigenvalues themselves
    are resolved only to `eps^(1/k)` for a Jordan block of size `k`. Prefer
    [`MatrixFunctionViaSchur`](@ref), which is Schur-based, for such matrices.
"""
@algdef MatrixFunctionViaEig

@deprecate(
    MatrixFunctionViaEig(eig_alg::AbstractAlgorithm; kwargs...),
    MatrixFunctionViaEig(; eig_alg, kwargs...)
)

"""
    MatrixFunctionViaSchur(; schur_alg, blocksize = 0, domain_atol = default_domain_atol(λ))

Algorithm type for computing a function of a matrix from its Schur decomposition, by evaluating the
function on the (quasi-)triangular Schur factor and transforming back.
It applies to [`squareroot`](@ref) only, where the triangular factor is obtained through the
recursion of Björck & Hammarling (1983), in the real quasi-triangular variant of Higham (1987) so
that a real input is treated in real arithmetic, and with the recursive blocking of Deadman, Higham
& Ralha (2013) to move the bulk of the work into matrix multiplications.
As this algorithm requires no LAPACK support beyond the Schur decomposition itself, it also applies
at arbitrary precision; it is however not suitable for GPU arrays, as the recursion indexes
individual entries.

The optional `schur_alg` specifies which Schur decomposition implementation to use, and defaults to
the one selected for the input.
`blocksize` is the size at which the recursion switches over to the entrywise algorithm, where `1`
recurses as far as it can and any value at or above the matrix size skips the recursion entirely.
It defaults to `0`, which selects a small threshold for the scalar types that have a level-3 BLAS
to spend the work in, and a large one for those that do not.
`domain_atol` is the absolute tolerance within which negative eigenvalues are treated as rounding
artifacts and clamped onto zero, and defaults to [`default_domain_atol`](@ref). Raising it accepts
more matrices; see [Domain considerations](@ref sec_matrixfunction_domain).

Unlike [`MatrixFunctionViaEig`](@ref), this algorithm is backward stable and does not rely on the
conditioning of the eigenbasis, so it is the appropriate choice for a defective or nearly defective
matrix.

## References

- Å. Björck and S. Hammarling, "A Schur method for the square root of a matrix",
  Linear Algebra Appl., 52/53, 127–140, 1983.
- N. J. Higham, "Computing real square roots of a real matrix", Linear Algebra Appl., 88/89,
  405–430, 1987.
- E. Deadman, N. J. Higham and R. Ralha, "Blocked Schur Algorithms for Computing the Matrix Square
  Root", Applied Parallel and Scientific Computing, Lecture Notes in Computer Science 7782,
  171–182, 2013.
"""
@algdef MatrixFunctionViaSchur
