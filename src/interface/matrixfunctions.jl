# ================================
# EXPONENTIAL ALGORITHMS
# ================================
"""
    MatrixFunctionViaLA()

Algorithm type to denote finding the exponential of `A` via the implementation of `LinearAlgebra`.
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
    MatrixFunctionViaEigh(eigh_alg)

Algorithm type for computing a function of a matrix by computing its hermitian eigenvalue decomposition and applying the function to the eigenvalues.
The `eigh_alg` specifies which hermitian eigendecomposition implementation to use.
"""
struct MatrixFunctionViaEigh{A <: AbstractAlgorithm} <: AbstractAlgorithm
    eigh_alg::A
end
function Base.show(io::IO, alg::MatrixFunctionViaEigh)
    print(io, "MatrixFunctionViaEigh(")
    _show_alg(io, alg.eigh_alg)
    return print(io, ")")
end

"""
    MatrixFunctionViaEig(eig_alg)

Algorithm type for computing a function of a matrix by computing its eigenvalue decomposition and applying the function to the eigenvalues.
The `eig_alg` specifies which eigendecomposition implementation to use.
"""
struct MatrixFunctionViaEig{A <: AbstractAlgorithm} <: AbstractAlgorithm
    eig_alg::A
end
function Base.show(io::IO, alg::MatrixFunctionViaEig)
    print(io, "MatrixFunctionViaEig(")
    _show_alg(io, alg.eig_alg)
    return print(io, ")")
end
