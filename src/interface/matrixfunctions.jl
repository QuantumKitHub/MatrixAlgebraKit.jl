# ================================
# EXPONENTIAL ALGORITHMS
# ================================
"""
    MatrixFunctionViaLA()

Algorithm type to denote finding the exponential of `A` via the implementation of `LinearAlgebra`.
"""
@algdef MatrixFunctionViaLA

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
