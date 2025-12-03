# ================================
# EXPONENTIAL ALGORITHMS
# ================================
"""
    MatrixFunctionViaLA()

Algorithm type to denote finding the exponential of `A` via the implementation of `LinearAlgebra`.
"""
@algdef MatrixFunctionViaLA

"""
    MatrixFunctionViaEigh()

Algorithm type to denote finding the exponential `A` by computing the hermitian eigendecomposition of `A`.
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
    MatrixFunctionViaEig()

Algorithm type to denote finding the exponential `A` by computing the eigendecomposition of `A`.
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
