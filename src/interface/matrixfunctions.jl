# ================================
# MATRIX FUNCTION ALGORITHMS
# ================================
"""
    MatrixFunctionViaLA()

Algorithm type to denote computing a function of a matrix `A` via the implementation of `LinearAlgebra`.
"""
@algdef MatrixFunctionViaLA

"""
    MatrixFunctionViaEigh(eigh_alg; domain_atol=nothing)

Algorithm type for computing a function of a matrix by computing its hermitian eigenvalue decomposition and applying the function to the eigenvalues.
The `eigh_alg` specifies which hermitian eigendecomposition implementation to use.
For matrix functions with a restricted domain (e.g. [`squareroot`](@ref) and [`logarithm`](@ref)),
`domain_atol` specifies the absolute tolerance below which out-of-domain eigenvalues are treated
as rounding artifacts and clamped to the domain boundary, with `nothing` denoting the default
tolerance [`default_domain_atol`](@ref).
"""
struct MatrixFunctionViaEigh{A <: AbstractAlgorithm, T <: Union{Nothing, Real}} <: AbstractAlgorithm
    eigh_alg::A
    domain_atol::T
end
function MatrixFunctionViaEigh(eigh_alg::AbstractAlgorithm; domain_atol::Union{Nothing, Real} = nothing)
    return MatrixFunctionViaEigh(eigh_alg, domain_atol)
end
function Base.show(io::IO, alg::MatrixFunctionViaEigh)
    print(io, "MatrixFunctionViaEigh(")
    _show_alg(io, alg.eigh_alg)
    isnothing(alg.domain_atol) || print(io, "; domain_atol=", alg.domain_atol)
    return print(io, ")")
end

"""
    MatrixFunctionViaEig(eig_alg; domain_atol=nothing)

Algorithm type for computing a function of a matrix by computing its eigenvalue decomposition and applying the function to the eigenvalues.
The `eig_alg` specifies which eigendecomposition implementation to use.
For matrix functions with a restricted domain (e.g. [`squareroot`](@ref) and [`logarithm`](@ref)),
`domain_atol` specifies the absolute tolerance below which out-of-domain eigenvalues are treated
as rounding artifacts and clamped to the domain boundary, with `nothing` denoting the default
tolerance [`default_domain_atol`](@ref).
"""
struct MatrixFunctionViaEig{A <: AbstractAlgorithm, T <: Union{Nothing, Real}} <: AbstractAlgorithm
    eig_alg::A
    domain_atol::T
end
function MatrixFunctionViaEig(eig_alg::AbstractAlgorithm; domain_atol::Union{Nothing, Real} = nothing)
    return MatrixFunctionViaEig(eig_alg, domain_atol)
end
function Base.show(io::IO, alg::MatrixFunctionViaEig)
    print(io, "MatrixFunctionViaEig(")
    _show_alg(io, alg.eig_alg)
    isnothing(alg.domain_atol) || print(io, "; domain_atol=", alg.domain_atol)
    return print(io, ")")
end
