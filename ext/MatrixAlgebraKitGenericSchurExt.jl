module MatrixAlgebraKitGenericSchurExt

using MatrixAlgebraKit
using MatrixAlgebraKit: check_input, GS
import MatrixAlgebraKit: geev!, gees!, eig_full!, eig_vals!, schur_full!, schur_vals!
using LinearAlgebra: Diagonal, sorteig!
using GenericSchur

const GSFloat = Union{Float16, ComplexF16, BigFloat, Complex{BigFloat}}

function MatrixAlgebraKit.default_eig_algorithm(::Type{T}; kwargs...) where {T <: StridedMatrix{<:GSFloat}}
    return Simple(; kwargs...)
end

MatrixAlgebraKit.default_driver(::Type{<:Simple}, ::Type{TA}) where {TA <: StridedMatrix{<:GSFloat}} = GS()

MatrixAlgebraKit.supports_schur(::GS, f::Symbol) = f === :simple
MatrixAlgebraKit.supports_eig(::GS, f::Symbol) = f === :simple

function geev!(::GS, A::AbstractMatrix, Dd::AbstractVector, V::AbstractMatrix; kwargs...)
    D, Vmat = GenericSchur.eigen!(A)
    copyto!(Dd, D)
    length(V) > 0 && copyto!(V, Vmat)
    return Dd, V
end

function gees!(::GS, A::AbstractMatrix, Z::AbstractMatrix, vals::AbstractVector)
    S = GenericSchur.gschur(A)
    copyto!(A, S.T)
    if length(Z) > 0
        copyto!(Z, S.Z)
        copyto!(vals, S.values)
    else
        copyto!(vals, sorteig!(S.values))
    end
    return A, Z, vals
end

Base.@deprecate(
    eig_full!(A, DV, alg::GS_QRIteration),
    eig_full!(A, DV, Simple(; driver = GS(), alg.kwargs...))
)
Base.@deprecate(
    eig_vals!(A, D, alg::GS_QRIteration),
    eig_vals!(A, D, Simple(; driver = GS(), alg.kwargs...))
)

Base.@deprecate(
    schur_full!(A, TZv, alg::GS_QRIteration),
    schur_full!(A, TZv, Simple(; driver = GS(), alg.kwargs...))
)
Base.@deprecate(
    schur_vals!(A, vals, alg::GS_QRIteration),
    schur_vals!(A, vals, Simple(; driver = GS(), alg.kwargs...))
)

end
