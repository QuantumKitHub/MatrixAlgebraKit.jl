module MatrixAlgebraKitGenericSchurExt

using MatrixAlgebraKit
using MatrixAlgebraKit: check_input, GS
import MatrixAlgebraKit: geev!
using LinearAlgebra: Diagonal, sorteig!
using GenericSchur

const GSFloat = Union{Float16, ComplexF16, BigFloat, Complex{BigFloat}}

function MatrixAlgebraKit.default_eig_algorithm(::Type{T}; kwargs...) where {T <: StridedMatrix{<:GSFloat}}
    return Simple(; kwargs...)
end

MatrixAlgebraKit.default_driver(::Type{<:Simple}, ::Type{TA}) where {TA <: StridedMatrix{<:GSFloat}} = GS()

function geev!(::GS, A::AbstractMatrix, Dd::AbstractVector, V::AbstractMatrix; kwargs...)
    D, Vmat = GenericSchur.eigen!(A)
    copyto!(Dd, D)
    length(V) > 0 && copyto!(V, Vmat)
    return Dd, V
end

Base.@deprecate(
    MatrixAlgebraKit.eig_full!(A, DV, alg::GS_QRIteration),
    MatrixAlgebraKit.eig_full!(A, DV, Simple(; driver = GS(), alg.kwargs...))
)
Base.@deprecate(
    MatrixAlgebraKit.eig_vals!(A, D, alg::GS_QRIteration),
    MatrixAlgebraKit.eig_vals!(A, D, Simple(; driver = GS(), alg.kwargs...))
)

function MatrixAlgebraKit.schur_full!(A::AbstractMatrix, TZv, alg::GS_QRIteration)
    check_input(schur_full!, A, TZv, alg)
    T, Z, vals = TZv
    S = GenericSchur.gschur(A)
    copyto!(T, S.T)
    copyto!(Z, S.Z)
    copyto!(vals, S.values)
    return T, Z, vals
end

function MatrixAlgebraKit.schur_vals!(A::AbstractMatrix, vals, alg::GS_QRIteration)
    check_input(schur_vals!, A, vals, alg)
    S = GenericSchur.gschur(A)
    copyto!(vals, sorteig!(S.values))
    return vals
end

end
