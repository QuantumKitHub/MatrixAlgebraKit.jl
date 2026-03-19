module MatrixAlgebraKitGenericSchurExt

using MatrixAlgebraKit
using MatrixAlgebraKit: check_input, GS, Driver
import MatrixAlgebraKit: geev!, geevx!, gees!, eig_full!, eig_vals!, schur_full!, schur_vals!
using LinearAlgebra: Diagonal, sorteig!
using GenericSchur

const GSFloat = Union{Float16, ComplexF16, BigFloat, Complex{BigFloat}}

function MatrixAlgebraKit.default_eig_algorithm(
        ::Type{T};
        balanced::Bool = false, driver::Driver = GS(), kwargs...
    ) where {T <: StridedMatrix{<:GSFloat}}
    return QRIteration(; driver, balanced, kwargs...)
end

function geev!(::GS, A::AbstractMatrix, Dd::AbstractVector, V::AbstractMatrix; kwargs...)
    D, Vmat = GenericSchur.eigen!(A)
    copyto!(Dd, D)
    length(V) > 0 && copyto!(V, Vmat)
    return Dd, V
end

function gees!(driver::GS, A::AbstractMatrix, Z::AbstractMatrix, vals::AbstractVector)
    S = GenericSchur.gschur(A)
    copyto!(A, S.T)
    length(Z) > 0 && copyto!(Z, S.Z)
    copyto!(vals, sorteig!(S.values))
    return A, Z, vals
end

Base.@deprecate(
    eig_full!(A, DV, alg::GS_QRIteration),
    eig_full!(A, DV, QRIteration(; driver = GS(), alg.kwargs...))
)
Base.@deprecate(
    eig_vals!(A, D, alg::GS_QRIteration),
    eig_vals!(A, D, QRIteration(; driver = GS(), alg.kwargs...))
)

Base.@deprecate(
    schur_full!(A, TZv, alg::GS_QRIteration),
    schur_full!(A, TZv, QRIteration(; driver = GS(), alg.kwargs...))
)
Base.@deprecate(
    schur_vals!(A, vals, alg::GS_QRIteration),
    schur_vals!(A, vals, QRIteration(; driver = GS(), alg.kwargs...))
)

end
