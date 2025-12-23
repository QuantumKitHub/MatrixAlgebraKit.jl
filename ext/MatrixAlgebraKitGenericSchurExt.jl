module MatrixAlgebraKitGenericSchurExt

using MatrixAlgebraKit
using MatrixAlgebraKit: check_input
using LinearAlgebra: Diagonal, sorteig!
using GenericSchur

function MatrixAlgebraKit.default_eig_algorithm(::Type{T}; kwargs...) where {T <: StridedMatrix{<:Union{Float16, ComplexF16, BigFloat, Complex{BigFloat}}}}
    return GS_QRIteration(; kwargs...)
end

MatrixAlgebraKit.initialize_output(::typeof(eig_full!), A::AbstractMatrix, ::GS_QRIteration) = (nothing, nothing)
MatrixAlgebraKit.initialize_output(::typeof(eig_vals!), A::AbstractMatrix, ::GS_QRIteration) = nothing

function MatrixAlgebraKit.eig_full!(A::AbstractMatrix, DV, ::GS_QRIteration)
    D_, V_ = GenericSchur.eigen!(A)
    D, V = DV
    if !isnothing(D)
        copyto!(D, Diagonal(D_))
    else
        D = D_
    end
    if !isnothing(V)
        copyto!(V, V_)
    else
        V = V_
    end
    return Diagonal(D), V
end

function MatrixAlgebraKit.eig_vals!(A::AbstractMatrix, D, ::GS_QRIteration)
    return GenericSchur.eigvals!(A)
end

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
