module MatrixAlgebraKitGenericSchurExt

using MatrixAlgebraKit
using MatrixAlgebraKit: check_input
using LinearAlgebra: Diagonal
using GenericSchur

function MatrixAlgebraKit.default_eig_algorithm(::Type{T}; kwargs...) where {T <: StridedMatrix{<:Union{BigFloat, Complex{BigFloat}}}}
    return GS_QRIteration(; kwargs...)
end

for f! in (:eig_full!, :eig_vals!)
    @eval MatrixAlgebraKit.initialize_output(::typeof($f!), A::AbstractMatrix, ::GS_QRIteration) = nothing
end

function MatrixAlgebraKit.eig_full!(A::AbstractMatrix, DV, ::GS_QRIteration)
    D, V = GenericSchur.eigen!(A)
    return Diagonal(D), V
end

function MatrixAlgebraKit.eig_vals!(A::AbstractMatrix, D, ::GS_QRIteration)
    return GenericSchur.eigvals!(A)
end

end
