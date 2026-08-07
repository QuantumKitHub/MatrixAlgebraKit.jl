module MatrixAlgebraKitStridedViewsExt

using MatrixAlgebraKit, StridedViews, LinearAlgebra
import MatrixAlgebraKit: default_svd_algorithm, default_eig_algorithm, default_eigh_algorithm, default_driver

MatrixAlgebraKit.default_svd_algorithm(::Type{<:StridedViews.StridedView{T, 2, A}}; kwargs...) where {T <: Number, A <: AbstractArray{T}} = MatrixAlgebraKit.default_svd_algorithm(A; kwargs...)
MatrixAlgebraKit.default_eig_algorithm(::Type{<:StridedViews.StridedView{T, 2, A}}; kwargs...) where {T <: Number, A <: AbstractArray{T}} = MatrixAlgebraKit.default_eig_algorithm(A; kwargs...)
MatrixAlgebraKit.default_eigh_algorithm(::Type{<:StridedViews.StridedView{T, 2, A}}; kwargs...) where {T <: Number, A <: AbstractArray{T}} = MatrixAlgebraKit.default_eigh_algorithm(A; kwargs...)
MatrixAlgebraKit.default_driver(::Type{<:StridedViews.StridedView{T, 2, A}}) where {T <: Number, A <: AbstractArray{T}} = MatrixAlgebraKit.default_driver(A)

LinearAlgebra.exp!(S::StridedViews.StridedView{T, 2, A}) where {T <: Number, A <: AbstractArray{T}} = MatrixAlgebraKit.exponential!(S, MatrixAlgebraKit.MatrixFunctionViaEig(default_eig_algorithm(S)))
function LinearAlgebra.svd!(S::StridedViews.StridedView{T, 2, A}; full::Bool = false, alg) where {T <: Number, A <: AbstractArray{T}}
    if full
        U, s, Vᴴ = svd_full!(S)
        return LinearAlgebra.SVD(U, s, Vᴴ)
    else
        U, s, Vᴴ = svd_compact!(S)
        return LinearAlgebra.SVD(U, MatrixAlgebraKit.diagview(s), Vᴴ)
    end
end
LinearAlgebra.schur!(S::StridedViews.StridedView{T, 2, A}) where {T <: Number, A <: AbstractArray{T}} = LinearAlgebra.Schur(MatrixAlgebraKit.schur_full!(S)...)
function LinearAlgebra.eigen!(hS::Hermitian{T, <:StridedViews.StridedView{T, 2, A}}; kwargs...) where {T <: Number, A <: AbstractArray{T}}
    D, V = MatrixAlgebraKit.eigh_full!(parent(hS))
    return LinearAlgebra.Eigen(MatrixAlgebraKit.diagview(D), V)
end

end
