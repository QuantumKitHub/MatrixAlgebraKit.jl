module MatrixAlgebraKitStridedViewsExt

using MatrixAlgebraKit, StridedViews
import MatrixAlgebraKit: default_svd_algorithm, default_eig_algorithm, default_eigh_algorithm, default_driver

MatrixAlgebraKit.default_svd_algorithm(::Type{<:StridedViews.StridedView{T, 2, A}}; kwargs...) where {T <: Number, A <: AbstractArray{T}} = MatrixAlgebraKit.default_svd_algorithm(A; kwargs...)
MatrixAlgebraKit.default_eig_algorithm(::Type{<:StridedViews.StridedView{T, 2, A}}; kwargs...) where {T <: Number, A <: AbstractArray{T}} = MatrixAlgebraKit.default_eig_algorithm(A; kwargs...)
MatrixAlgebraKit.default_eigh_algorithm(::Type{<:StridedViews.StridedView{T, 2, A}}; kwargs...) where {T <: Number, A <: AbstractArray{T}} = MatrixAlgebraKit.default_eigh_algorithm(A; kwargs...)
MatrixAlgebraKit.default_driver(::Type{<:StridedViews.StridedView{T, 2, A}}) where {T <: Number, A <: AbstractArray{T}} = MatrixAlgebraKit.default_driver(A)

end
