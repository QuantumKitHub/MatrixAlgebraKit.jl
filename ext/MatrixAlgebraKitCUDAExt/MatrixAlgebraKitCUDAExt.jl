module MatrixAlgebraKitCUDAExt

using MatrixAlgebraKit
using MatrixAlgebraKit: @algdef, Algorithm, check_input
using MatrixAlgebraKit: one!, zero!, uppertriangular!, lowertriangular!
using MatrixAlgebraKit: diagview, sign_safe
using MatrixAlgebraKit: LQViaTransposedQR
using MatrixAlgebraKit: default_qr_algorithm, default_lq_algorithm, default_svd_algorithm, default_eig_algorithm, default_eigh_algorithm
import MatrixAlgebraKit: _gpu_geqrf!, _gpu_ungqr!, _gpu_unmqr!, _gpu_gesvd!, _gpu_Xgesvdp!, _gpu_Xgesvdr!, _gpu_gesvdj!, _gpu_geev!
import MatrixAlgebraKit: _gpu_heevj!, _gpu_heevd!
using CUDA
using LinearAlgebra
using LinearAlgebra: BlasFloat

include("yacusolver.jl")

function MatrixAlgebraKit.default_qr_algorithm(::Type{T}; kwargs...) where {T<:StridedCuMatrix}
    return CUSOLVER_HouseholderQR(; kwargs...)
end
function MatrixAlgebraKit.default_lq_algorithm(::Type{T}; kwargs...) where {T<:StridedCuMatrix}
    qr_alg = CUSOLVER_HouseholderQR(; kwargs...)
    return LQViaTransposedQR(qr_alg)
end
function MatrixAlgebraKit.default_svd_algorithm(::Type{T}; kwargs...) where {T<:StridedCuMatrix}
    return CUSOLVER_QRIteration(; kwargs...)
end
function MatrixAlgebraKit.default_eig_algorithm(::Type{T}; kwargs...) where {T<:StridedCuMatrix}
    return CUSOLVER_Simple(; kwargs...)
end
function MatrixAlgebraKit.default_eigh_algorithm(::Type{T}; kwargs...) where {T<:StridedCuMatrix}
    return CUSOLVER_DivideAndConquer(; kwargs...)
end


_gpu_geev!(A::StridedCuMatrix, D::StridedCuVector, V::StridedCuMatrix) = YACUSOLVER.Xgeev!(A, D, V)
_gpu_geqrf!(A::StridedCuMatrix) = YACUSOLVER.geqrf!(A)
_gpu_ungqr!(A::StridedCuMatrix, τ::StridedCuVector) = YACUSOLVER.ungqr!(A, τ)
_gpu_unmqr!(side::AbstractChar, trans::AbstractChar, A::StridedCuMatrix, τ::StridedCuVector, C::StridedCuVecOrMat) = YACUSOLVER.unmqr!(side, trans, A, τ, C)
_gpu_gesvd!(A::StridedCuMatrix, S::StridedCuVector, U::StridedCuMatrix, Vᴴ::StridedCuMatrix) = YACUSOLVER.gesvd!(A, S, U, Vᴴ) 
_gpu_Xgesvdp!(A::StridedCuMatrix, S::StridedCuVector, U::StridedCuMatrix, Vᴴ::StridedCuMatrix; kwargs...) = YACUSOLVER.Xgesvdp!(A, S, U, Vᴴ; kwargs...) 
_gpu_Xgesvdr!(A::StridedCuMatrix, S::StridedCuVector, U::StridedCuMatrix, Vᴴ::StridedCuMatrix; kwargs...) = YACUSOLVER.Xgesvdr!(A, S, U, Vᴴ; kwargs...) 
_gpu_gesvdj!(A::StridedCuMatrix, S::StridedCuVector, U::StridedCuMatrix, Vᴴ::StridedCuMatrix; kwargs...) = YACUSOLVER.gesvdj!(A, S, U, Vᴴ; kwargs...)

_gpu_heevj!(A::StridedCuMatrix, Dd::StridedCuVector, V::StridedCuMatrix; kwargs...) = YACUSOLVER.heevj!(A, Dd, V; kwargs...)
_gpu_heevd!(A::StridedCuMatrix, Dd::StridedCuVector, V::StridedCuMatrix; kwargs...) = YACUSOLVER.heevd!(A, Dd, V; kwargs...)

end
