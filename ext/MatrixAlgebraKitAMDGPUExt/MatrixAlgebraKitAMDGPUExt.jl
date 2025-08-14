module MatrixAlgebraKitAMDGPUExt

using MatrixAlgebraKit
using MatrixAlgebraKit: @algdef, Algorithm, check_input
using MatrixAlgebraKit: one!, zero!, uppertriangular!, lowertriangular!
using MatrixAlgebraKit: diagview, sign_safe
using MatrixAlgebraKit: LQViaTransposedQR
using MatrixAlgebraKit: default_qr_algorithm, default_lq_algorithm, default_svd_algorithm
import MatrixAlgebraKit: _gpu_geqrf!, _gpu_ungqr!, _gpu_unmqr!, _gpu_gesvd!, _gpu_Xgesvdp!, _gpu_gesvdj!  
using AMDGPU
using LinearAlgebra
using LinearAlgebra: BlasFloat

include("yarocsolver.jl")

function MatrixAlgebraKit.default_qr_algorithm(::Type{T}; kwargs...) where {T<:StridedROCMatrix}
    return ROCSOLVER_HouseholderQR(; kwargs...)
end
function MatrixAlgebraKit.default_lq_algorithm(::Type{T}; kwargs...) where {T<:StridedROCMatrix}
    qr_alg = ROCSOLVER_HouseholderQR(; kwargs...)
    return LQViaTransposedQR(qr_alg)
end
function MatrixAlgebraKit.default_svd_algorithm(::Type{T}; kwargs...) where {T<:StridedROCMatrix}
    return ROCSOLVER_QRIteration(; kwargs...)
end

_gpu_geqrf!(A::StridedROCMatrix) = YArocSOLVER.geqrf!(A)
_gpu_ungqr!(A::StridedROCMatrix, τ::StridedROCVector) = YArocSOLVER.ungqr!(A, τ)
_gpu_unmqr!(side::AbstractChar, trans::AbstractChar, A::StridedROCMatrix, τ::StridedROCVector, C::StridedROCVecOrMat) = YArocSOLVER.unmqr!(side, trans, A, τ, C)
_gpu_gesvd!(A::StridedROCMatrix, S::StridedROCVector, U::StridedROCMatrix, Vᴴ::StridedROCMatrix) = YArocSOLVER.gesvd!(A, S, U, Vᴴ) 
# not yet supported
#_gpu_Xgesvdp!(A::StridedROCMatrix, S::StridedROCVector, U::StridedROCMatrix, Vᴴ::StridedROCMatrix; kwargs...) = YArocSOLVER.Xgesvdp!(A, S, U, Vᴴ; kwargs...) 
_gpu_gesvdj!(A::StridedROCMatrix, S::StridedROCVector, U::StridedROCMatrix, Vᴴ::StridedROCMatrix; kwargs...) = YArocSOLVER.gesvdj!(A, S, U, Vᴴ; kwargs...)

end
