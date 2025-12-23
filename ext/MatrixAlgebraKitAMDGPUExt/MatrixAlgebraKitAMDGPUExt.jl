module MatrixAlgebraKitAMDGPUExt

using MatrixAlgebraKit
using MatrixAlgebraKit: @algdef, Algorithm, check_input
using MatrixAlgebraKit: one!, zero!, uppertriangular!, lowertriangular!
using MatrixAlgebraKit: diagview, sign_safe
using MatrixAlgebraKit: LQViaTransposedQR, TruncationStrategy, NoTruncation, TruncationByValue, AbstractAlgorithm
using MatrixAlgebraKit: default_qr_algorithm, default_lq_algorithm, default_svd_algorithm, default_eigh_algorithm
import MatrixAlgebraKit: _gpu_geqrf!, _gpu_ungqr!, _gpu_unmqr!, _gpu_gesvd!, _gpu_Xgesvdp!, _gpu_gesvdj!
import MatrixAlgebraKit: _gpu_heevj!, _gpu_heevd!, _gpu_heev!, _gpu_heevx!
using AMDGPU
using LinearAlgebra
using LinearAlgebra: BlasFloat

include("yarocsolver.jl")

function MatrixAlgebraKit.default_qr_algorithm(::Type{T}; kwargs...) where {T <: StridedROCMatrix}
    return ROCSOLVER_HouseholderQR(; kwargs...)
end
function MatrixAlgebraKit.default_lq_algorithm(::Type{T}; kwargs...) where {T <: StridedROCMatrix}
    qr_alg = ROCSOLVER_HouseholderQR(; kwargs...)
    return LQViaTransposedQR(qr_alg)
end
function MatrixAlgebraKit.default_svd_algorithm(::Type{T}; kwargs...) where {T <: StridedROCMatrix}
    return ROCSOLVER_QRIteration(; kwargs...)
end
function MatrixAlgebraKit.default_eigh_algorithm(::Type{T}; kwargs...) where {T <: StridedROCMatrix}
    return ROCSOLVER_DivideAndConquer(; kwargs...)
end

_gpu_geqrf!(A::StridedROCMatrix) = YArocSOLVER.geqrf!(A)
_gpu_ungqr!(A::StridedROCMatrix, τ::StridedROCVector) = YArocSOLVER.ungqr!(A, τ)
_gpu_unmqr!(side::AbstractChar, trans::AbstractChar, A::StridedROCMatrix, τ::StridedROCVector, C::StridedROCVecOrMat) =
    YArocSOLVER.unmqr!(side, trans, A, τ, C)
_gpu_gesvd!(A::StridedROCMatrix, S::StridedROCVector, U::StridedROCMatrix, Vᴴ::StridedROCMatrix) =
    YArocSOLVER.gesvd!(A, S, U, Vᴴ)
# not yet supported
# _gpu_Xgesvdp!(A::StridedROCMatrix, S::StridedROCVector, U::StridedROCMatrix, Vᴴ::StridedROCMatrix; kwargs...) =
#     YArocSOLVER.Xgesvdp!(A, S, U, Vᴴ; kwargs...)
_gpu_gesvdj!(A::StridedROCMatrix, S::StridedROCVector, U::StridedROCMatrix, Vᴴ::StridedROCMatrix; kwargs...) =
    YArocSOLVER.gesvdj!(A, S, U, Vᴴ; kwargs...)
_gpu_heevj!(A::StridedROCMatrix, Dd::StridedROCVector, V::StridedROCMatrix; kwargs...) =
    YArocSOLVER.heevj!(A, Dd, V; kwargs...)
_gpu_heevd!(A::StridedROCMatrix, Dd::StridedROCVector, V::StridedROCMatrix; kwargs...) =
    YArocSOLVER.heevd!(A, Dd, V; kwargs...)
_gpu_heev!(A::StridedROCMatrix, Dd::StridedROCVector, V::StridedROCMatrix; kwargs...) =
    YArocSOLVER.heev!(A, Dd, V; kwargs...)
_gpu_heevx!(A::StridedROCMatrix, Dd::StridedROCVector, V::StridedROCMatrix; kwargs...) =
    YArocSOLVER.heevx!(A, Dd, V; kwargs...)

function MatrixAlgebraKit.findtruncated_svd(values::StridedROCVector, strategy::TruncationByValue)
    return MatrixAlgebraKit.findtruncated(values, strategy)
end

# COV_EXCL_START
function _project_hermitian_offdiag_kernel(Au, Al, Bu, Bl, ::Val{true})
    m, n = size(Au)
    j = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x
    j > n && return
    for i in 1:m
        @inbounds begin
            val = (Au[i, j] - adjoint(Al[j, i])) / 2
            Bu[i, j] = val
            Bl[j, i] = -adjoint(val)
        end
    end
    return
end

function _project_hermitian_offdiag_kernel(Au, Al, Bu, Bl, ::Val{false})
    m, n = size(Au)
    j = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x
    j > n && return
    for i in 1:m
        @inbounds begin
            val = (Au[i, j] + adjoint(Al[j, i])) / 2
            Bu[i, j] = val
            Bl[j, i] = adjoint(val)
        end
    end
    return
end

function _project_hermitian_diag_kernel(A, B, ::Val{true})
    n = size(A, 1)
    j = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x
    j > n && return
    @inbounds begin
        for i in 1:(j - 1)
            val = (A[i, j] - adjoint(A[j, i])) / 2
            B[i, j] = val
            B[j, i] = -adjoint(val)
        end
        B[j, j] = MatrixAlgebraKit._imimag(A[j, j])
    end
    return
end

function _project_hermitian_diag_kernel(A, B, ::Val{false})
    n = size(A, 1)
    j = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x
    j > n && return
    @inbounds begin
        for i in 1:(j - 1)
            val = (A[i, j] + adjoint(A[j, i])) / 2
            B[i, j] = val
            B[j, i] = adjoint(val)
        end
        B[j, j] = real(A[j, j])
    end
    return
end
# COV_EXCL_STOP

const SupportedROCMatrix{T} = Union{AnyROCMatrix{T}, SubArray{T, 2, <:AnyROCMatrix{T}}}

function MatrixAlgebraKit._project_hermitian_offdiag!(
        Au::SupportedROCMatrix, Al::SupportedROCMatrix, Bu::SupportedROCMatrix, Bl::SupportedROCMatrix, ::Val{anti}
    ) where {anti}
    thread_dim = 512
    block_dim = cld(size(Au, 2), thread_dim)
    @roc groupsize = thread_dim gridsize = block_dim _project_hermitian_offdiag_kernel(Au, Al, Bu, Bl, Val(anti))
    return nothing
end
function MatrixAlgebraKit._project_hermitian_diag!(A::SupportedROCMatrix, B::SupportedROCMatrix, ::Val{anti}) where {anti}
    thread_dim = 512
    block_dim = cld(size(A, 1), thread_dim)
    @roc groupsize = thread_dim gridsize = block_dim _project_hermitian_diag_kernel(A, B, Val(anti))
    return nothing
end

# avoids calling the `StridedMatrix` specialization to avoid scalar indexing,
# use (allocating) fallback instead until we write a dedicated kernel
MatrixAlgebraKit.ishermitian_exact(A::StridedROCMatrix) = A == A'
MatrixAlgebraKit.ishermitian_approx(A::StridedROCMatrix; atol, rtol, kwargs...) =
    norm(project_antihermitian(A; kwargs...)) ≤ max(atol, rtol * norm(A))
MatrixAlgebraKit.isantihermitian_exact(A::StridedROCMatrix) = A == -A'
MatrixAlgebraKit.isantihermitian_approx(A::StridedROCMatrix; atol, rtol, kwargs...) =
    norm(project_hermitian(A; kwargs...)) ≤ max(atol, rtol * norm(A))

function MatrixAlgebraKit._avgdiff!(A::StridedROCMatrix, B::StridedROCMatrix)
    axes(A) == axes(B) || throw(DimensionMismatch())
    # COV_EXCL_START
    function _avgdiff_kernel(A, B)
        j = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x
        j > length(A) && return
        @inbounds begin
            a = A[j]
            b = B[j]
            A[j] = (a + b) / 2
            B[j] = b - a
        end
        return
    end
    # COV_EXCL_STOP
    thread_dim = 512
    block_dim = cld(length(A), thread_dim)
    @roc groupsize = thread_dim gridsize = block_dim _avgdiff_kernel(A, B)
    return A, B
end

end
