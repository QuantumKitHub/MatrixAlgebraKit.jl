module MatrixAlgebraKitCUDAExt

using MatrixAlgebraKit
using MatrixAlgebraKit: @algdef, Algorithm, check_input
using MatrixAlgebraKit: one!, zero!, uppertriangular!, lowertriangular!
using MatrixAlgebraKit: diagview, sign_safe
using MatrixAlgebraKit: LQViaTransposedQR, TruncationByValue, AbstractAlgorithm
using MatrixAlgebraKit: default_qr_algorithm, default_lq_algorithm, default_svd_algorithm, default_eig_algorithm, default_eigh_algorithm
import MatrixAlgebraKit: _gpu_geqrf!, _gpu_ungqr!, _gpu_unmqr!, _gpu_gesvd!, _gpu_Xgesvdp!, _gpu_Xgesvdr!, _gpu_gesvdj!, _gpu_geev!
import MatrixAlgebraKit: _gpu_heevj!, _gpu_heevd!
using CUDA
using CUDA: i32
using LinearAlgebra
using LinearAlgebra: BlasFloat

include("yacusolver.jl")

function MatrixAlgebraKit.default_qr_algorithm(::Type{T}; kwargs...) where {T <: StridedCuMatrix}
    return CUSOLVER_HouseholderQR(; kwargs...)
end
function MatrixAlgebraKit.default_lq_algorithm(::Type{T}; kwargs...) where {T <: StridedCuMatrix}
    qr_alg = CUSOLVER_HouseholderQR(; kwargs...)
    return LQViaTransposedQR(qr_alg)
end
function MatrixAlgebraKit.default_svd_algorithm(::Type{T}; kwargs...) where {T <: StridedCuMatrix}
    return CUSOLVER_QRIteration(; kwargs...)
end
function MatrixAlgebraKit.default_eig_algorithm(::Type{T}; kwargs...) where {T <: StridedCuMatrix}
    return CUSOLVER_Simple(; kwargs...)
end
function MatrixAlgebraKit.default_eigh_algorithm(::Type{T}; kwargs...) where {T <: StridedCuMatrix}
    return CUSOLVER_DivideAndConquer(; kwargs...)
end


_gpu_geev!(A::StridedCuMatrix, D::StridedCuVector, V::StridedCuMatrix) =
    YACUSOLVER.Xgeev!(A, D, V)
_gpu_geqrf!(A::StridedCuMatrix) =
    YACUSOLVER.geqrf!(A)
_gpu_ungqr!(A::StridedCuMatrix, τ::StridedCuVector) =
    YACUSOLVER.ungqr!(A, τ)
_gpu_unmqr!(side::AbstractChar, trans::AbstractChar, A::StridedCuMatrix, τ::StridedCuVector, C::StridedCuVecOrMat) =
    YACUSOLVER.unmqr!(side, trans, A, τ, C)
_gpu_gesvd!(A::StridedCuMatrix, S::StridedCuVector, U::StridedCuMatrix, Vᴴ::StridedCuMatrix) =
    YACUSOLVER.gesvd!(A, S, U, Vᴴ)
_gpu_Xgesvdp!(A::StridedCuMatrix, S::StridedCuVector, U::StridedCuMatrix, Vᴴ::StridedCuMatrix; kwargs...) =
    YACUSOLVER.Xgesvdp!(A, S, U, Vᴴ; kwargs...)
_gpu_Xgesvdr!(A::StridedCuMatrix, S::StridedCuVector, U::StridedCuMatrix, Vᴴ::StridedCuMatrix; kwargs...) =
    YACUSOLVER.Xgesvdr!(A, S, U, Vᴴ; kwargs...)
_gpu_gesvdj!(A::StridedCuMatrix, S::StridedCuVector, U::StridedCuMatrix, Vᴴ::StridedCuMatrix; kwargs...) =
    YACUSOLVER.gesvdj!(A, S, U, Vᴴ; kwargs...)

_gpu_heevj!(A::StridedCuMatrix, Dd::StridedCuVector, V::StridedCuMatrix; kwargs...) =
    YACUSOLVER.heevj!(A, Dd, V; kwargs...)
_gpu_heevd!(A::StridedCuMatrix, Dd::StridedCuVector, V::StridedCuMatrix; kwargs...) =
    YACUSOLVER.heevd!(A, Dd, V; kwargs...)

function MatrixAlgebraKit.findtruncated_svd(values::StridedCuVector, strategy::TruncationByValue)
    return MatrixAlgebraKit.findtruncated(values, strategy)
end

# COV_EXCL_START
function _project_hermitian_offdiag_kernel(Au, Al, Bu, Bl, ::Val{true})
    m, n = size(Au)
    j = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
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
    j = threadIdx().x + (blockIdx().x - 1) * blockDim().x
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
    j = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j > n && return
    @inbounds begin
        for i in 1i32:(j - 1i32)
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
    j = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
    j > n && return
    @inbounds begin
        for i in 1i32:(j - 1i32)
            val = (A[i, j] + adjoint(A[j, i])) / 2
            B[i, j] = val
            B[j, i] = adjoint(val)
        end
        B[j, j] = real(A[j, j])
    end
    return
end
# COV_EXCL_STOP

function MatrixAlgebraKit._project_hermitian_offdiag!(
        Au::StridedCuMatrix, Al::StridedCuMatrix, Bu::StridedCuMatrix, Bl::StridedCuMatrix, ::Val{anti}
    ) where {anti}
    thread_dim = 512
    block_dim = cld(size(Au, 2), thread_dim)
    @cuda threads = thread_dim blocks = block_dim _project_hermitian_offdiag_kernel(Au, Al, Bu, Bl, Val(anti))
    return nothing
end
function MatrixAlgebraKit._project_hermitian_diag!(A::StridedCuMatrix, B::StridedCuMatrix, ::Val{anti}) where {anti}
    thread_dim = 512
    block_dim = cld(size(A, 1), thread_dim)
    @cuda threads = thread_dim blocks = block_dim _project_hermitian_diag_kernel(A, B, Val(anti))
    return nothing
end

MatrixAlgebraKit.ishermitian_exact(A::StridedCuMatrix) =
    all(A .== adjoint(A))
MatrixAlgebraKit.ishermitian_exact(A::Diagonal{T, <:StridedCuVector{T}}) where {T} =
    all(A.diag .== adjoint(A.diag))
MatrixAlgebraKit.ishermitian_approx(A::StridedCuMatrix; kwargs...) =
    @invoke MatrixAlgebraKit.ishermitian_approx(A::Any; kwargs...)

MatrixAlgebraKit.isantihermitian_exact(A::StridedCuMatrix) =
    all(A .== -adjoint(A))
MatrixAlgebraKit.isantihermitian_exact(A::Diagonal{T, <:StridedCuVector{T}}) where {T} =
    all(A.diag .== -adjoint(A.diag))
MatrixAlgebraKit.isantihermitian_approx(A::StridedCuMatrix; kwargs...) =
    @invoke MatrixAlgebraKit.isantihermitian_approx(A::Any; kwargs...)

function MatrixAlgebraKit._avgdiff!(A::StridedCuMatrix, B::StridedCuMatrix)
    axes(A) == axes(B) || throw(DimensionMismatch())
    # COV_EXCL_START
    function _avgdiff_kernel(A, B)
        j = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
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
    @cuda threads = thread_dim blocks = block_dim _avgdiff_kernel(A, B)
    return A, B
end

end
