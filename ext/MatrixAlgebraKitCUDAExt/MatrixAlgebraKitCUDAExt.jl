module MatrixAlgebraKitCUDAExt

using MatrixAlgebraKit
using MatrixAlgebraKit: @algdef, Algorithm, check_input
using MatrixAlgebraKit: one!, zero!, uppertriangular!, lowertriangular!
using MatrixAlgebraKit: diagview, sign_safe
using MatrixAlgebraKit: LQViaTransposedQR, TruncationByValue
using MatrixAlgebraKit: default_qr_algorithm, default_lq_algorithm, default_svd_algorithm, default_eig_algorithm, default_eigh_algorithm
import MatrixAlgebraKit: _gpu_geqrf!, _gpu_ungqr!, _gpu_unmqr!, _gpu_gesvd!, _gpu_Xgesvdp!, _gpu_Xgesvdr!, _gpu_gesvdj!, _gpu_geev!
import MatrixAlgebraKit: _gpu_heevj!, _gpu_heevd!
using CUDA, CUDA.CUBLAS
using LinearAlgebra
using LinearAlgebra: BlasFloat

using CUDA: i32

include("yacusolver.jl")

function MatrixAlgebraKit.default_qr_algorithm(::Type{T}; kwargs...) where {TT<:BlasFloat, T<:StridedCuMatrix{TT}}
    return CUSOLVER_HouseholderQR(; kwargs...)
end
function MatrixAlgebraKit.default_lq_algorithm(::Type{T}; kwargs...) where {TT<:BlasFloat, T<:StridedCuMatrix{TT}}
    qr_alg = CUSOLVER_HouseholderQR(; kwargs...)
    return LQViaTransposedQR(qr_alg)
end
function MatrixAlgebraKit.default_svd_algorithm(::Type{T}; kwargs...) where {TT<:BlasFloat, T<:StridedCuMatrix{TT}}
    return CUSOLVER_QRIteration(; kwargs...)
end
function MatrixAlgebraKit.default_eig_algorithm(::Type{T}; kwargs...) where {TT<:BlasFloat, T<:StridedCuMatrix{TT}}
    return CUSOLVER_Simple(; kwargs...)
end
function MatrixAlgebraKit.default_eigh_algorithm(::Type{T}; kwargs...) where {TT<:BlasFloat, T<:StridedCuMatrix{TT}}
    return CUSOLVER_DivideAndConquer(; kwargs...)
end

# include for block sector support
function MatrixAlgebraKit.default_qr_algorithm(::Type{Base.ReshapedArray{T,2,SubArray{T,1,A,Tuple{UnitRange{Int}},true},Tuple{}}}; kwargs...) where {T<:BlasFloat, A<:CuVecOrMat{T}}
    return CUSOLVER_HouseholderQR(; kwargs...)
end
function MatrixAlgebraKit.default_lq_algorithm(::Type{Base.ReshapedArray{T,2,SubArray{T,1,A,Tuple{UnitRange{Int}},true},Tuple{}}}; kwargs...) where {T<:BlasFloat, A<:CuVecOrMat{T}}
    qr_alg = CUSOLVER_HouseholderQR(; kwargs...)
    return LQViaTransposedQR(qr_alg)
end
function MatrixAlgebraKit.default_svd_algorithm(::Type{Base.ReshapedArray{T,2,SubArray{T,1,A,Tuple{UnitRange{Int}},true},Tuple{}}}; kwargs...) where {T<:BlasFloat, A<:CuVecOrMat{T}}
    return CUSOLVER_Jacobi(; kwargs...)
end
function MatrixAlgebraKit.default_eig_algorithm(::Type{Base.ReshapedArray{T,2,SubArray{T,1,A,Tuple{UnitRange{Int}},true},Tuple{}}}; kwargs...) where {T<:BlasFloat, A<:CuVecOrMat{T}}
    return CUSOLVER_Simple(; kwargs...)
end
function MatrixAlgebraKit.default_eigh_algorithm(::Type{Base.ReshapedArray{T,2,SubArray{T,1,A,Tuple{UnitRange{Int}},true},Tuple{}}}; kwargs...) where {T<:BlasFloat, A<:CuVecOrMat{T}}
    return CUSOLVER_DivideAndConquer(; kwargs...)
end

MatrixAlgebraKit.ishermitian_exact(A::StridedCuMatrix) = ishermitian(A)

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


function _project_hermitian_offdiag_kernel(Au, Al, Bu, Bl, ::Val{true})
    m, n = size(Au)
    j = threadIdx().x + blockIdx().x * (blockDim().x - 1i32)
    j > n && return
    for i in 1:m
        @inbounds begin
            val = (Au[i, j] - adjoint(Al[j, i])) / 2
            Bu[i, j] = val
            aval = adjoint(val)
            Bl[j, i] = -aval
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
            val = Au[i, j] + adjoint(Al[j, i]) / 2
            Bu[i, j] = val
            aval = adjoint(val)
            Bl[j, i] = aval
        end
    end
    return
end

function _project_hermitian_diag_kernel(A, B, n::Int32, ::Val{true})
    j = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j > n && return
    @inbounds begin
        for i in 1i32:(j - 1i32)
            val = (A[i, j] - conj(A[j, i])) /2
            B[i, j] = val * 0.5
            aval = adjoint(val)
            B[j, i] = -aval
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
            val = (A[i, j] + adjoint(A[j, i]))
            B[i, j] = val * 0.5
            aval = adjoint(val)
            B[j, i] = aval
        end
        B[j, j] = real(A[j, j])
    end
    return
end

function MatrixAlgebraKit._project_hermitian_offdiag!(
        Au::StridedCuMatrix, Al::StridedCuMatrix, Bu::StridedCuMatrix, Bl::StridedCuMatrix, ::Val{anti}
    ) where {anti}
    thread_dim = 512
    block_dim = cld(size(Au, 1), thread_dim)
    @cuda threads=thread_dim blocks=block_dim _project_hermitian_offdiag_kernel(Au, Al, Bu, Bl, Val(anti))
    return nothing
end
function MatrixAlgebraKit._project_hermitian_diag!(A::StridedCuMatrix, B::StridedCuMatrix, ::Val{anti}) where {anti}
    thread_dim = 512
    block_dim = cld(size(A, 1), thread_dim)
    @cuda threads=thread_dim blocks=block_dim _project_hermitian_diag_kernel(A, B, Int32(size(A, 1)), Val(anti))
    return nothing
end

end
