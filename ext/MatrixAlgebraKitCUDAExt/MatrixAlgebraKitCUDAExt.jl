module MatrixAlgebraKitCUDAExt

using MatrixAlgebraKit
using MatrixAlgebraKit: @algdef, Algorithm, check_input
using MatrixAlgebraKit: one!, zero!, uppertriangular!, lowertriangular!
using MatrixAlgebraKit: diagview, sign_safe
using MatrixAlgebraKit: CUSOLVER, LQViaTransposedQR, TruncationByValue, AbstractAlgorithm
using MatrixAlgebraKit: default_qr_algorithm, default_lq_algorithm, default_svd_algorithm, default_eig_algorithm, default_eigh_algorithm
import MatrixAlgebraKit: geqrf!, ungqr!, unmqr!, gesvd!, gesvdp!, gesvdr!, gesvdj!
import MatrixAlgebraKit: heevj!, heevd!, geev!
import MatrixAlgebraKit: _gpu_Xgesvdr!, _sylvester, svd_rank
using CUDA, CUDA.CUBLAS
using CUDA: i32
using LinearAlgebra
using LinearAlgebra: BlasFloat

include("yacusolver.jl")

MatrixAlgebraKit.default_driver(::Type{TA}) where {TA <: StridedCuVecOrMat{<:BlasFloat}} = CUSOLVER()

function MatrixAlgebraKit.default_svd_algorithm(::Type{T}; kwargs...) where {T <: StridedCuVecOrMat{<:BlasFloat}}
    return QRIteration(; kwargs...)
end
function MatrixAlgebraKit.default_eig_algorithm(::Type{T}; kwargs...) where {T <: StridedCuVecOrMat{<:BlasFloat}}
    return QRIteration(; balanced = false, kwargs...)
end
function MatrixAlgebraKit.default_eigh_algorithm(::Type{T}; kwargs...) where {T <: StridedCuVecOrMat{<:BlasFloat}}
    return DivideAndConquer(; kwargs...)
end


for f in (:geqrf!, :ungqr!, :unmqr!)
    @eval $f(::CUSOLVER, args...) = YACUSOLVER.$f(args...)
end

MatrixAlgebraKit.supports_svd(::CUSOLVER, f::Symbol) = f in (:qr_iteration, :jacobi, :svd_polar)
MatrixAlgebraKit.supports_svd_full(::CUSOLVER, f::Symbol) = f in (:qr_iteration, :jacobi, :svd_polar)

function gesvd!(::CUSOLVER, A::StridedCuMatrix, S::StridedCuVector, U::StridedCuMatrix, Vᴴ::StridedCuMatrix; kwargs...)
    m, n = size(A)
    m >= n && return YACUSOLVER.gesvd!(A, S, U, Vᴴ)
    return MatrixAlgebraKit.svd_via_adjoint!(gesvd!, CUSOLVER(), A, S, U, Vᴴ; kwargs...)
end

function gesvdj!(::CUSOLVER, A::StridedCuMatrix, S::StridedCuVector, U::StridedCuMatrix, Vᴴ::StridedCuMatrix; kwargs...)
    m, n = size(A)
    m >= n && return YACUSOLVER.gesvdj!(A, S, U, Vᴴ; kwargs...)
    return MatrixAlgebraKit.svd_via_adjoint!(gesvdj!, CUSOLVER(), A, S, U, Vᴴ; kwargs...)
end

gesvdp!(::CUSOLVER, A::StridedCuMatrix, S::StridedCuVector, U::StridedCuMatrix, Vᴴ::StridedCuMatrix; kwargs...) =
    YACUSOLVER.gesvdp!(A, S, U, Vᴴ; kwargs...)

_gpu_Xgesvdr!(A::StridedCuMatrix, S::StridedCuVector, U::StridedCuMatrix, Vᴴ::StridedCuMatrix; kwargs...) =
    YACUSOLVER.gesvdr!(A, S, U, Vᴴ; kwargs...)

geev!(::CUSOLVER, A::StridedCuMatrix, Dd::StridedCuVector, V::StridedCuMatrix) =
    YACUSOLVER.Xgeev!(A, Dd, V)

heevj!(::CUSOLVER, A::StridedCuMatrix, Dd::StridedCuVector, V::StridedCuMatrix; kwargs...) =
    YACUSOLVER.heevj!(A, Dd, V; kwargs...)
heevd!(::CUSOLVER, A::StridedCuMatrix, Dd::StridedCuVector, V::StridedCuMatrix; kwargs...) =
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

const SupportedCuMatrix{T} = Union{AnyCuMatrix{T}, SubArray{T, 2, <:AnyCuMatrix{T}}}

function MatrixAlgebraKit._project_hermitian_offdiag!(
        Au::SupportedCuMatrix, Al::SupportedCuMatrix, Bu::SupportedCuMatrix, Bl::SupportedCuMatrix, ::Val{anti}
    ) where {anti}
    thread_dim = 512
    block_dim = cld(size(Au, 2), thread_dim)
    @cuda threads = thread_dim blocks = block_dim _project_hermitian_offdiag_kernel(Au, Al, Bu, Bl, Val(anti))
    return nothing
end
function MatrixAlgebraKit._project_hermitian_diag!(A::SupportedCuMatrix, B::SupportedCuMatrix, ::Val{anti}) where {anti}
    thread_dim = 512
    block_dim = cld(size(A, 1), thread_dim)
    @cuda threads = thread_dim blocks = block_dim _project_hermitian_diag_kernel(A, B, Val(anti))
    return nothing
end

# avoids calling the `StridedMatrix` specialization to avoid scalar indexing,
# use (allocating) fallback instead until we write a dedicated kernel
MatrixAlgebraKit.ishermitian_exact(A::StridedCuMatrix) = A == A'
MatrixAlgebraKit.ishermitian_approx(A::StridedCuMatrix; atol, rtol, kwargs...) =
    norm(project_antihermitian(A; kwargs...)) ≤ max(atol, rtol * norm(A))
MatrixAlgebraKit.isantihermitian_exact(A::StridedCuMatrix) = A == -A'
MatrixAlgebraKit.isantihermitian_approx(A::StridedCuMatrix; atol, rtol, kwargs...) =
    norm(project_hermitian(A; kwargs...)) ≤ max(atol, rtol * norm(A))

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

# avoids calling the BlasMat specialization that assumes syrk! or herk! is called
# TODO: remove once syrk! or herk! is defined
function MatrixAlgebraKit._mul_herm!(C::StridedCuMatrix{T}, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    mul!(C, A, A')
    project_hermitian!(C)
    return C
end

# TODO: intersect/union don't work on GPU
MatrixAlgebraKit._ind_intersect(A::CuVector{Int}, B::CuVector{Int}) =
    MatrixAlgebraKit._ind_intersect(collect(A), collect(B))
MatrixAlgebraKit._ind_union(A::AbstractVector{<:Integer}, B::CuVector{Int}) =
    MatrixAlgebraKit._ind_union(A, collect(B))
MatrixAlgebraKit._ind_union(A::CuVector{Int}, B::AbstractVector{<:Integer}) =
    MatrixAlgebraKit._ind_union(collect(A), B)
MatrixAlgebraKit._ind_union(A::CuVector{Int}, B::CuVector{Int}) =
    MatrixAlgebraKit._ind_union(collect(A), collect(B))

function _sylvester(A::AnyCuMatrix, B::AnyCuMatrix, C::AnyCuMatrix)
    # https://github.com/JuliaGPU/CUDA.jl/issues/3021
    # to add native sylvester to CUDA
    hX = sylvester(collect(A), collect(B), collect(C))
    return CuArray(hX)
end

svd_rank(S::AnyCuVector, rank_atol) = findlast(s -> s ≥ rank_atol, S)

end
