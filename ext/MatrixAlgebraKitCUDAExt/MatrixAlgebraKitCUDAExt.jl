module MatrixAlgebraKitCUDAExt

using MatrixAlgebraKit
using MatrixAlgebraKit: @algdef, Algorithm, check_input
using MatrixAlgebraKit: one!, zero!, uppertriangular!, lowertriangular!
using MatrixAlgebraKit: diagview, sign_safe
using MatrixAlgebraKit: CUSOLVER, LQViaTransposedQR, TruncationByValue, AbstractAlgorithm
using MatrixAlgebraKit: default_qr_algorithm, default_lq_algorithm, default_svd_algorithm, default_eig_algorithm, default_eigh_algorithm
import MatrixAlgebraKit: geqrf!, ungqr!, unmqr!, gesvd!, gesvdp!, gesvdr!, gesvdj!
import MatrixAlgebraKit: gesvdj_batched!, heev_batched!, heevj_batched!
import MatrixAlgebraKit: heevj!, heevd!, geev!
import MatrixAlgebraKit: _gpu_Xgesvdr!, _sylvester
using CUDA, CUDA.cuBLAS
using CUDA: i32
using LinearAlgebra
using LinearAlgebra: BlasFloat

include("yacusolver.jl")

MatrixAlgebraKit.default_driver(::Type{TA}) where {TA <: StridedCuArray{<:BlasFloat}} = CUSOLVER()

function MatrixAlgebraKit.default_svd_algorithm(::Type{T}; kwargs...) where {T <: StridedCuVecOrMat{<:BlasFloat}}
    return QRIteration(; kwargs...)
end
function MatrixAlgebraKit.default_svd_algorithm(::Type{T}; kwargs...) where {T <: StridedCuArray{<:BlasFloat, 3}}
    return Jacobi(; kwargs...)
end
# CUSOLVER only batches `Jacobi`, so don't fall through to the single-matrix default
function MatrixAlgebraKit.default_svd_algorithm(::Type{T}; kwargs...) where {T <: AbstractVector{<:StridedCuMatrix{<:BlasFloat}}}
    return default_svd_algorithm(CuArray{eltype(eltype(T)), 3}; kwargs...)
end
function MatrixAlgebraKit.default_eig_algorithm(::Type{T}; kwargs...) where {T <: StridedCuVecOrMat{<:BlasFloat}}
    return QRIteration(; kwargs...)
end
function MatrixAlgebraKit.default_eigh_algorithm(::Type{T}; kwargs...) where {T <: StridedCuVecOrMat{<:BlasFloat}}
    return DivideAndConquer(; kwargs...)
end
function MatrixAlgebraKit.default_eigh_algorithm(::Type{T}; kwargs...) where {T <: StridedCuArray{<:BlasFloat, 3}}
    return Jacobi(; kwargs...)
end
function MatrixAlgebraKit.default_eigh_algorithm(::Type{T}; kwargs...) where {T <: Vector{<:StridedCuMatrix{<:BlasFloat}}}
    return Jacobi(; kwargs...)
end

for f in (:geqrf!, :ungqr!, :unmqr!)
    @eval $f(::CUSOLVER, args...) = YACUSOLVER.$f(args...)
end

MatrixAlgebraKit.prefers_ungqr(::CUSOLVER) = true

MatrixAlgebraKit.supports_svd_full(::CUSOLVER, f::Symbol) = f in (:qr_iteration, :jacobi, :svd_polar)

# `cusolverDnXgesvdjBatched` only accepts matrices up to 32x32
MatrixAlgebraKit.max_batched_blocksize(::AbstractAlgorithm, ::CUSOLVER, ::Type{<:AnyCuArray}) = 32
# CUSOLVER only provides a batched kernel for `Jacobi`
MatrixAlgebraKit.supports_ragged_batch(::typeof(svd_full!), ::Jacobi, ::CUSOLVER, ::Type{<:AnyCuArray}) = true
MatrixAlgebraKit.supports_ragged_batch(::typeof(svd_compact!), ::Jacobi, ::CUSOLVER, ::Type{<:AnyCuArray}) = true
MatrixAlgebraKit.supports_ragged_batch(::typeof(svd_vals!), ::Jacobi, ::CUSOLVER, ::Type{<:AnyCuArray}) = true
MatrixAlgebraKit.supports_ragged_batch(::typeof(eigh_full!), ::Jacobi, ::CUSOLVER, ::Type{<:AnyCuArray}) = true
MatrixAlgebraKit.supports_ragged_batch(::typeof(eigh_vals!), ::Jacobi, ::CUSOLVER, ::Type{<:AnyCuArray}) = true

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

heevj_batched!(::CUSOLVER, As::StridedCuArray{T, 3}, Ds::StridedCuMatrix, Vs::StridedCuArray{T, 3}; kwargs...) where {T <: BlasFloat} =
    YACUSOLVER.heevj_batched!(As, Ds, Vs; kwargs...)

heev_batched!(::CUSOLVER, As::StridedCuArray{T, 3}, Ds::StridedCuMatrix, Vs::StridedCuArray{T, 3}; kwargs...) where {T <: BlasFloat} =
    YACUSOLVER.heev_batched!(As, Ds, Vs; kwargs...)

gesvdj_batched!(::CUSOLVER, As::StridedCuArray{T, 3}, Ss::StridedCuMatrix, Us::StridedCuArray{T, 3}, Vᴴs::StridedCuArray{T, 3}; kwargs...) where {T <: BlasFloat} =
    YACUSOLVER.gesvdj_batched!(As, Ss, Us, Vᴴs; kwargs...)

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

# avoids calling the BlasMat specialization that assumes syrk! or herk! is called
# TODO: remove once syrk! or herk! is defined
function MatrixAlgebraKit._mul_herm!(C::StridedCuMatrix{T}, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    mul!(C, A, A')
    project_hermitian!(C)
    return C
end

function _sylvester(A::AnyCuMatrix, B::AnyCuMatrix, C::AnyCuMatrix)
    # https://github.com/JuliaGPU/CUDA.jl/issues/3021
    # to add native sylvester to CUDA
    hX = sylvester(collect(A), collect(B), collect(C))
    return CuArray(hX)
end

end
