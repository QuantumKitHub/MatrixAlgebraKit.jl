module MatrixAlgebraKitAMDGPUExt

using MatrixAlgebraKit
using MatrixAlgebraKit: @algdef, Algorithm, check_input
using MatrixAlgebraKit: one!, zero!, uppertriangular!, lowertriangular!
using MatrixAlgebraKit: diagview, sign_safe
using MatrixAlgebraKit: ROCSOLVER, LQViaTransposedQR, TruncationStrategy, NoTruncation, TruncationByValue, AbstractAlgorithm
using MatrixAlgebraKit: default_qr_algorithm, default_lq_algorithm, default_svd_algorithm, default_eigh_algorithm
import MatrixAlgebraKit: geqrf!, ungqr!, unmqr!, gesvd!, gesdd!, gesvdx!, gesvdj!
import MatrixAlgebraKit: gesvdj_batched!, gesdd_batched!, gesvd_batched!, gesvdx_batched!
import MatrixAlgebraKit: heevj!, heevd!, heev!, heevx!
import MatrixAlgebraKit: heevj_batched!, heevd_batched!, heev_batched!, heevx_batched!
import MatrixAlgebraKit: _sylvester, svd_rank, svd_pullback!
using AMDGPU
using LinearAlgebra
using LinearAlgebra: BlasFloat

include("yarocsolver.jl")

MatrixAlgebraKit.default_driver(::Type{TA}) where {TA <: StridedROCArray{<:BlasFloat}} = ROCSOLVER()
MatrixAlgebraKit.default_driver(::Type{TA}) where {TA <: AbstractVector{<:StridedROCMatrix{<:BlasFloat}}} = ROCSOLVER()

function MatrixAlgebraKit.default_svd_algorithm(::Type{T}; kwargs...) where {T <: StridedROCMatrix{<:BlasFloat}}
    return QRIteration(; kwargs...)
end
function MatrixAlgebraKit.default_svd_algorithm(::Type{T}; kwargs...) where {T <: StridedROCArray{<:BlasFloat, 3}}
    return QRIteration(; kwargs...)
end
function MatrixAlgebraKit.default_eigh_algorithm(::Type{T}; kwargs...) where {T <: StridedROCVecOrMat{<:BlasFloat}}
    return DivideAndConquer(; kwargs...)
end
function MatrixAlgebraKit.default_eigh_algorithm(::Type{T}; kwargs...) where {T <: StridedROCArray{<:BlasFloat, 3}}
    return Bisection(; kwargs...)
end
function MatrixAlgebraKit.default_eigh_algorithm(::Type{T}; kwargs...) where {T <: AbstractVector{<:StridedROCMatrix{<:BlasFloat}}}
    return Bisection(; kwargs...)
end

for f in (:geqrf!, :ungqr!, :unmqr!)
    @eval $f(::ROCSOLVER, args...) = YArocSOLVER.$f(args...)
end

MatrixAlgebraKit.supports_svd_full(::ROCSOLVER, f::Symbol) = f in (:qr_iteration, :jacobi, :divide_and_conquer, :bisection)

# rocSOLVER's `gesvd*_batched` functions take the batch as an array
# of device pointers, so a group of equally sized matrices
# doesn't need a copy into a 3D ROCArray.
# opt-in per algorithm to avoid picking up `DiagonalAlgorithm`
# which doesn't support batching
for alg in (:Bisection, :Jacobi, :QRIteration, :DivideAndConquer)
    @eval MatrixAlgebraKit.supports_pointer_batch(::$alg, ::ROCSOLVER, ::Type{<:StridedROCMatrix{<:BlasFloat}}) = true
    @eval MatrixAlgebraKit.supports_ragged_batch(::typeof(svd_full!), ::$alg, ::ROCSOLVER, ::Type{<:AnyROCArray}) = true
    @eval MatrixAlgebraKit.supports_ragged_batch(::typeof(svd_compact!), ::$alg, ::ROCSOLVER, ::Type{<:AnyROCArray}) = true
    @eval MatrixAlgebraKit.supports_ragged_batch(::typeof(svd_vals!), ::$alg, ::ROCSOLVER, ::Type{<:AnyROCArray}) = true
end

function gesvd!(::ROCSOLVER, A::StridedROCMatrix, S::StridedROCVector, U::StridedROCMatrix, Vᴴ::StridedROCMatrix; kwargs...)
    m, n = size(A)
    m >= n && return YArocSOLVER.gesvd!(A, S, U, Vᴴ)
    return MatrixAlgebraKit.svd_via_adjoint!(gesvd!, ROCSOLVER(), A, S, U, Vᴴ; kwargs...)
end

function gesvdj!(::ROCSOLVER, A::StridedROCMatrix, S::StridedROCVector, U::StridedROCMatrix, Vᴴ::StridedROCMatrix; kwargs...)
    m, n = size(A)
    m >= n && return YArocSOLVER.gesvdj!(A, S, U, Vᴴ; kwargs...)
    return MatrixAlgebraKit.svd_via_adjoint!(gesvdj!, ROCSOLVER(), A, S, U, Vᴴ; kwargs...)
end

function gesvdx!(::ROCSOLVER, A::StridedROCMatrix, S::StridedROCVector, U::StridedROCMatrix, Vᴴ::StridedROCMatrix; kwargs...)
    if min(size(A)...) == 1
        _gesvdx_rank1!(A, S, U, Vᴴ; kwargs...)
    else
        YArocSOLVER.gesvdx!(A, S, U, Vᴴ; kwargs...)
    end
    return S, U, Vᴴ
end

# rocSOLVER's batched `gesvd` requires m ≥ n, so wide matrices go through the adjoint
function gesvd_batched!(::ROCSOLVER, As::AbstractVector{<:StridedROCMatrix}, Ss::StridedROCMatrix, Us::StridedROCArray{T, 3}, Vᴴs::StridedROCArray{T, 3}; kwargs...) where {T <: BlasFloat}
    m, n = size(first(As))
    m >= n && return YArocSOLVER.gesvd_batched!(As, Ss, Us, Vᴴs; kwargs...)
    return MatrixAlgebraKit.batched_svd_via_adjoint!(gesvd_batched!, ROCSOLVER(), As, Ss, Us, Vᴴs; kwargs...)
end
function gesvd_batched!(::ROCSOLVER, As::StridedROCArray{T, 3}, Ss::StridedROCMatrix, Us::StridedROCArray{T, 3}, Vᴴs::StridedROCArray{T, 3}; kwargs...) where {T <: BlasFloat}
    m, n, _ = size(As)
    m >= n && return YArocSOLVER.gesvd_strided_batched!(As, Ss, Us, Vᴴs; kwargs...)
    return MatrixAlgebraKit.batched_svd_via_adjoint!(gesvd_batched!, ROCSOLVER(), As, Ss, Us, Vᴴs; kwargs...)
end

gesdd_batched!(::ROCSOLVER, As::AbstractVector{<:StridedROCMatrix}, Ss::StridedROCMatrix, Us::StridedROCArray{T, 3}, Vᴴs::StridedROCArray{T, 3}; kwargs...) where {T <: BlasFloat} =
    YArocSOLVER.gesdd_batched!(As, Ss, Us, Vᴴs; kwargs...)
gesdd_batched!(::ROCSOLVER, As::StridedROCArray{T, 3}, Ss::StridedROCMatrix, Us::StridedROCArray{T, 3}, Vᴴs::StridedROCArray{T, 3}; kwargs...) where {T <: BlasFloat} =
    YArocSOLVER.gesdd_strided_batched!(As, Ss, Us, Vᴴs; kwargs...)

gesvdj_batched!(::ROCSOLVER, As::AbstractVector{<:StridedROCMatrix}, Ss::StridedROCMatrix, Us::StridedROCArray{T, 3}, Vᴴs::StridedROCArray{T, 3}; kwargs...) where {T <: BlasFloat} =
    YArocSOLVER.gesvdj_batched!(As, Ss, Us, Vᴴs; kwargs...)
gesvdj_batched!(::ROCSOLVER, As::StridedROCArray{T, 3}, Ss::StridedROCMatrix, Us::StridedROCArray{T, 3}, Vᴴs::StridedROCArray{T, 3}; kwargs...) where {T <: BlasFloat} =
    YArocSOLVER.gesvdj_strided_batched!(As, Ss, Us, Vᴴs; kwargs...)

function gesvdx_batched!(::ROCSOLVER, As::AbstractVector{<:StridedROCMatrix}, Ss::StridedROCMatrix, Us::StridedROCArray{T, 3}, Vᴴs::StridedROCArray{T, 3}; kwargs...) where {T <: BlasFloat}
    if min(size(first(As))...) == 1
        _gesvdx_rank1_batched!(As, Ss, Us, Vᴴs; kwargs...)
    else
        YArocSOLVER.gesvdx_batched!(As, Ss, Us, Vᴴs; kwargs...)
    end
    return Ss, Us, Vᴴs
end
function gesvdx_batched!(::ROCSOLVER, As::StridedROCArray{T, 3}, Ss::StridedROCMatrix, Us::StridedROCArray{T, 3}, Vᴴs::StridedROCArray{T, 3}; kwargs...) where {T <: BlasFloat}
    if min(size(As, 1), size(As, 2)) == 1
        _gesvdx_rank1_batched!(As, Ss, Us, Vᴴs; kwargs...)
    else
        YArocSOLVER.gesvdx_strided_batched!(As, Ss, Us, Vᴴs; kwargs...)
    end
    return Ss, Us, Vᴴs
end

gesdd!(::ROCSOLVER, A::StridedROCMatrix, S::StridedROCVector, U::StridedROCMatrix, Vᴴ::StridedROCMatrix; kwargs...) =
    YArocSOLVER.gesdd!(A, S, U, Vᴴ; kwargs...)

function _gesvdx_rank1!(A::StridedROCMatrix, S::StridedROCVector, U::StridedROCMatrix, Vᴴ::StridedROCMatrix; kwargs...)
    select_kwargs = Base.structdiff(NamedTuple(kwargs), (; irange = nothing, vl = nothing, vu = nothing))
    nrmA = norm(A)
    fill!(S, nrmA)
    if !isempty(U) && !isempty(Vᴴ)
        u, v = view(U, :, 1), view(Vᴴ, 1, :)
        x, e = size(A, 1) == 1 ? (v, u) : (u, v)
        if iszero(nrmA)
            zero!(x)
            fill!(view(x, 1:1), one(eltype(x)))
        else
            copyto!(x, vec(A))
            x ./= nrmA
        end
        fill!(e, one(eltype(e)))
    end
    _gesvdx_apply_range!(S, select_kwargs)
    return S, U, Vᴴ
end

function _gesvdx_rank1_batched!(As, Ss::StridedROCMatrix, Us, Vᴴs; kwargs...)
    select_kwargs = Base.structdiff(NamedTuple(kwargs), (; irange = nothing, vl = nothing, vu = nothing))
    gesvdj_batched!(ROCSOLVER(), As, Ss, Us, Vᴴs; select_kwargs...)
    _gesvdx_apply_range!(Ss, select_kwargs)
    return Ss, Us, Vᴴs
end

function _gesvdx_apply_range!(S, select::NamedTuple)
    if haskey(select, :irange)
        1 in convert(UnitRange{Int}, select.irange) || zero!(S)
    elseif haskey(select, :vl) || haskey(select, :vu)
        vl = convert(eltype(S), get(select, :vl, -Inf))
        vu = convert(eltype(S), get(select, :vu, Inf))
        S .= ifelse.((vl .<= S) .& (S .< vu), S, zero(eltype(S)))
    end
    return S
end

heevj!(::ROCSOLVER, A::StridedROCMatrix, Dd::StridedROCVector, V::StridedROCMatrix; kwargs...) =
    YArocSOLVER.heevj!(A, Dd, V; kwargs...)
heevj_batched!(::ROCSOLVER, As::AbstractVector{<:StridedROCMatrix}, Ds::StridedROCMatrix, Vs::StridedROCArray{T, 3}; kwargs...) where {T <: BlasFloat} =
    YArocSOLVER.heevj_batched!(As, Ds, Vs; kwargs...)
heevj_batched!(::ROCSOLVER, As::StridedROCArray{T, 3}, Ds::StridedROCMatrix, Vs::StridedROCArray{T, 3}; kwargs...) where {T <: BlasFloat} =
    YArocSOLVER.heevj_strided_batched!(As, Ds, Vs; kwargs...)
heevd!(::ROCSOLVER, A::StridedROCMatrix, Dd::StridedROCVector, V::StridedROCMatrix; kwargs...) =
    YArocSOLVER.heevd!(A, Dd, V; kwargs...)
heevd_batched!(::ROCSOLVER, As::AbstractVector{<:StridedROCMatrix}, Ds::StridedROCMatrix, Vs::StridedROCArray{T, 3}; kwargs...) where {T <: BlasFloat} =
    YArocSOLVER.heevd_batched!(As, Ds, Vs; kwargs...)
heevd_batched!(::ROCSOLVER, As::StridedROCArray{T, 3}, Ds::StridedROCMatrix, Vs::StridedROCArray{T, 3}; kwargs...) where {T <: BlasFloat} =
    YArocSOLVER.heevd_strided_batched!(As, Ds, Vs; kwargs...)
heev!(::ROCSOLVER, A::StridedROCMatrix, Dd::StridedROCVector, V::StridedROCMatrix; kwargs...) =
    YArocSOLVER.heev!(A, Dd, V; kwargs...)
heev_batched!(::ROCSOLVER, As::AbstractVector{<:StridedROCMatrix}, Ds::StridedROCMatrix, Vs::StridedROCArray{T, 3}; kwargs...) where {T <: BlasFloat} =
    YArocSOLVER.heev_batched!(As, Ds, Vs; kwargs...)
heev_batched!(::ROCSOLVER, As::StridedROCArray{T, 3}, Ds::StridedROCMatrix, Vs::StridedROCArray{T, 3}; kwargs...) where {T <: BlasFloat} =
    YArocSOLVER.heev_strided_batched!(As, Ds, Vs; kwargs...)
heevx!(::ROCSOLVER, A::StridedROCMatrix, Dd::StridedROCVector, V::StridedROCMatrix; kwargs...) =
    YArocSOLVER.heevx!(A, Dd, V; kwargs...)
heevx_batched!(::ROCSOLVER, As::AbstractVector{<:StridedROCMatrix}, Ds::StridedROCMatrix, Vs::StridedROCArray{T, 3}; kwargs...) where {T <: BlasFloat} =
    YArocSOLVER.heevx_batched!(As, Ds, Vs; kwargs...)
heevx_batched!(::ROCSOLVER, As::StridedROCArray{T, 3}, Ds::StridedROCMatrix, Vs::StridedROCArray{T, 3}; kwargs...) where {T <: BlasFloat} =
    YArocSOLVER.heevx_strided_batched!(As, Ds, Vs; kwargs...)

# avoids calling the BlasMat specialization that assumes syrk! or herk! is called
# TODO: remove once syrk! or herk! is defined
function MatrixAlgebraKit._mul_herm!(C::StridedROCMatrix{T}, A::StridedROCMatrix{T}) where {T <: BlasFloat}
    mul!(C, A, A')
    project_hermitian!(C)
    return C
end

function _sylvester(A::AnyROCMatrix, B::AnyROCMatrix, C::AnyROCMatrix)
    hX = sylvester(collect(A), collect(B), collect(C))
    return ROCArray(hX)
end

end
