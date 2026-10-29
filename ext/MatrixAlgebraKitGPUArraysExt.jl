module MatrixAlgebraKitGPUArraysExt

using MatrixAlgebraKit, GPUArrays, LinearAlgebra
using GPUArrays.KernelAbstractions: @kernel, @index, get_backend

# one kernel for the whole triangle instead of one per column
function MatrixAlgebraKit.uppertriangular!(A::AnyGPUMatrix)
    m, n = size(A)
    A .= ifelse.((1:m) .> transpose(1:n), zero(eltype(A)), A)
    return A
end

@kernel function _project_hermitian_kernel(A, B, ::Val{anti}) where {anti}
    n = size(A, 1)
    i, j = @index(Global, NTuple)
    if !(i > n || j > n || i > j)
        @inbounds if i == j
            B[j, j] = anti ? MatrixAlgebraKit._imimag(A[j, j]) : real(A[j, j])
        else
            aij = A[i, j]
            aji = A[j, i]
            val = anti ? (aij - adjoint(aji)) / 2 : (aij + adjoint(aji)) / 2
            B[i, j] = val
            B[j, i] = anti ? -adjoint(val) : adjoint(val)
        end
    end
end

const SupportedGPUMatrix{T} = Union{AnyGPUMatrix{T}, SubArray{T, 2, <:AnyGPUMatrix{T}}}

function MatrixAlgebraKit.project_hermitian_native!(
        A::SupportedGPUMatrix, B::SupportedGPUMatrix, anti::Val; blocksize = 32
    )
    n = size(A, 1)
    n == 0 && return B
    _project_hermitian_kernel(get_backend(A), (16, 16), (n, n))(A, B, anti)
    return B
end

@kernel function _avgdiff_kernel(A, B)
    j = @index(Global)
    if j ≤ length(A)
        @inbounds begin
            a = A[j]
            b = B[j]
            A[j] = (a + b) / 2
            B[j] = b - a
        end
    end
end

function MatrixAlgebraKit._avgdiff!(A::AnyGPUMatrix, B::AnyGPUMatrix)
    axes(A) == axes(B) || throw(DimensionMismatch())
    _avgdiff_kernel(get_backend(A), 512, length(A))(A, B)
    return A, B
end

function MatrixAlgebraKit.findtruncated_svd(values::AnyGPUVector, strategy::MatrixAlgebraKit.TruncationByValue)
    return MatrixAlgebraKit.findtruncated(values, strategy)
end

# avoids calling the `StridedMatrix` specialization to avoid scalar indexing,
# use (allocating) fallback instead until we write a dedicated kernel
MatrixAlgebraKit.ishermitian_exact(A::AnyGPUMatrix) = A == A'
MatrixAlgebraKit.ishermitian_approx(A::AnyGPUMatrix; atol, rtol, kwargs...) =
    norm(project_antihermitian(A; kwargs...)) ≤ max(atol, rtol * norm(A))
MatrixAlgebraKit.isantihermitian_exact(A::AnyGPUMatrix) = A == -A'
MatrixAlgebraKit.isantihermitian_approx(A::AnyGPUMatrix; atol, rtol, kwargs...) =
    norm(project_hermitian(A; kwargs...)) ≤ max(atol, rtol * norm(A))

# TODO: intersect/union don't work on GPU
MatrixAlgebraKit._ind_intersect(A::AnyGPUVector{Int}, B::AnyGPUVector{Int}) =
    MatrixAlgebraKit._ind_intersect(collect(A), collect(B))
MatrixAlgebraKit._ind_union(A::AbstractVector{<:Integer}, B::AnyGPUVector{Int}) =
    MatrixAlgebraKit._ind_union(A, collect(B))
MatrixAlgebraKit._ind_union(A::AnyGPUVector{Int}, B::AbstractVector{<:Integer}) =
    MatrixAlgebraKit._ind_union(collect(A), B)
MatrixAlgebraKit._ind_union(A::AnyGPUVector{Int}, B::AnyGPUVector{Int}) =
    MatrixAlgebraKit._ind_union(collect(A), collect(B))

function MatrixAlgebraKit.svd_rank(S::AnyGPUVector; rank_atol = MatrixAlgebraKit.default_pullback_rank_atol(S))
    return something(findlast(≥(rank_atol), S), 0)
end

function MatrixAlgebraKit.svd_pullback!(ΔA::AnyGPUMatrix, A, USVᴴ, ΔUSVᴴ, ind::AnyGPUVector; kwargs...)
    return MatrixAlgebraKit.svd_pullback!(ΔA, A, USVᴴ, ΔUSVᴴ, collect(ind); kwargs...)
end

# have to override this as methods are missing in GPUArrays for the various
# views of Diagonal of ΔA
function MatrixAlgebraKit.svd_pushforward!(
        ΔA::Diagonal{T, <:AnyGPUVector{T}}, A, USVᴴ, ΔUSVᴴ, ind = Colon();
        rank_atol::Real = MatrixAlgebraKit.default_pullback_rank_atol(USVᴴ[2]),
        degeneracy_atol::Real = MatrixAlgebraKit.default_pullback_rank_atol(USVᴴ[2])
    ) where {T}
    return MatrixAlgebraKit.svd_pushforward!(diagm(diagview(ΔA)), A, USVᴴ, ΔUSVᴴ, ind; rank_atol, degeneracy_atol)
end

end
