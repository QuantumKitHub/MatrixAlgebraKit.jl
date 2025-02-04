# SVD functions
# -------------
"""
    svd_full(A; kwargs...) -> U, S, Vᴴ
    svd_full(A, alg::AbstractAlgorithm) -> U, S, Vᴴ
    svd_full!(A, [USVᴴ]; kwargs...) -> U, S, Vᴴ
    svd_full!(A, [USVᴴ], alg::AbstractAlgorithm) -> U, S, Vᴴ

Compute the full singular value decomposition (SVD) of the rectangular matrix `A` of size
`(m, n)`, such that `A = U * S * Vᴴ`. Here, `U` and `Vᴴ` are unitary matrices of size
`(m, m)` and `(n, n)` respectively, and `S` is a diagonal matrix of size `(m, n)`.

!!! note
    The bang method `svd_full!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `USVᴴ` as output.

See also [`svd_compact(!)`](@ref svd_compact), [`svd_vals(!)`](@ref svd_vals) and
[`svd_trunc(!)`](@ref svd_trunc).
"""
@functiondef svd_full

"""
    svd_compact(A; kwargs...) -> U, S, Vᴴ
    svd_compact(A, alg::AbstractAlgorithm) -> U, S, Vᴴ
    svd_compact!(A, [USVᴴ]; kwargs...) -> U, S, Vᴴ
    svd_compact!(A, [USVᴴ], alg::AbstractAlgorithm) -> U, S, Vᴴ

Compute the compact singular value decomposition (SVD) of the rectangular matrix `A` of size
`(m, n)`, such that `A = U * S * Vᴴ`. Here, `U` is an isometric matrix (orthonormal columns)
of size `(m, k)`, whereas  `Vᴴ` is a matrix of size `(k, n)` with orthonormal rows and `S`
is a square diagonal matrix of size `(k, k)`, with `k = min(m, n)`.

!!! note
    The bang method `svd_compact!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `USVᴴ` as output.

See also [`svd_full(!)`](@ref svd_full), [`svd_vals(!)`](@ref svd_vals) and
[`svd_trunc(!)`](@ref svd_trunc).
"""
@functiondef svd_compact

# TODO: decide if we should have `svd_trunc!!` instead
"""
    svd_trunc(A; kwargs...) -> U, S, Vᴴ
    svd_trunc(A, alg::AbstractAlgorithm) -> U, S, Vᴴ
    svd_trunc!(A, [USVᴴ]; kwargs...) -> U, S, Vᴴ
    svd_trunc!(A, [USVᴴ], alg::AbstractAlgorithm) -> U, S, Vᴴ

Compute a partial or truncated singular value decomposition (SVD) of `A`, such that
`A * (Vᴴ)' =  U * S`. Here, `U` is an isometric matrix (orthonormal columns) of size
`(m, k)`, whereas  `Vᴴ` is a matrix of size `(k, n)` with orthonormal rows and `S` is a
square diagonal matrix of size `(k, k)`, with `k` is set by the truncation strategy.

!!! note
    The bang method `svd_trunc!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `USVᴴ` as output.


See also [`svd_full(!)`](@ref svd_full), [`svd_compact(!)`](@ref svd_compact) and
[`svd_vals(!)`](@ref svd_vals).
"""
@functiondef svd_trunc

# # TODO: could be `nullspace` with a `SVDAlgorithm` instead. 
# # -> Yes, disable for now
# # TODO: update docs for kwargs?
# """
#     svd_null(A; kwargs...) -> N
#     svd_null(A, alg::AbstractAlgorithm) -> N
#     svd_null!(A, [USVᴴ]; kwargs...) -> N
#     svd_null!(A, [USVᴴ], alg::AbstractAlgorithm) -> N

# Compute a basis for the nullspace of `A`, such that `A * N ≈ 0`. This is done
# by including the singular vectors of `A` whose singular values have magnitudes
# smaller than `max(atol, rtol*σ₁)`, where `σ₁` is the largest singular value.

# See also [`svd_full(!)`](@ref svd_full), [`svd_compact(!)`](@ref svd_compact),
# [`svd_trunc(!)`](@ref svd_trunc) and [`svd_vals(!)`](@ref svd_vals).
# """
# @functiondef svd_null

"""
    svd_vals(A; kwargs...) -> S
    svd_vals(A, alg::AbstractAlgorithm) -> S
    svd_vals!(A, [S]; kwargs...) -> S
    svd_vals!(A, [S], alg::AbstractAlgorithm) -> S

Compute the vector of singular values of `A`, such that for an M×N matrix `A`,
`S` is a vector of size `K = min(M, N)`, the number of kept singular values.

See also [`svd_full(!)`](@ref svd_full), [`svd_compact(!)`](@ref svd_compact) and
[`svd_trunc(!)`](@ref svd_trunc).
"""
@functiondef svd_vals

# Algorithm selection
# -------------------
for f in (:svd_full, :svd_compact, :svd_vals)
    f! = Symbol(f, :!)
    @eval begin
        function select_algorithm(::typeof($f), A; kwargs...)
            return select_algorithm($f!, A; kwargs...)
        end
        function select_algorithm(::typeof($f!), A; alg=nothing, kwargs...)
            if alg isa AbstractAlgorithm
                return alg
            elseif alg isa Symbol
                return Algorithm{alg}(; kwargs...)
            else
                isnothing(alg) || throw(ArgumentError("Unknown alg $alg"))
                return default_svd_algorithm(A; kwargs...)
            end
        end
    end
end

function select_algorithm(::typeof(svd_trunc), A; kwargs...)
    return select_algorithm(svd_trunc!, A; kwargs...)
end
function select_algorithm(::typeof(svd_trunc!), A; alg=nothing, trunc=nothing, kwargs...)
    alg_svd = select_algorithm(svd_compact!, A; alg, kwargs...)
    alg_trunc = trunc isa TruncationStrategy ? trunc :
                trunc isa NamedTuple ? TruncationStrategy(; trunc...) :
                isnothing(trunc) ? NoTruncation() :
                throw(ArgumentError("Unknown truncation strategy: $trunc"))
    return TruncatedAlgorithm(alg_svd, alg_trunc)
end

# Default to LAPACK SDD for `StridedMatrix{<:BlasFloat}`
function default_svd_algorithm(A::StridedMatrix{<:BlasFloat}; kwargs...)
    isempty(kwargs) || throw(ArgumentError("Unexpected kwargs: $kwargs"))
    return LAPACK_DivideAndConquer(; kwargs...)
end
