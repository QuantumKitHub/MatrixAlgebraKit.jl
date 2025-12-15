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

"""
    svd_trunc(A; [trunc], kwargs...) -> U, S, Vᴴ, ϵ
    svd_trunc(A, alg::AbstractAlgorithm) -> U, S, Vᴴ, ϵ
    svd_trunc!(A, [USVᴴ]; [trunc], kwargs...) -> U, S, Vᴴ, ϵ
    svd_trunc!(A, [USVᴴ], alg::AbstractAlgorithm) -> U, S, Vᴴ, ϵ

Compute a partial or truncated singular value decomposition (SVD) of `A`, such that
`A * (Vᴴ)' ≈ U * S`. Here, `U` is an isometric matrix (orthonormal columns) of size
`(m, k)`, whereas  `Vᴴ` is a matrix of size `(k, n)` with orthonormal rows and `S` is a
square diagonal matrix of size `(k, k)`, with `k` is set by the truncation strategy.

The function also returns `ϵ`, the truncation error defined as the 2-norm of the 
discarded singular values.

## Truncation
The truncation strategy can be controlled via the `trunc` keyword argument. This can be
either a `NamedTuple` or a [`TruncationStrategy`](@ref). If `trunc` is not provided or
nothing, all values will be kept.

### `trunc::NamedTuple`
The supported truncation keyword arguments are:

$docs_truncation_kwargs

### `trunc::TruncationStrategy`
For more control, a truncation strategy can be supplied directly.
By default, MatrixAlgebraKit supplies the following:

$docs_truncation_strategies

## Keyword arguments
Other keyword arguments are passed to the algorithm selection procedure. If no explicit
`alg` is provided, these keywords are used to select and configure the algorithm through
[`MatrixAlgebraKit.select_algorithm`](@ref). The remaining keywords after algorithm
selection are passed to the algorithm constructor. See [`MatrixAlgebraKit.default_algorithm`](@ref)
for the default algorithm selection behavior.

When `alg` is a [`TruncatedAlgorithm`](@ref), the `trunc` keyword cannot be specified as the
truncation strategy is already embedded in the algorithm.

!!! note
    The bang method `svd_trunc!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `USVᴴ` as output.

See also [`svd_trunc_no_error(!)`](@ref svd_trunc), [`svd_full(!)`](@ref svd_full),
[`svd_compact(!)`](@ref svd_compact), [`svd_vals(!)`](@ref svd_vals),
and [Truncations](@ref) for more information on truncation strategies.
"""
@functiondef svd_trunc

"""
    svd_trunc_no_error(A; [trunc], kwargs...) -> U, S, Vᴴ
    svd_trunc_no_error(A, alg::AbstractAlgorithm) -> U, S, Vᴴ
    svd_trunc_no_error!(A, [USVᴴ]; [trunc], kwargs...) -> U, S, Vᴴ
    svd_trunc_no_error!(A, [USVᴴ], alg::AbstractAlgorithm) -> U, S, Vᴴ

Compute a partial or truncated singular value decomposition (SVD) of `A`, such that
`A * (Vᴴ)' ≈ U * S`. Here, `U` is an isometric matrix (orthonormal columns) of size
`(m, k)`, whereas  `Vᴴ` is a matrix of size `(k, n)` with orthonormal rows and `S` is a
square diagonal matrix of size `(k, k)`, with `k` is set by the truncation strategy.
The truncation error is *not* returned.

## Truncation
The truncation strategy can be controlled via the `trunc` keyword argument. This can be
either a `NamedTuple` or a [`TruncationStrategy`](@ref). If `trunc` is not provided or
nothing, all values will be kept.

### `trunc::NamedTuple`
The supported truncation keyword arguments are:

$docs_truncation_kwargs

### `trunc::TruncationStrategy`
For more control, a truncation strategy can be supplied directly.
By default, MatrixAlgebraKit supplies the following:

$docs_truncation_strategies

## Keyword arguments
Other keyword arguments are passed to the algorithm selection procedure. If no explicit
`alg` is provided, these keywords are used to select and configure the algorithm through
[`MatrixAlgebraKit.select_algorithm`](@ref). The remaining keywords after algorithm
selection are passed to the algorithm constructor. See [`MatrixAlgebraKit.default_algorithm`](@ref)
for the default algorithm selection behavior.

When `alg` is a [`TruncatedAlgorithm`](@ref), the `trunc` keyword cannot be specified as the
truncation strategy is already embedded in the algorithm.

!!! note
    The bang method `svd_trunc_no_error!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `USVᴴ` as output.

See also [`svd_full(!)`](@ref svd_full), [`svd_compact(!)`](@ref svd_compact),
[`svd_vals(!)`](@ref svd_vals), [`svd_trunc(!)`](@ref svd_trunc) and
[Truncations](@ref) for more information on truncation strategies.
"""
@functiondef svd_trunc_no_error

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
default_svd_algorithm(A; kwargs...) = default_svd_algorithm(typeof(A); kwargs...)
function default_svd_algorithm(T::Type; kwargs...)
    throw(MethodError(default_svd_algorithm, (T,)))
end
function default_svd_algorithm(::Type{T}; kwargs...) where {T <: YALAPACK.MaybeBlasMat}
    return LAPACK_DivideAndConquer(; kwargs...)
end
function default_svd_algorithm(::Type{T}; kwargs...) where {T <: Diagonal}
    return DiagonalAlgorithm(; kwargs...)
end

for f in (:svd_full!, :svd_compact!, :svd_vals!)
    @eval function default_algorithm(::typeof($f), ::Type{A}; kwargs...) where {A}
        return default_svd_algorithm(A; kwargs...)
    end
end

for f in (:svd_trunc!, :svd_trunc_no_error!)
    @eval function select_algorithm(::typeof($f), A, alg; trunc = nothing, kwargs...)
        if alg isa TruncatedAlgorithm
            isnothing(trunc) ||
                throw(ArgumentError("`trunc` can't be specified when `alg` is a `TruncatedAlgorithm`"))
            return alg
        else
            alg_svd = select_algorithm(svd_compact!, A, alg; kwargs...)
            return TruncatedAlgorithm(alg_svd, select_truncation(trunc))
        end
    end
end
