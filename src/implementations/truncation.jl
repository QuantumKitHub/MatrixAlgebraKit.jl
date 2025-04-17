"""
    abstract type TruncationStrategy end

Supertype to denote different strategies for truncated decompositions that are implemented via post-truncation.

See also [`truncate!`](@ref)
"""
abstract type TruncationStrategy end

function TruncationStrategy(; atol=nothing, rtol=nothing, maxrank=nothing)
    if isnothing(maxrank) && isnothing(atol) && isnothing(rtol)
        return NoTruncation()
    elseif isnothing(maxrank)
        atol = @something atol 0
        rtol = @something rtol 0
        return TruncationKeepAbove(atol, rtol)
    else
        if isnothing(atol) && isnothing(rtol)
            return truncrank(maxrank)
        else
            atol = @something atol 0
            rtol = @something rtol 0
            return truncrank(maxrank) & TruncationKeepAbove(atol, rtol)
        end
    end
end

"""
    NoTruncation()

Trivial truncation strategy that keeps all values, mostly for testing purposes.
"""
struct NoTruncation <: TruncationStrategy end

# TODO: how do we deal with sorting/filters that treat zeros differently
# since these are implicitly discarded by selecting compact/full

"""
    TruncationKeepSorted(howmany::Int, sortby::Function, rev::Bool)

Truncation strategy to keep the first `howmany` values when sorted according to `sortby` or the last `howmany` if `rev` is true.
"""
struct TruncationKeepSorted{F} <: TruncationStrategy
    howmany::Int
    sortby::F
    rev::Bool
end

"""
    TruncationKeepFiltered(filter::Function)

Truncation strategy to keep the values for which `filter` returns true.
"""
struct TruncationKeepFiltered{F} <: TruncationStrategy
    filter::F
end

struct TruncationKeepAbove{T<:Real} <: TruncationStrategy
    atol::T
    rtol::T
end
TruncationKeepAbove(atol::Real, rtol::Real) = TruncationKeepAbove(promote(atol, rtol)...)

struct TruncationKeepBelow{T<:Real} <: TruncationStrategy
    atol::T
    rtol::T
end
TruncationKeepBelow(atol::Real, rtol::Real) = TruncationKeepBelow(promote(atol, rtol)...)

# TODO: better names for these functions of the above types
"""
    truncrank(howmany::Int, by=abs, rev=true)

Truncation strategy to keep the first `howmany` values when sorted according to `by` or the last `howmany` if `rev` is true.
"""
truncrank(howmany::Int, by=abs, rev=true) = TruncationKeepSorted(howmany, by, rev)

"""
    trunctol(atol::Real)

Truncation strategy to discard the values that are smaller than `atol` in absolute value.
"""
trunctol(atol) = TruncationKeepFiltered(≥(atol) ∘ abs)

"""
    truncabove(atol::Real)

Truncation strategy to discard the values that are larger than `atol` in absolute value.
"""
truncabove(atol) = TruncationKeepFiltered(≤(atol) ∘ abs)

"""
    TruncationIntersection(trunc1::TruncationStrategy, trunc2::TruncationStrategy)

Compose two truncation strategies, keeping values common between the two strategies.
"""
struct TruncationIntersection{T<:Tuple{Vararg{TruncationStrategy}}} <:
       TruncationStrategy
    components::T
end
function Base.:&(trunc1::TruncationStrategy, trunc2::TruncationStrategy)
    return TruncationIntersection((trunc1, trunc2))
end
function Base.:&(trunc1::TruncationIntersection, trunc2::TruncationIntersection)
    return TruncationIntersection((trunc1.components..., trunc2.components...))
end
function Base.:&(trunc1::TruncationIntersection, trunc2::TruncationStrategy)
    return TruncationIntersection((trunc1.components..., trunc2))
end
function Base.:&(trunc1::TruncationStrategy, trunc2::TruncationIntersection)
    return TruncationIntersection((trunc1, trunc2.components...))
end

# truncate!
# ---------
# Generic implementation: `findtruncated` followed by indexing
@doc """
    truncate!(f, out, strategy::TruncationStrategy)

Generic interface for post-truncating a decomposition, specified in `out`.
""" truncate!
# TODO: should we return a view?
function truncate!(::typeof(svd_trunc!), (U, S, Vᴴ), strategy::TruncationStrategy)
    ind = findtruncated(diagview(S), strategy)
    return U[:, ind], Diagonal(diagview(S)[ind]), Vᴴ[ind, :]
end
function truncate!(::typeof(eig_trunc!), (D, V), strategy::TruncationStrategy)
    ind = findtruncated(diagview(D), strategy)
    return Diagonal(diagview(D)[ind]), V[:, ind]
end
function truncate!(::typeof(eigh_trunc!), (D, V), strategy::TruncationStrategy)
    ind = findtruncated(diagview(D), strategy)
    return Diagonal(diagview(D)[ind]), V[:, ind]
end
function truncate!(::typeof(left_null!), (U, S), strategy::TruncationStrategy)
    # TODO: avoid allocation?
    extended_S = vcat(diagview(S), zeros(eltype(S), max(0, size(S, 1) - size(S, 2))))
    ind = findtruncated(extended_S, strategy)
    return U[:, ind]
end
function truncate!(::typeof(right_null!), (S, Vᴴ), strategy::TruncationStrategy)
    # TODO: avoid allocation?
    extended_S = vcat(diagview(S), zeros(eltype(S), max(0, size(S, 2) - size(S, 1))))
    ind = findtruncated(extended_S, strategy)
    return Vᴴ[ind, :]
end

# findtruncated
# -------------
# specific implementations for finding truncated values
findtruncated(values::AbstractVector, ::NoTruncation) = Colon()

# TODO: this may also permute the eigenvalues, decide if we want to allow this or not
# can be solved by going to simply sorting the resulting `ind`
function findtruncated(values::AbstractVector, strategy::TruncationKeepSorted)
    sorted = sortperm(values; by=strategy.sortby, rev=strategy.rev)
    howmany = min(strategy.howmany, length(sorted))
    ind = sorted[1:howmany]
    return ind # TODO: consider sort!(ind)
end

# TODO: consider if worth using that values are sorted when filter is `<` or `>`.
function findtruncated(values::AbstractVector, strategy::TruncationKeepFiltered)
    ind = findall(strategy.filter, values)
    return ind
end

function findtruncated(values::AbstractVector, strategy::TruncationKeepBelow)
    atol = max(strategy.atol, strategy.rtol * first(values))
    i = @something findfirst(≤(atol), values) length(values) + 1
    return i:length(values)
end
function findtruncated(values::AbstractVector, strategy::TruncationKeepAbove)
    atol = max(strategy.atol, strategy.rtol * first(values))
    i = @something findlast(≥(atol), values) 0
    return 1:i
end

function findtruncated(values::AbstractVector, strategy::TruncationIntersection)
    inds = map(Base.Fix1(findtruncated, values), strategy.components)
    return intersect(inds...)
end

"""
    TruncatedAlgorithm(alg::AbstractAlgorithm, trunc::TruncationAlgorithm)

Generic wrapper type for algorithms that consist of first using `alg`, followed by a
truncation through `trunc`.
"""
struct TruncatedAlgorithm{A,T} <: AbstractAlgorithm
    alg::A
    trunc::T
end
