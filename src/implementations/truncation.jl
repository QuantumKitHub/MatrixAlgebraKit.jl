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

function select_truncation(trunc)
    if isnothing(trunc)
        return NoTruncation()
    elseif trunc isa NamedTuple
        return TruncationStrategy(; trunc...)
    elseif trunc isa TruncationStrategy
        return trunc
    else
        return throw(ArgumentError("Unknown truncation strategy: $trunc"))
    end
end

# TODO: how do we deal with sorting/filters that treat zeros differently
# since these are implicitly discarded by selecting compact/full

"""
    TruncationKeepSorted(howmany::Int, by::Function, rev::Bool)

Truncation strategy to keep the first `howmany` values when sorted according to `by` in increasing (decreasing) order if `rev` is false (true).
"""
struct TruncationKeepSorted{F} <: TruncationStrategy
    howmany::Int
    by::F
    rev::Bool
end

"""
    TruncationKeepFiltered(filter::Function)

Truncation strategy to keep the values for which `filter` returns true.
"""
struct TruncationKeepFiltered{F} <: TruncationStrategy
    filter::F
end

struct TruncationKeepAbove{T<:Real,F} <: TruncationStrategy
    atol::T
    rtol::T
    p::Int
    by::F
end
function TruncationKeepAbove(; atol::Real, rtol::Real, p::Int=2, by=abs)
    return TruncationKeepAbove(atol, rtol, p, by)
end
function TruncationKeepAbove(atol::Real, rtol::Real, p::Int=2, by=abs)
    return TruncationKeepAbove(promote(atol, rtol)..., p, by)
end

struct TruncationKeepBelow{T<:Real,F} <: TruncationStrategy
    atol::T
    rtol::T
    p::Int
    by::F
end
function TruncationKeepBelow(; atol::Real, rtol::Real, p::Int=2, by=abs)
    return TruncationKeepBelow(atol, rtol, p, by)
end
function TruncationKeepBelow(atol::Real, rtol::Real, p::Int=2, by=abs)
    return TruncationKeepBelow(promote(atol, rtol)..., p, by)
end

# TODO: better names for these functions of the above types
"""
    truncrank(howmany::Int; by=abs, rev=true)

Truncation strategy to keep the first `howmany` values when sorted according to `by` or the last `howmany` if `rev` is true.
"""
truncrank(howmany::Int; by=abs, rev=true) = TruncationKeepSorted(howmany, by, rev)

"""
    trunctol(atol::Real; by=abs)

Truncation strategy to discard the values that are smaller than `atol` according to `by`.
"""
trunctol(atol; by=abs) = TruncationKeepFiltered(≥(atol) ∘ by)

"""
    truncabove(atol::Real; by=abs)

Truncation strategy to discard the values that are larger than `atol` according to `by`.
"""
truncabove(atol; by=abs) = TruncationKeepFiltered(≤(atol) ∘ by)

"""
    TruncationIntersection(trunc::TruncationStrategy, truncs::TruncationStrategy...)

Composition of multiple truncation strategies, keeping values common between them.
"""
struct TruncationIntersection{T<:Tuple{Vararg{TruncationStrategy}}} <:
       TruncationStrategy
    components::T
end
function TruncationIntersection(trunc::TruncationStrategy, truncs::TruncationStrategy...)
    return TruncationIntersection((trunc, truncs...))
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
    ind = findtruncated_sorted(diagview(S), strategy)
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
@doc """
    MatrixAlgebraKit.findtruncated(values::AbstractVector, strategy::TruncationStrategy)

Generic interface for finding truncated values of the spectrum of a decomposition
based on the `strategy`. The output should be a collection of indices specifying
which values to keep. `MatrixAlgebraKit.findtruncated` is used inside of the default
implementation of [`truncate!`](@ref) to perform the truncation. It does not assume that the
values are sorted. For a version that assumes the values are reverse sorted (which is the
standard case for SVD) see [`MatrixAlgebraKit.findtruncated_sorted`](@ref).
""" findtruncated

@doc """
    MatrixAlgebraKit.findtruncated_sorted(values::AbstractVector, strategy::TruncationStrategy)

Like [`MatrixAlgebraKit.findtruncated`](@ref) but assumes that the values are sorted in reverse order.
They are assumed to be sorted in a way that is consistent with the truncation strategy,
which generally means they are sorted by absolute value but some truncation strategies allow
customizing that. However, note that this assumption is not checked, so passing values that are not sorted
in the correct way can silently give unexpected results. This is used in the default implementation of
[`svd_trunc!`](@ref).
""" findtruncated_sorted

findtruncated(values::AbstractVector, ::NoTruncation) = Colon()

# TODO: this may also permute the eigenvalues, decide if we want to allow this or not
# can be solved by going to simply sorting the resulting `ind`
function findtruncated(values::AbstractVector, strategy::TruncationKeepSorted)
    howmany = min(strategy.howmany, length(values))
    return partialsortperm(values, 1:howmany; by=strategy.by, rev=strategy.rev)
end
function findtruncated_sorted(values::AbstractVector, strategy::TruncationKeepSorted)
    howmany = min(strategy.howmany, length(values))
    return 1:howmany
end

# TODO: consider if worth using that values are sorted when filter is `<` or `>`.
function findtruncated(values::AbstractVector, strategy::TruncationKeepFiltered)
    ind = findall(strategy.filter, values)
    return ind
end

function findtruncated(values::AbstractVector, strategy::TruncationKeepBelow)
    atol = max(strategy.atol, strategy.rtol * norm(values, strategy.p))
    return findall(≤(atol) ∘ strategy.by, values)
end
function findtruncated_sorted(values::AbstractVector, strategy::TruncationKeepBelow)
    atol = max(strategy.atol, strategy.rtol * norm(values, strategy.p))
    i = searchsortedfirst(values, atol; by=strategy.by, rev=true)
    return i:length(values)
end

function findtruncated(values::AbstractVector, strategy::TruncationKeepAbove)
    atol = max(strategy.atol, strategy.rtol * norm(values, strategy.p))
    return findall(≥(atol) ∘ strategy.by, values)
end
function findtruncated_sorted(values::AbstractVector, strategy::TruncationKeepAbove)
    atol = max(strategy.atol, strategy.rtol * norm(values, strategy.p))
    i = searchsortedlast(values, atol; by=strategy.by, rev=true)
    return 1:i
end

function findtruncated(values::AbstractVector, strategy::TruncationIntersection)
    inds = map(Base.Fix1(findtruncated, values), strategy.components)
    return intersect(inds...)
end

# Generic fallback.
function findtruncated_sorted(values::AbstractVector, strategy::TruncationStrategy)
    return findtruncated(values, strategy)
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
