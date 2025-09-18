# Utility combination
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
findtruncated(values::AbstractVector, ::NoTruncation) = Colon()

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
function findtruncated_sorted(values::AbstractVector, strategy::TruncationIntersection)
    inds = map(Base.Fix1(findtruncated_sorted, values), strategy.components)
    return intersect(inds...)
end

# Generic fallback.
function findtruncated_sorted(values::AbstractVector, strategy::TruncationStrategy)
    return findtruncated(values, strategy)
end

