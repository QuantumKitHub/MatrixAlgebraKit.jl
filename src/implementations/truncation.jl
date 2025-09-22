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
# Generic fallback
function findtruncated_sorted(values::AbstractVector, strategy::TruncationStrategy)
    return findtruncated(values, strategy)
end

# specific implementations for finding truncated values
findtruncated(values::AbstractVector, ::NoTruncation) = Colon()

function findtruncated(values::AbstractVector, strategy::TruncationByOrder)
    howmany = min(strategy.howmany, length(values))
    return partialsortperm(values, 1:howmany; strategy.by, strategy.rev)
end
function findtruncated_sorted(values::AbstractVector, strategy::TruncationByOrder)
    howmany = min(strategy.howmany, length(values))
    return strategy.rev ? (1:howmany) : ((length(values) - howmany + 1):length(values))
end

function findtruncated(values::AbstractVector, strategy::TruncationByFilter)
    ind = findall(strategy.filter, values)
    return ind
end

function findtruncated(values::AbstractVector, strategy::TruncationByValue)
    atol = max(strategy.atol, strategy.rtol * norm(values, strategy.p))
    filter = (strategy.rev ? ≤(atol) : ≥(atol)) ∘ strategy.by
    return findall(filter, values)
end
function findtruncated_sorted(values::AbstractVector, strategy::TruncationByValue)
    atol = max(strategy.atol, strategy.rtol * norm(values, strategy.p))
    @assert strategy.by === abs || strategy.by === real "sorting strategy incompatible with implementation"
    if strategy.rev
        i = searchsortedfirst(values, atol; by=strategy.by, rev=true)
        return i:length(values)
    else
        i = searchsortedlast(values, atol; by=strategy.by, rev=true)
        return 1:i
    end
end

function findtruncated(values::AbstractVector, strategy::TruncationByError)
    I = sortperm(values; by=abs, rev=true)
    I′ = _truncerr_impl(values, I; strategy.atol, strategy.rtol, strategy.p)
    return I[I′]
end
function findtruncated_sorted(values::AbstractVector, strategy::TruncationByError)
    I = eachindex(values)
    I′ = _truncerr_impl(values, I; strategy.atol, strategy.rtol, strategy.p)
    return I[I′]
end
function _truncerr_impl(values::AbstractVector, I; atol::Real=0, rtol::Real=0, p::Real=2)
    by = Base.Fix2(^, p) ∘ abs
    Nᵖ = sum(by, values)
    ϵᵖ = max(atol^p, rtol^p * Nᵖ)

    # fast path to avoid checking all values
    ϵᵖ ≥ Nᵖ && return Base.OneTo(0)

    truncerrᵖ = zero(real(eltype(values)))
    rank = length(values)
    for i in reverse(I)
        truncerrᵖ += by(values[i])
        truncerrᵖ ≥ ϵᵖ && break
        rank -= 1
    end

    return Base.OneTo(rank)
end

function findtruncated(values::AbstractVector, strategy::TruncationIntersection)
    inds = map(Base.Fix1(findtruncated, values), strategy.components)
    return intersect(inds...)
end
function findtruncated_sorted(values::AbstractVector, strategy::TruncationIntersection)
    inds = map(Base.Fix1(findtruncated_sorted, values), strategy.components)
    return intersect(inds...)
end
