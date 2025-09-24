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
function findtruncated_svd(values::AbstractVector, strategy::TruncationByOrder)
    if strategy.by === abs
        howmany = min(strategy.howmany, length(values))
        return strategy.rev ? (1:howmany) : ((length(values) - howmany + 1):length(values))
    else
        return findtruncated(values, strategy)
    end
end

function findtruncated(values::AbstractVector, strategy::TruncationByFilter)
    # pre-allocate bitvector to enforce the filter function returns a Bool
    mask = similar(BitArray, eachindex(values))
    mask .= strategy.filter.(values)
    return mask
end

function findtruncated(values::AbstractVector, strategy::TruncationByValue)
    atol = max(strategy.atol, strategy.rtol * norm(values, strategy.p))
    filter = (strategy.keep_below ? ≤(atol) : ≥(atol)) ∘ strategy.by
    return findtruncated(values, truncfilter(filter))
end
function findtruncated_svd(values::AbstractVector, strategy::TruncationByValue)
    strategy.by === abs || return findtruncated(values, strategy)

    atol = max(strategy.atol, strategy.rtol * norm(values, strategy.p))
    if strategy.keep_below
        i = searchsortedfirst(values, atol; by=abs, rev=true)
        return i:length(values)
    else
        i = searchsortedlast(values, atol; by=abs, rev=true)
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
    return mapreduce(Base.Fix1(findtruncated, values), _ind_intersect, strategy.components;
                     init=trues(length(values)))
end
function findtruncated_sorted(values::AbstractVector, strategy::TruncationIntersection)
    return mapreduce(Base.Fix1(findtruncated_sorted, values), _ind_intersect,
                     strategy.components; init=trues(length(values)))
end

# when one of the ind selections is a bitvector, have to handle differently
function _ind_intersect(A::AbstractVector{Bool}, B::AbstractVector)
    result = falses(length(A))
    result[B] .= @view A[B]
    return result
end
_ind_intersect(A::AbstractVector, B::AbstractVector{Bool}) = _ind_intersect(B, A)
_ind_intersect(A::AbstractVector{Bool}, B::AbstractVector{Bool}) = A .& B
_ind_intersect(A, B) = intersect(A, B)
