# truncate
# --------
# Generic implementation: `findtruncated` followed by indexing
function truncate(::typeof(svd_trunc!), (U, S, Vᴴ), strategy::TruncationStrategy)
    ind = findtruncated_svd(diagview(S), strategy)
    return (U[:, ind], Diagonal(diagview(S)[ind]), Vᴴ[ind, :]), ind
end
function truncate(::typeof(eig_trunc!), (D, V), strategy::TruncationStrategy)
    ind = findtruncated(diagview(D), strategy)
    return (Diagonal(diagview(D)[ind]), V[:, ind]), ind
end
function truncate(::typeof(eigh_trunc!), (D, V), strategy::TruncationStrategy)
    ind = findtruncated(diagview(D), strategy)
    return (Diagonal(diagview(D)[ind]), V[:, ind]), ind
end
function truncate(::typeof(left_null!), (U, S), strategy::TruncationStrategy)
    # TODO: avoid allocation?
    extended_S = vcat(diagview(S), zeros(eltype(S), max(0, size(S, 1) - size(S, 2))))
    ind = findtruncated(extended_S, strategy)
    return U[:, ind], ind
end
function truncate(::typeof(right_null!), (S, Vᴴ), strategy::TruncationStrategy)
    # TODO: avoid allocation?
    extended_S = vcat(diagview(S), zeros(eltype(S), max(0, size(S, 2) - size(S, 1))))
    ind = findtruncated(extended_S, strategy)
    return Vᴴ[ind, :], ind
end

# special case `NoTruncation` for null: should keep exact zeros due to rectangularity
function truncate(::typeof(left_null!), (U, S), strategy::NoTruncation)
    m, n = size(S)
    ind = (n + 1):m
    return U[:, ind], ind
end
function truncate(::typeof(right_null!), (S, Vᴴ), strategy::NoTruncation)
    m, n = size(S)
    ind = (m + 1):n
    return Vᴴ[ind, :], ind
end


# findtruncated
# -------------
# Generic fallback
findtruncated_svd(values, strategy::TruncationStrategy) = findtruncated(values, strategy)

# specific implementations for finding truncated values
findtruncated(values::AbstractVector, ::NoTruncation) = Colon()

function findtruncated(values::AbstractVector, strategy::TruncationByOrder)
    howmany = min(strategy.howmany, length(values))
    return sortperm(values; strategy.by, strategy.rev)[1:howmany]
end
function findtruncated_svd(values::AbstractVector, strategy::TruncationByOrder)
    strategy.by === abs || return findtruncated(values, strategy)
    howmany = min(strategy.howmany, length(values))
    return strategy.rev ? (1:howmany) : ((length(values) - howmany + 1):length(values))
end

function findtruncated(values::AbstractVector, strategy::TruncationByFilter)
    return findall(strategy.filter, values)
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
        i = searchsortedfirst(values, atol; by = abs, rev = true)
        return i:length(values)
    else
        i = searchsortedlast(values, atol; by = abs, rev = true)
        return 1:i
    end
end

function findtruncated(values::AbstractVector, strategy::TruncationByError)
    I = sortperm(values; by = abs, rev = true)
    I′ = _truncerr_impl(values, I; strategy.atol, strategy.rtol, strategy.p)
    return I[I′]
end
function findtruncated_svd(values::AbstractVector, strategy::TruncationByError)
    I = eachindex(values)
    I′ = _truncerr_impl(values, I; strategy.atol, strategy.rtol, strategy.p)
    return I[I′]
end
function _truncerr_impl(values::AbstractVector, I; atol::Real = 0, rtol::Real = 0, p::Real = 2)
    by = Base.Fix2(^, p) ∘ abs
    Nᵖ = sum(by, values)
    ϵᵖ = max(atol^p, rtol^p * Nᵖ)

    # fast path to avoid checking all values
    ϵᵖ ≥ Nᵖ && return Base.OneTo(0)

    truncerrᵖ_array = cumsum(map(by, view(values, reverse(I))))
    rank = length(values) - (findfirst(≥(ϵᵖ), truncerrᵖ_array) - 1)
    return Base.OneTo(rank)
end

function findtruncated(values::AbstractVector, strategy::TruncationIntersection)
    length(strategy.components) == 0 && return eachindex(values)
    length(strategy.components) == 1 && return findtruncated(values, only(strategy.components))

    ind1 = findtruncated(values, strategy.components[1])
    ind2 = findtruncated(values, TruncationIntersection(Base.tail(strategy.components)))
    return _ind_intersect(ind1, ind2)
end
function findtruncated_svd(values::AbstractVector, strategy::TruncationIntersection)
    length(strategy.components) == 0 && return eachindex(values)
    length(strategy.components) == 1 && return findtruncated_svd(values, only(strategy.components))

    ind1 = findtruncated_svd(values, strategy.components[1])
    ind2 = findtruncated_svd(values, TruncationIntersection(Base.tail(strategy.components)))
    return _ind_intersect(ind1, ind2)
end

# when one of the ind selections is a bitvector, have to handle differently
function _ind_intersect(A::AbstractVector{Bool}, B::AbstractVector)
    result = falses(length(A))
    result[B] .= @view A[B]
    return result
end
_ind_intersect(A::AbstractVector, B::AbstractVector{Bool}) = _ind_intersect(B, A)
_ind_intersect(A::AbstractVector{Bool}, B::AbstractVector{Bool}) = A .& B

# when one of the ind selections is a unitrange, filter is more efficient than intersect
# since we know both selections only contain unique entries
# (This is also more GPU-friendly!)
_ind_intersect(A::AbstractVector{Int}, B::AbstractUnitRange{Int}) = filter(in(B), A)
_ind_intersect(A::AbstractUnitRange{Int}, B::AbstractVector{Int}) = _ind_intersect(B, A)

# when all else fails, call intersect
_ind_intersect(A, B) = intersect(A, B)

# Truncation error
# ----------------
truncation_error(values::AbstractVector, ind) = truncation_error!(copy(values), ind)
# destroys input in order to maximize accuracy:
# sqrt(norm(values)^2 - norm(values[ind])^2) might suffer from floating point error
function truncation_error!(values::AbstractVector, ind)
    values[ind] .= zero(eltype(values))
    return norm(values)
end
