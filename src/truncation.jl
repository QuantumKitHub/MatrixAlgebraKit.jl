abstract type TruncationStrategy end

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

# TODO: better names for these functions of the above types
truncrank(howmany::Int, by=abs, rev=true) = TruncationKeepSorted(howmany, by, rev)
trunctol(atol) = TruncationKeepFiltered(>=(atol))

function findtruncated(values::AbstractVector, strategy::TruncationKeepSorted)
    sorted = sortperm(values; by=strategy.sortby, rev=strategy.rev)
    howmany = min(strategy.howmany, length(sorted))
    ind = sorted[1:howmany]
    return ind
end

function findtruncated(values::AbstractVector, strategy::TruncationKeepFiltered)
    ind = findall(strategy.filter, values)
    return ind
end
