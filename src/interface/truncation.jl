"""
    TruncationStrategy(; kwargs...)

Select a truncation strategy based on the provided keyword arguments.

## Keyword arguments
- `atol=nothing` : Absolute tolerance for the truncation
- `rtol=nothing` : Relative tolerance for the truncation
- `maxrank=nothing` : Maximal rank for the truncation
"""
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
See also [`notrunc()`](@ref).
"""
struct NoTruncation <: TruncationStrategy end

"""
    TruncationKeepSorted(howmany::Int, by::Function, rev::Bool)

Truncation strategy to keep the first `howmany` values when sorted according to `by` in increasing (decreasing) order if `rev` is false (true).
See also [`truncrank`](@ref).
"""
struct TruncationKeepSorted{F} <: TruncationStrategy
    howmany::Int
    by::F
    rev::Bool
end

"""
    truncrank(howmany::Int; by=abs, rev=true)

Truncation strategy to keep the first `howmany` values when sorted according to `by` or the last `howmany` if `rev` is true.
"""
truncrank(howmany::Int; by=abs, rev=true) = TruncationKeepSorted(howmany, by, rev)

"""
    TruncationKeepFiltered(filter::Function)

Truncation strategy to keep the values for which `filter` returns true.
"""
struct TruncationKeepFiltered{F} <: TruncationStrategy
    filter::F
end

"""
    trunctol(val::Real; by=abs)

Truncation strategy to discard the values that are smaller than `val` according to `by`.
"""
trunctol(val::Real; by=abs) = TruncationKeepFiltered(≥(val) ∘ by)

"""
    truncabove(val::Real; by=abs)

Truncation strategy to discard the values that are larger than `val` according to `by`.
"""
truncabove(val::Real; by=abs) = TruncationKeepFiltered(≤(val) ∘ by)

struct TruncationKeepAbove{T<:Real,P<:Real,F} <: TruncationStrategy
    atol::T
    rtol::T
    p::P
    by::F
end
function TruncationKeepAbove(; atol::Real, rtol::Real, p::Real=2, by=abs)
    return TruncationKeepAbove(atol, rtol, p, by)
end
function TruncationKeepAbove(atol::Real, rtol::Real, p::Real=2, by=abs)
    return TruncationKeepAbove(promote(atol, rtol)..., p, by)
end

"""
    TruncationKeepBelow(; atol::Real, rtol::Real, p=2, by=abs)

Truncation strategy to discard the values that are smaller than the norm of the values.
"""
struct TruncationKeepBelow{T<:Real,P<:Real,F} <: TruncationStrategy
    atol::T
    rtol::T
    p::P
    by::F
end
function TruncationKeepBelow(; atol::Real, rtol::Real, p::Real=2, by=abs)
    return TruncationKeepBelow(atol, rtol, p, by)
end
function TruncationKeepBelow(atol::Real, rtol::Real, p::Real=2, by=abs)
    return TruncationKeepBelow(promote(atol, rtol)..., p, by)
end

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
