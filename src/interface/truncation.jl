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
        return trunctol(; atol, rtol)
    else
        if isnothing(atol) && isnothing(rtol)
            return truncrank(maxrank)
        else
            atol = @something atol 0
            rtol = @something rtol 0
            return truncrank(maxrank) & trunctol(; atol, rtol)
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
    notrunc()

Truncation strategy that does nothing, and keeps all the values.
"""
notrunc() = NoTruncation()

"""
    TruncationByOrder(howmany::Int, by::Function, rev::Bool)

Truncation strategy to keep the first `howmany` values when sorted according to `by` in increasing (decreasing) order if `rev` is false (true).

See also [`truncrank`](@ref).
"""
struct TruncationByOrder{F} <: TruncationStrategy
    howmany::Int
    by::F
    rev::Bool
end

"""
    truncrank(howmany::Integer; by=abs, rev::Bool=true)

Truncation strategy to keep the first `howmany` values when sorted according to `by` or the last `howmany` if `rev` is true.
"""
truncrank(howmany::Integer; by=abs, rev::Bool=true) = TruncationByOrder(howmany, by, rev)

"""
    TruncationByFilter(filter::Function)

Truncation strategy to keep the values for which `filter` returns true.

See also [`truncfilter`](@ref).
"""
struct TruncationByFilter{F} <: TruncationStrategy
    filter::F
end

"""
    truncfilter(filter)

Truncation strategy to keep the values for which `filter` returns true.
"""
truncfilter(f) = TruncationByFilter(f)

"""
    TruncationByValue(atol::Real, rtol::Real, p::Real, by, rev::Bool=true)

Truncation strategy to keep the values that satisfy `by(val) < max(atol, rtol * norm(values, p)`
if `rev = true`, or discard them when `rev = false`.
See also [`trunctol`](@ref)
"""
struct TruncationByValue{T<:Real,P<:Real,F} <: TruncationStrategy
    atol::T
    rtol::T
    p::P
    by::F
    rev::Bool
end
function TruncationByValue(atol::Real, rtol::Real, p::Real=2, by=abs, rev::Bool=true)
    return TruncationByValue(promote(atol, rtol)..., p, by, rev)
end

"""
    trunctol(; atol::Real=0, rtol::Real=0, p::Real=2, by=abs, )

Truncation strategy to keep the values that satisfy `by(val) < max(atol, rtol * norm(values, p)`
if `rev = true`, or discard them when `rev = false`.
"""
function trunctol(; atol::Real=0, rtol::Real=0, p::Real=2, by=abs, rev::Bool=true)
    return TruncationByValue(atol, rtol, p, by, rev)
end

"""
    TruncationByError(; atol::Real, rtol::Real, p::Real)

Truncation strategy to discard values until the error caused by the discarded values exceeds some tolerances.
See also [`truncerror`](@ref).
"""
struct TruncationByError{T<:Real,P<:Real} <: TruncationStrategy
    atol::T
    rtol::T
    p::P
end
function TruncationError(atol::Real, rtol::Real, p::Real=2)
    return TruncationError(promote(atol, rtol)..., p)
end

"""
    truncerror(; atol::Real=0, rtol::Real=0, p::Real=2)

Create a truncation strategy for truncating such that the error in the factorization
is smaller than `max(atol, rtol * norm)`, where the error is determined using the `p`-norm.
"""
function truncerror(; atol::Real=0, rtol::Real=0, p::Real=2)
    return TruncationByError(promote(atol, rtol)..., p)
end

"""
    TruncationIntersection(trunc::TruncationStrategy, truncs::TruncationStrategy...)

Composition of multiple truncation strategies, keeping values common between them.
"""
struct TruncationIntersection{T<:Tuple{Vararg{TruncationStrategy}}} <: TruncationStrategy
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
