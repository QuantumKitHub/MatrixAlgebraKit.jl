"""
    TruncationStrategy(; kwargs...)

Select a truncation strategy based on the provided keyword arguments.

## Keyword arguments
The following keyword arguments are all optional, and their default value (`nothing`)
will be ignored. It is also allowed to combine multiple of these, in which case the kept
values will consist of the intersection of the different truncated strategies.

- `atol::Real` : Absolute tolerance for the truncation
- `rtol::Real` : Relative tolerance for the truncation
- `maxrank::Real` : Maximal rank for the truncation
- `maxerror::Real` : Maximal truncation error.
- `filter` : Custom filter to select truncated values.
"""
function TruncationStrategy(;
        atol::Union{Real, Nothing} = nothing,
        rtol::Union{Real, Nothing} = nothing,
        maxrank::Union{Real, Nothing} = nothing,
        maxerror::Union{Real, Nothing} = nothing,
        filter = nothing
    )
    strategy = notrunc()

    if !isnothing(atol) || !isnothing(rtol)
        atol = @something atol 0
        rtol = @something rtol 0
        strategy &= trunctol(; atol, rtol)
    end

    isnothing(maxrank) || (strategy &= truncrank(maxrank))
    isnothing(maxerror) || (strategy &= truncerror(maxerror))
    isnothing(filter) || (strategy &= truncfilter(filter))

    return strategy
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

# TODO: Base.Ordering?
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
function truncrank(howmany::Integer; by = abs, rev::Bool = true)
    return TruncationByOrder(howmany, by, rev)
end

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
    TruncationByValue(atol::Real, rtol::Real, p::Real, by, keep_below::Bool=false)

Truncation strategy to keep the values that satisfy `by(val) > max(atol, rtol * norm(values, p)`.
If `keep_below = true`, discard these values instead.
See also [`trunctol`](@ref)
"""
struct TruncationByValue{T <: Real, P <: Real, F} <: TruncationStrategy
    atol::T
    rtol::T
    p::P
    by::F
    keep_below::Bool
end
function TruncationByValue(atol::Real, rtol::Real, p::Real = 2, by = abs, keep_below::Bool = true)
    return TruncationByValue(promote(atol, rtol)..., p, by, keep_below)
end

"""
    trunctol(; atol::Real=0, rtol::Real=0, p::Real=2, by=abs, keep_below::Bool=false)

Truncation strategy to keep the values that satisfy `by(val) > max(atol, rtol * norm(values, p)`.
If `keep_below = true`, discard these values instead.
"""
function trunctol(; atol::Real = 0, rtol::Real = 0, p::Real = 2, by = abs, keep_below::Bool = false)
    return TruncationByValue(atol, rtol, p, by, keep_below)
end

"""
    TruncationByError(; atol::Real, rtol::Real, p::Real)

Truncation strategy to discard values until the error caused by the discarded values exceeds some tolerances.
See also [`truncerror`](@ref).
"""
struct TruncationByError{T <: Real, P <: Real} <: TruncationStrategy
    atol::T
    rtol::T
    p::P
end
function TruncationError(atol::Real, rtol::Real, p::Real = 2)
    return TruncationError(promote(atol, rtol)..., p)
end

"""
    truncerror(; atol::Real=0, rtol::Real=0, p::Real=2)

Truncation strategy for truncating values such that the error in the factorization
is smaller than `max(atol, rtol * norm)`, where the error is determined using the `p`-norm.
"""
function truncerror(; atol::Real = 0, rtol::Real = 0, p::Real = 2)
    return TruncationByError(promote(atol, rtol)..., p)
end

"""
    TruncationIntersection(trunc::TruncationStrategy, truncs::TruncationStrategy...)

Truncation strategy that composes multiple truncation strategies, keeping values that are
common between them.
"""
struct TruncationIntersection{T <: Tuple{Vararg{TruncationStrategy}}} <: TruncationStrategy
    components::T
end
function TruncationIntersection(trunc::TruncationStrategy, truncs::TruncationStrategy...)
    return TruncationIntersection((trunc, truncs...))
end

function Base.:&(trunc1::TruncationStrategy, trunc2::TruncationStrategy)
    return TruncationIntersection((trunc1, trunc2))
end

# flatten components
function Base.:&(trunc1::TruncationIntersection, trunc2::TruncationIntersection)
    return TruncationIntersection((trunc1.components..., trunc2.components...))
end
function Base.:&(trunc1::TruncationIntersection, trunc2::TruncationStrategy)
    return TruncationIntersection((trunc1.components..., trunc2))
end
function Base.:&(trunc1::TruncationStrategy, trunc2::TruncationIntersection)
    return TruncationIntersection((trunc1, trunc2.components...))
end

# drop notrunc
Base.:&(::NoTruncation, trunc::TruncationStrategy) = trunc
Base.:&(trunc::TruncationStrategy, ::NoTruncation) = trunc
Base.:&(::NoTruncation, ::NoTruncation) = notrunc()

# disambiguate
Base.:&(::NoTruncation, trunc::TruncationIntersection) = trunc
Base.:&(trunc::TruncationIntersection, ::NoTruncation) = trunc
