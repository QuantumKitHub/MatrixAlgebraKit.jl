"""
    abstract type AbstractAlgorithm end

Supertype to dispatch on specific implementations of different the different functions.
Concrete subtypes should represent both a way to dispatch to a given implementation, as
well as the configuration of that implementation.

See also [`select_algorithm`](@ref).
"""
abstract type AbstractAlgorithm end

"""
    Algorithm{name,KW} <: AbstractAlgorithm

Bare-bones implementation of an algorithm, where `name` should be a `Symbol` to dispatch on,
and `KW` is typically a `NamedTuple` indicating the keyword arguments.

See also [`@algdef`](@ref).
"""
struct Algorithm{name,K} <: AbstractAlgorithm
    kwargs::K
end
name(alg::Algorithm) = name(typeof(alg))
name(::Type{<:Algorithm{N}}) where {N} = N

# TODO: do we want to restrict this to Algorithm{name,<:NamedTuple}?
# Pretend like kwargs are part of the properties of the algorithm
Base.propertynames(alg::Algorithm) = (:kwargs, propertynames(getfield(alg, :kwargs))...)
@inline function Base.getproperty(alg::Algorithm, f::Symbol)
    kwargs = getfield(alg, :kwargs)
    return f === :kwargs ? kwargs : getproperty(kwargs, f)
end

# TODO: do we want to simply define this for all `Algorithm{N,<:NamedTuple}`?
# need print to make strings/symbols parseable,
# show to make objects parseable
function _show_alg(io::IO, alg::Algorithm)
    print(io, name(alg))
    print(io, "(")
    properties = filter(!=(:kwargs), propertynames(alg))
    next = iterate(properties)
    isnothing(next) && return print(io, ")")
    f, state = next
    print(io, "; ", f, "=")
    show(io, getproperty(alg, f))
    next = iterate(properties, state)
    while !isnothing(next)
        f, state = next
        print(io, ", ", f, "=")
        show(io, getproperty(alg, f))
        next = iterate(properties, state)
    end
    return print(io, ")")
end

@doc """
    select_algorithm(f, A; kwargs...)

Given some keyword arguments and an input `A`, decide on an algrithm to use for
implementing the function `f` on inputs of type `A`.
"""
function select_algorithm end

@doc """
    copy_input(f, A)

Preprocess the input `A` for a given function, such that it may be handled correctly later.
This may include a copy whenever the implementation would destroy the original matrix,
or a change of element type to something that is supported.
"""
function copy_input end

@doc """
    initialize_output(f, A, alg)

Whenever possible, allocate the destination for applying a given algorithm in-place.
If this is not possible, for example when the output size is not known a priori or immutable,
this function may return `nothing`.
"""
function initialize_output end

# Utility macros
# --------------

"""
    @algdef AlgorithmName

Convenience macro to define an algorithm `AlgorithmName` that accepts generic keywords.
This defines an exported alias for [`Algorithm{:AlgorithmName}`](@ref Algorithm)
along with some utility methods.
"""
macro algdef(name)
    esc(quote
            const $name{K} = Algorithm{$(QuoteNode(name)),K}
            function $name(; kwargs...)
                # TODO: is this necessary/useful?
                kw = NamedTuple(kwargs) # normalize type
                return $name{typeof(kw)}(kw)
            end
            function Base.show(io::IO, alg::$name)
                return _show_alg(io, alg)
            end

            Core.@__doc__ $name
        end)
end

"""
    @functiondef f

Convenience macro to define the boilerplate code that dispatches between several versions of `f` and `f!`.
By default, this enables the following signatures to be defined in terms of
the final `f!(A, out, alg::Algorithm)`.

```julia
    f(A; kwargs...)
    f(A, alg::Algorithm)
    f!(A, [out]; kwargs...)
    f!(A, alg::Algorithm)
```

See also [`copy_input`](@ref), [`select_algorithm`](@ref) and [`initialize_output`](@ref).
"""
macro functiondef(f)
    f isa Symbol || throw(ArgumentError("Unsupported usage of `@functiondef`"))
    f! = Symbol(f, :!)

    return esc(quote
                   # out of place to inplace
                   $f(A; kwargs...) = $f!(copy_input($f, A); kwargs...)
                   $f(A, alg::AbstractAlgorithm) = $f!(copy_input($f, A), alg)

                   # fill in arguments
                   $f!(A; kwargs...) = $f!(A, select_algorithm($f!, A; kwargs...))
                   function $f!(A, out; kwargs...)
                       return $f!(A, out, select_algorithm($f!, A; kwargs...))
                   end
                   function $f!(A, alg::AbstractAlgorithm)
                       return $f!(A, initialize_output($f!, A, alg), alg)
                   end

                   # copy documentation to both functions
                   Core.@__doc__ $f, $f!
               end)
end

"""
    @check_scalar(x, y, [op], [eltype])

Check if `eltype(x) == op(eltype(y))` and throw an error if not.
By default `op = identity` and `eltype = eltype'.
"""
macro check_scalar(x, y, op=:identity, eltype=:eltype)
    error_message = "Unexpected scalar type: "
    error_message *= string(eltype) * "(" * string(x) * ")"
    if op == :identity
        error_message *= " != " * string(eltype) * "(" * string(y) * ")"
    else
        error_message *= " != " * string(op) * "(" * string(eltype) * "(" * string(y) * "))"
    end
    return esc(quote
                   $eltype($x) == $op($eltype($y)) || throw(ArgumentError($error_message))
               end)
end

"""
    @check_size(x, sz, [size])

Check if `size(x) == sz` and throw an error if not.
By default, `size = size`.
"""
macro check_size(x, sz, size=:size)
    msgstart = string(size) * "(" * string(x) * ") = "
    err = gensym()
    return esc(quote
                   szx = $size($x)
                   $err = $msgstart * string(szx) * " instead of expected value " *
                          string($sz)
                   szx == $sz || throw(DimensionMismatch($err))
               end)
end