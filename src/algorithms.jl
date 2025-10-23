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
struct Algorithm{name, K} <: AbstractAlgorithm
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

# Algorithm traits
# ----------------

"""
    left_null_kind(alg::AbstractAlgorithm) -> f!

Select an appropriate factorization function for applying `left_null!(A, alg)`.
By default, this is either `left_null_qr!` or `left_null_svd!`, but this can be extended
to insert arbitrary other decomposition functions, which should follow the signature
`f!(A, F, alg) -> F`
"""
function left_null_kind(alg::AbstractAlgorithm)
    left_orth_kind(alg) === left_orth_qr! && return left_null_qr!
    left_orth_kind(alg) === left_orth_svd! && return left_null_svd!
    return error(
        """
        Unkown or invalid `left_null` algorithm type `$(typeof(alg))`.
        To register the algorithm type, define:

                MatrixAlgebraKit.left_null_kind(alg) = f!

        where `f!` should be the factorization function that will be used.
        By default, this is either `left_null_qr!` or `left_null_svd!`.
        """
    )
end

"""
    right_null_kind(alg::AbstractAlgorithm) -> f!

Select an appropriate factorization function for applying `right_null!(A, alg)`.
By default, this is either `right_null_lq!` or `right_null_svd!`, but this can be extended
to insert arbitrary other decomposition functions, which should follow the signature
`f!(A, F, alg) -> F`
"""
function right_null_kind(alg::AbstractAlgorithm)
    right_orth_kind(alg) === right_orth_lq! && return right_null_lq!
    right_orth_kind(alg) === right_orth_svd! && return right_null_svd!
    return error(
        """
        Unkown or invalid `right_null` algorithm type `$(typeof(alg))`.
        To register the algorithm type, define:

                MatrixAlgebraKit.right_null_kind(alg) = f!

        where `f!` should be the factorization function that will be used.
        By default, this is either `right_null_lq!` or `right_null_svd!`.
        """
    )
end

"""
    does_truncate(alg::AbstractAlgorithm) -> Bool

Check whether or not an algorithm can be used for a truncated decomposition.
"""
does_truncate(alg::AbstractAlgorithm) = false

# Algorithm selection
# -------------------
@doc """
    MatrixAlgebraKit.select_algorithm(f, A, alg::AbstractAlgorithm)
    MatrixAlgebraKit.select_algorithm(f, A, alg::Symbol; kwargs...)
    MatrixAlgebraKit.select_algorithm(f, A, alg::Type; kwargs...)
    MatrixAlgebraKit.select_algorithm(f, A; kwargs...)
    MatrixAlgebraKit.select_algorithm(f, A, (; kwargs...))

Decide on an algorithm to use for implementing the function `f` on inputs of type `A`.
This can be obtained both for values `A` or types `A`.

If `alg` is an `AbstractAlgorithm` instance, it will be returned as-is.

If `alg` is a `Symbol` or a `Type` of algorithm, the return value is obtained
by calling the corresponding algorithm constructor;
keyword arguments in `kwargs` are passed along  to this constructor.

If `alg` is not specified (or `nothing`), an algorithm will be selected 
automatically with [`MatrixAlgebraKit.default_algorithm`](@ref) and 
the keyword arguments in `kwargs` will be passed to the algorithm constructor.
Finally, the same behavior is obtained when the keyword arguments are
passed as the third positional argument in the form of a `NamedTuple`. 
""" select_algorithm

function select_algorithm(f::F, A, alg::Alg = nothing; kwargs...) where {F, Alg}
    if isnothing(alg)
        return default_algorithm(f, A; kwargs...)
    elseif alg isa Symbol
        return Algorithm{alg}(; kwargs...)
    elseif alg isa Type
        return alg(; kwargs...)
    elseif alg isa NamedTuple
        isempty(kwargs) ||
            throw(ArgumentError("Additional keyword arguments are not allowed when algorithm parameters are specified."))
        return default_algorithm(f, A; alg...)
    elseif alg isa AbstractAlgorithm
        isempty(kwargs) ||
            throw(ArgumentError("Additional keyword arguments are not allowed when algorithm parameters are specified."))
        return alg
    end

    throw(ArgumentError("Unknown alg $alg"))
end

@doc """
    MatrixAlgebraKit.default_algorithm(f, A; kwargs...)
    MatrixAlgebraKit.default_algorithm(f, ::Type{TA}; kwargs...) where {TA}

Select the default algorithm for a given factorization function `f` and input `A`.
In general, this is called by [`select_algorithm`](@ref) if no algorithm is specified
explicitly.
New types should prefer to register their default algorithms in the type domain.
""" default_algorithm
default_algorithm(f::F, A; kwargs...) where {F} = default_algorithm(f, typeof(A); kwargs...)
default_algorithm(f::F, A, B; kwargs...) where {F} = default_algorithm(f, typeof(A), typeof(B); kwargs...)
# avoid infinite recursion:
function default_algorithm(f::F, ::Type{T}; kwargs...) where {F, T}
    throw(MethodError(default_algorithm, (f, T)))
end
function default_algorithm(f::F, ::Type{TA}, ::Type{TB}; kwargs...) where {F, TA, TB}
    throw(MethodError(default_algorithm, (f, TA, TB)))
end

@doc """
    copy_input(f, A)

Preprocess the input `A` for a given function, such that it may be handled correctly later.
This may include a copy whenever the implementation would destroy the original matrix,
or a change of element type to something that is supported.
""" copy_input

@doc """
    initialize_output(f, A, alg)

Whenever possible, allocate the destination for applying a given algorithm in-place.
If this is not possible, for example when the output size is not known a priori or immutable,
this function may return `nothing`.
""" initialize_output

# Truncation strategy
# -------------------
"""
    abstract type TruncationStrategy end

Supertype to denote different strategies for truncated decompositions that are implemented via post-truncation.

See also [`truncate`](@ref)
"""
abstract type TruncationStrategy end

@doc """
    MatrixAlgebraKit.select_truncation(trunc)

Construct a [`TruncationStrategy`](@ref) from the given `NamedTuple` of keywords or input strategy.
""" select_truncation

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

@doc """
    MatrixAlgebraKit.select_null_truncation(trunc)

Construct a [`TruncationStrategy`](@ref) from the given `NamedTuple` of keywords or input strategy, to implement a nullspace selection.
""" select_null_truncation

function select_null_truncation(trunc)
    if isnothing(trunc)
        return NoTruncation()
    elseif trunc isa NamedTuple
        return null_truncation_strategy(; trunc...)
    elseif trunc isa TruncationStrategy
        return trunc
    else
        return throw(ArgumentError("Unknown truncation strategy: $trunc"))
    end
end

@doc """
    MatrixAlgebraKit.findtruncated(values::AbstractVector, strategy::TruncationStrategy)

Generic interface for finding truncated values of the spectrum of a decomposition
based on the `strategy`. The output should be a collection of indices specifying
which values to keep. `MatrixAlgebraKit.findtruncated` is used inside of the default
implementation of [`truncate`](@ref) to perform the truncation. It does not assume that the
values are sorted. For a version that assumes the values are reverse sorted (which is the
standard case for SVD) see [`MatrixAlgebraKit.findtruncated_svd`](@ref).
""" findtruncated

@doc """
    MatrixAlgebraKit.findtruncated_svd(values::AbstractVector, strategy::TruncationStrategy)

Like [`MatrixAlgebraKit.findtruncated`](@ref) but assumes that the values are real and
sorted in descending order, as typically obtained by the SVD. This assumption is not
checked, and this is used in the default implementation of [`svd_trunc!`](@ref).
""" findtruncated_svd

@doc """
    truncate(::typeof(f), F, strategy::TruncationStrategy) -> F′, ind

Given a factorization function `f` and truncation `strategy`, truncate the factors `F` such
that the rows or columns at the indices `ind` are kept.

See also [`findtruncated`](@ref) and [`findtruncated_svd`](@ref) for determining the indices.
"""
function truncate end

"""
    TruncatedAlgorithm(alg::AbstractAlgorithm, trunc::TruncationAlgorithm)

Generic wrapper type for algorithms that consist of first using `alg`, followed by a
truncation through `trunc`.
"""
struct TruncatedAlgorithm{A, T} <: AbstractAlgorithm
    alg::A
    trunc::T
end

left_orth_kind(alg::TruncatedAlgorithm) = left_orth_kind(alg.alg)
right_orth_kind(alg::TruncatedAlgorithm) = right_orth_kind(alg.alg)
does_truncate(::TruncatedAlgorithm) = true

# Utility macros
# --------------

"""
    @algdef AlgorithmName

Convenience macro to define an algorithm `AlgorithmName` that accepts generic keywords.
This defines an exported alias for [`Algorithm{:AlgorithmName}`](@ref Algorithm)
along with some utility methods.
"""
macro algdef(name)
    return esc(
        quote
            const $name{K} = Algorithm{$(QuoteNode(name)), K}
            function $name(; kwargs...)
                # TODO: is this necessary/useful?
                kw = NamedTuple(kwargs) # normalize type
                return $name{typeof(kw)}(kw)
            end
            function Base.show(io::IO, alg::$name)
                return ($_show_alg)(io, alg)
            end

            Core.@__doc__ $name
        end
    )
end

function _arg_expr(::Val{1}, f, f!)
    return quote # out of place to inplace
        @inline $f(A; alg = nothing, kwargs...) = $f(A, select_algorithm($f, A, alg; kwargs...))
        # $f(A; kwargs...) = $f!(copy_input($f, A); kwargs...)
        $f(A, alg::AbstractAlgorithm) = $f!(copy_input($f, A), alg)

        # fill in arguments
        function $f!(A; alg = nothing, kwargs...)
            return $f!(A, select_algorithm($f!, A, alg; kwargs...))
        end
        function $f!(A, out; alg = nothing, kwargs...)
            return $f!(A, out, select_algorithm($f!, A, alg; kwargs...))
        end
        function $f!(A, alg::AbstractAlgorithm)
            return $f!(A, initialize_output($f!, A, alg), alg)
        end

        # define fallbacks for algorithm selection
        @inline function select_algorithm(::typeof($f), A, alg::Alg; kwargs...) where {Alg}
            return select_algorithm($f!, A, alg; kwargs...)
        end
        # define default algorithm fallbacks for out-of-place functions
        # in terms of the corresponding in-place function
        @inline function default_algorithm(::typeof($f), A; kwargs...)
            return default_algorithm($f!, A; kwargs...)
        end
        # define default algorithm fallbacks for out-of-place functions
        # in terms of the corresponding in-place function for types,
        # in principle this is covered by the definition above but
        # it is necessary to avoid ambiguity errors with the generic definitions:
        # ```julia
        # default_algorithm(f::F, A; kwargs...) where {F} = default_algorithm(f, typeof(A); kwargs...)
        # function default_algorithm(f::F, ::Type{T}; kwargs...) where {F,T}
        #     throw(MethodError(default_algorithm, (f, T)))
        # end
        # ```
        @inline function default_algorithm(::typeof($f), ::Type{A}; kwargs...) where {A}
            return default_algorithm($f!, A; kwargs...)
        end

        # copy documentation to both functions
        Core.@__doc__ $f, $f!
    end
end

function _arg_expr(::Val{2}, f, f!)
    return quote
        # out of place to inplace
        $f(A, B; kwargs...) = $f!(copy_input($f, A, B)...; kwargs...)
        $f(A, B, alg::AbstractAlgorithm) = $f!(copy_input($f, A, B)..., alg)

        # fill in arguments
        function $f!(A, B; alg = nothing, kwargs...)
            return $f!(A, B, select_algorithm($f!, (A, B), alg; kwargs...))
        end
        function $f!(A, B, out; alg = nothing, kwargs...)
            return $f!(A, B, out, select_algorithm($f!, (A, B), alg; kwargs...))
        end
        function $f!(A, B, alg::AbstractAlgorithm)
            return $f!(A, B, initialize_output($f!, A, B, alg), alg)
        end

        # define fallbacks for algorithm selection
        @inline function select_algorithm(::typeof($f), A, alg::Alg; kwargs...) where {Alg}
            return select_algorithm($f!, A, alg; kwargs...)
        end
        # define default algorithm fallbacks for out-of-place functions
        # in terms of the corresponding in-place function
        @inline function default_algorithm(::typeof($f), A, B; kwargs...)
            return default_algorithm($f!, A, B; kwargs...)
        end
        # define default algorithm fallbacks for out-of-place functions
        # in terms of the corresponding in-place function for types,
        # in principle this is covered by the definition above but
        # it is necessary to avoid ambiguity errors with the generic definitions:
        # ```julia
        # default_algorithm(f::F, A; kwargs...) where {F} = default_algorithm(f, typeof(A); kwargs...)
        # function default_algorithm(f::F, ::Type{T}; kwargs...) where {F,T}
        #     throw(MethodError(default_algorithm, (f, T)))
        # end
        # ```
        @inline function default_algorithm(::typeof($f), ::Type{A}, ::Type{B}; kwargs...) where {A, B}
            return default_algorithm($f!, A, B; kwargs...)
        end

        # copy documentation to both functions
        Core.@__doc__ $f, $f!
    end
end

"""
    @functiondef [n_args=1] f

Convenience macro to define the boilerplate code that dispatches between several versions of `f` and `f!`.
By default, `f` accepts a single argument `A`.  This enables the following signatures to be defined in terms of
the final `f!(A, out, alg::Algorithm)`.

```julia
    f(A; kwargs...)
    f(A, alg::Algorithm)
    f!(A, [out]; kwargs...)
    f!(A, alg::Algorithm)
```

The number of inputs can be set with the `n_args` keyword
argument, so that 

```julia
@functiondef n_args=2 f
```

would create 

```julia
    f(A, B; kwargs...)
    f(A, B, alg::Algorithm)
    f!(A, B, [out]; kwargs...)
    f!(A, B, alg::Algorithm)
```

See also [`copy_input`](@ref), [`select_algorithm`](@ref) and [`initialize_output`](@ref).
"""
macro functiondef(args...)
    kwargs = map(args[1:(end - 1)]) do kwarg
        if kwarg isa Symbol
            :($kwarg = $kwarg)
        elseif Meta.isexpr(kwarg, :(=))
            kwarg
        else
            throw(ArgumentError("Invalid keyword argument '$kwarg'"))
        end
    end
    isempty(kwargs) || length(kwargs) == 1 || throw(ArgumentError("Only one keyword argument to `@functiondef` is supported"))
    f_n_args = 1 # default
    if length(kwargs) == 1
        kwarg = only(kwargs) # only one kwarg is currently supported, TODO modify if we support more
        key::Symbol, val = kwarg.args
        key === :n_args || throw(ArgumentError("Unsupported keyword argument $key to `@functiondef`"))
        (isa(val, Integer) && val > 0) || throw(ArgumentError("`n_args` keyword argument to `@functiondef` should be an integer > 0"))
        f_n_args = val
    end

    f = args[end]
    f isa Symbol || throw(ArgumentError("Unsupported usage of `@functiondef`"))
    f! = Symbol(f, :!)

    return esc(_arg_expr(Val(f_n_args), f, f!))
end

"""
    @check_scalar(x, y, [op], [eltype])

Check if `eltype(x) == op(eltype(y))` and throw an error if not.
By default `op = identity` and `eltype = eltype'.
"""
macro check_scalar(x, y, op = :identity, eltype = :eltype)
    error_message = "Unexpected scalar type: "
    error_message *= string(eltype) * "(" * string(x) * ")"
    if op == :identity
        error_message *= " != " * string(eltype) * "(" * string(y) * ")"
    else
        error_message *= " != " * string(op) * "(" * string(eltype) * "(" * string(y) * "))"
    end
    return esc(
        quote
            $eltype($x) == $op($eltype($y)) || throw(ArgumentError($error_message))
        end
    )
end

"""
    @check_size(x, sz, [size])

Check if `size(x) == sz` and throw an error if not.
By default, `size = size`.
"""
macro check_size(x, sz, size = :size)
    msgstart = string(size) * "(" * string(x) * ") = "
    err = gensym()
    return esc(
        quote
            szx = $size($x)
            $err = $msgstart * string(szx) * " instead of expected value " *
                string($sz)
            szx == $sz || throw(DimensionMismatch($err))
        end
    )
end
