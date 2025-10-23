# Orth functions
# --------------
"""
    left_orth(A; [trunc], kwargs...) -> V, C
    left_orth!(A, [VC]; [trunc], kwargs...) -> V, C

Compute an orthonormal basis `V` for the image of the matrix `A`, as well as a  matrix `C`
(the corestriction) such that `A` factors as `A = V * C`.

This is a high-level wrapper where the keyword arguments can be used to specify and control
the specific orthogonal decomposition that should be used to factor `A`, whereas `trunc`
can optionally be used to control the precision in determining the rank of `A`, typically
via its singular values.

## Truncation
The optional truncation strategy can be controlled via the `trunc` keyword argument, and
any non-trivial strategy typically requires an SVD-based decompositions. This keyword can
be either a `NamedTuple` or a [`TruncationStrategy`](@ref).

### `trunc::NamedTuple`
The supported truncation keyword arguments are:

$(docs_truncation_kwargs)

### `trunc::TruncationStrategy`
For more control, a truncation strategy can be supplied directly.
By default, MatrixAlgebraKit supplies the following:

$(docs_truncation_strategies)

## Keyword arguments
There are 3 major modes of operation, based on the `alg` keyword, with slightly different
application purposes.

### `alg::Nothing`
This default mode uses the presence of a truncation strategy `trunc` to determine an optimal
decomposition type, which will be QR-based for no truncation, or SVD-based for truncation.
The remaining keyword arguments are passed on directly to the algorithm selection procedure
of the chosen decomposition type.

### `alg::Symbol`
Here, the driving selector is `alg`, and depending on its value, the algorithm selection
procedure takes other keywords into account:

* `:qr` : Factorize via QR decomposition, with further customizations through the
  `qr` keyword. This mode requires `isnothing(trunc)`, and is equivalent to
```julia
        V, C = qr_compact(A; alg = qr)
```

* `:polar` : Factorize via polar decomposition, with further customizations through the
  `polar` keyword. This mode requires `isnothing(trunc)`, and is equivalent to
```julia
        V, C = left_polar(A; alg = polar)
```

* `:svd` : Factorize via SVD, with further customizations through the `svd` keyword.
  This mode further allows truncation, which can be selected through the `trunc` argument.
  This mode is roughly equivalent to:
```julia
        V, S, C = svd_trunc(A; trunc, alg = svd)
        C = S * C
```

### `alg::AbstractAlgorithm`
In this expert mode the algorithm is supplied directly, and the kind of decomposition is
deduced from that. This is achieved either directly by providing a
[`LeftOrthAlgorithm{kind}`](@ref LeftOrthAlgorithm), or automatically by attempting to
deduce the decomposition kind with `LeftOrthAlgorithm(alg)`.

---

!!! note
    The bang method `left_orth!` optionally accepts the output structure and possibly
    destroys the input matrix `A`. Always use the return value of the function as it may
    not always be possible to use the provided `CV` as output.

See also [`right_orth(!)`](@ref right_orth), [`left_null(!)`](@ref left_null), [`right_null(!)`](@ref right_null)
"""
@functiondef left_orth

"""
    right_orth(A; [trunc], kwargs...) -> C, Vᴴ
    right_orth!(A, [CVᴴ]; [trunc], kwargs...) -> C, Vᴴ

Compute an orthonormal basis `V = adjoint(Vᴴ)` for the coimage of the matrix `A`, i.e. for
the image of `adjoint(A)`, as well as a matrix `C` such that `A` factors as `A = C * Vᴴ`.

This is a high-level wrapper where the keyword arguments can be used to specify and control
the specific orthogonal decomposition that should be used to factor `A`, whereas `trunc` can
optionally be used to control the precision in determining the rank of `A`, typically via
its singular values.

## Truncation
The optional truncation strategy can be controlled via the `trunc` keyword argument, and
any non-trivial strategy typically requires an SVD-based decompositions. This keyword can
be either a `NamedTuple` or a [`TruncationStrategy`](@ref).

### `trunc::NamedTuple`
The supported truncation keyword arguments are:

$(docs_truncation_kwargs)

### `trunc::TruncationStrategy`
For more control, a truncation strategy can be supplied directly.
By default, MatrixAlgebraKit supplies the following:

$(docs_truncation_strategies)

## Keyword arguments
There are 3 major modes of operation, based on the `alg` keyword, with slightly different
application purposes.

### `alg::Nothing`
This default mode uses the presence of a truncation strategy `trunc` to determine an optimal
decomposition type, which will be LQ-based for no truncation, or SVD-based for truncation.
The remaining keyword arguments are passed on directly to the algorithm selection procedure
of the chosen decomposition type.

### `alg::Symbol`
Here, the driving selector is `alg`, and depending on its value, the algorithm selection
procedure takes other keywords into account:

* `:lq` : Factorize via LQ decomposition, with further customizations through the
  `lq` keyword. This mode requires `isnothing(trunc)`, and is equivalent to
```julia
        C, Vᴴ = lq_compact(A; alg = lq)
```

* `:polar` : Factorize via polar decomposition, with further customizations through the
  `polar` keyword. This mode requires `isnothing(trunc)`, and is equivalent to
```julia
        C, Vᴴ = right_polar(A; alg = polar)
```

* `:svd` : Factorize via SVD, with further customizations through the `svd` keyword.
  This mode further allows truncation, which can be selected through the `trunc` argument.
  This mode is roughly equivalent to:
```julia
        C, S, Vᴴ = svd_trunc(A; trunc, alg = svd)
        C = C * S
```

### `alg::AbstractAlgorithm`
In this expert mode the algorithm is supplied directly, and the kind of decomposition is
deduced from that. This is achieved either directly by providing a
[`RightOrthAlgorithm{kind}`](@ref RightOrthAlgorithm), or automatically by attempting to
deduce the decomposition kind with `RightOrthAlgorithm(alg)`.

---

!!! note
    The bang method `right_orth!` optionally accepts the output structure and possibly
    destroys the input matrix `A`. Always use the return value of the function as it may not
    always be possible to use the provided `CVᴴ` as output.

See also [`left_orth(!)`](@ref left_orth), [`left_null(!)`](@ref left_null),
[`right_null(!)`](@ref right_null)
"""
@functiondef right_orth

# Null functions
# --------------
"""
    left_null(A; [trunc], kwargs...) -> N
    left_null!(A, [N]; [trunc], kwargs...) -> N

Compute an orthonormal basis `N` for the cokernel of the matrix `A`, i.e. the nullspace of
`adjoint(A)`, such that `adjoint(A) * N ≈ 0` and `N' * N ≈ I`.

This is a high-level wrapper where the keyword arguments can be used to specify and control
the underlying orthogonal decomposition that should be used to find the null space of `A'`,
whereas `trunc` can optionally  be used to control the precision in determining the rank of
`A`, typically via its singular values.

## Truncation
The optional truncation strategy can be controlled via the `trunc` keyword argument, and any
non-trivial strategy typically requires an SVD-based decomposition. This keyword can be
either a `NamedTuple` or a [`TruncationStrategy`](@ref).

### `trunc::NamedTuple`
The supported truncation keyword arguments are:

$(docs_null_truncation_kwargs)

### `trunc::TruncationStrategy`
For more control, a truncation strategy can be supplied directly. By default,
MatrixAlgebraKit supplies the following:

$(docs_truncation_strategies)

!!! note
    Here [`notrunc`](@ref) has special meaning, and signifies keeping the values that
    correspond to the exact zeros determined from the additional columns of `A`.

## Keyword arguments
There are 3 major modes of operation, based on the `alg` keyword, with slightly different
application purposes.

### `alg::Nothing`
This default mode uses the presence of a truncation strategy `trunc` to determine an optimal
decomposition type, which will be QR-based for no truncation, or SVD-based for truncation.
The remaining keyword arguments are passed on directly to the algorithm selection procedure
of the chosen decomposition type.

### `alg::Symbol`
Here, the driving selector is `alg`, and depending on its value, the algorithm selection
procedure takes other keywords into account:

* `:qr` : Factorize via QR nullspace, with further customizations through the `qr`
  keyword. This mode requires `isnothing(trunc)`, and is equivalent to
```julia
        N = qr_null(A; alg = qr)
```

* `:svd` : Factorize via SVD, with further customizations through the `svd` keyword.
  This mode further allows truncation, which can be selected through the `trunc` argument.
  It is roughly equivalent to:
```julia
        U, S, _ = svd_trunc(A; trunc, alg = svd)
        N = truncate(left_null, (U, S), trunc)
```

### `alg::AbstractAlgorithm`
In this expert mode the algorithm is supplied directly, and the kind of decomposition is
deduced from that. This is achieved either directly by providing a
[`LeftNullAlgorithm{kind}`](@ref LeftNullAlgorithm), or automatically by attempting to
deduce the decomposition kind with `LeftNullAlgorithm(alg)`.

---

!!! note
    The bang method `left_null!` optionally accepts the output structure and possibly
    destroys the input matrix `A`. Always use the return value of the function as it may not
    always be possible to use the provided `N` as output.

See also [`right_null(!)`](@ref right_null), [`left_orth(!)`](@ref left_orth),
[`right_orth(!)`](@ref right_orth)
"""
@functiondef left_null

"""
    right_null(A; [trunc], kwargs...) -> Nᴴ
    right_null!(A, [Nᴴ]; [trunc], kwargs...) -> Nᴴ

Compute an orthonormal basis `N = adjoint(Nᴴ)` for the kernel of the matrix `A`, i.e. the
nullspace of `A`, such that `A * Nᴴ' ≈ 0` and `Nᴴ * Nᴴ' ≈ I`.

This is a high-level wrapper where the keyword arguments can be used to specify and control
the underlying orthogonal decomposition that should be used to find the null space of `A`,
whereas `trunc` can optionally  be used to control the precision in determining the rank of
`A`, typically via its singular values.

## Truncation
The optional truncation strategy can be controlled via the `trunc` keyword argument, and any
non-trivial strategy typically requires an SVD-based decomposition. This keyword can be
either a `NamedTuple` or a [`TruncationStrategy`](@ref).

### `trunc::NamedTuple`
The supported truncation keyword arguments are:

$(docs_null_truncation_kwargs)

### `trunc::TruncationStrategy`
For more control, a truncation strategy can be supplied directly. By default,
MatrixAlgebraKit supplies the following:

$(docs_truncation_strategies)

!!! note
    Here [`notrunc`](@ref) has special meaning, and signifies keeping the values that
    correspond to the exact zeros determined from the additional rows of `A`.

## Keyword arguments
There are 3 major modes of operation, based on the `alg` keyword, with slightly different
application purposes.

### `alg::Nothing`
This default mode uses the presence of a truncation strategy `trunc` to determine an optimal
decomposition type, which will be LQ-based for no truncation, or SVD-based for truncation.
The remaining keyword arguments are passed on directly to the algorithm selection procedure
of the chosen decomposition type.

### `alg::Symbol`
Here, the driving selector is `alg`, and depending on its value, the algorithm selection
procedure takes other keywords into account:

* `:lq` : Factorize via LQ nullspace, with further customizations through the `lq`
  keyword. This mode requires `isnothing(trunc)`, and is equivalent to
```julia
        Nᴴ = lq_null(A; alg = lq)
```

* `:svd` : Factorize via SVD, with further customizations through the `svd` keyword.
  This mode further allows truncation, which can be selected through the `trunc` argument.
  It is roughly equivalent to:
```julia
        _, S, Vᴴ = svd_trunc(A; trunc, alg = svd)
        Nᴴ = truncate(right_null, (S, Vᴴ), trunc)
```

### `alg::AbstractAlgorithm`
In this expert mode the algorithm is supplied directly, and the kind of decomposition is
deduced from that. This is achieved either directly by providing a
[`RightNullAlgorithm{kind}`](@ref RightNullAlgorithm), or automatically by attempting to
deduce the decomposition kind with `RightNullAlgorithm(alg)`.

---

!!! note
    The bang method `right_null!` optionally accepts the output structure and possibly
    destroys the input matrix `A`. Always use the return value of the function as it may not
    always be possible to use the provided `Nᴴ` as output.

See also [`left_null(!)`](@ref left_null), [`left_orth(!)`](@ref left_orth),
[`right_orth(!)`](@ref right_orth)
"""
@functiondef right_null

# Algorithm selection
# -------------------
# specific override for `alg::Symbol` case, to allow for choosing the kind of factorization.
@inline select_algorithm(::typeof(left_orth!), A, alg::Symbol; kwargs...) =
    select_algorithm(left_orth!, A, Val(alg); kwargs...)
@inline select_algorithm(::typeof(right_orth!), A, alg::Symbol; kwargs...) =
    select_algorithm(right_orth!, A, Val(alg); kwargs...)
@inline select_algorithm(::typeof(left_null!), A, alg::Symbol; kwargs...) =
    select_algorithm(left_null!, A, Val(alg); kwargs...)
@inline select_algorithm(::typeof(right_null!), A, alg::Symbol; kwargs...) =
    select_algorithm(right_null!, A, Val(alg); kwargs...)

function select_algorithm(::typeof(left_orth!), A, ::Val{:qr}; trunc = nothing, kwargs...)
    isnothing(trunc) ||
        throw(ArgumentError("QR-based `left_orth` is incompatible with specifying `trunc`"))
    alg′ = select_algorithm(qr_compact!, A, get(kwargs, :qr, nothing))
    return LeftOrthViaQR(alg′)
end
function select_algorithm(::typeof(left_orth!), A, ::Val{:polar}; trunc = nothing, kwargs...)
    isnothing(trunc) ||
        throw(ArgumentError("Polar-based `left_orth` is incompatible with specifying `trunc`"))
    alg′ = select_algorithm(left_polar!, A, get(kwargs, :polar, nothing))
    return LeftOrthViaPolar(alg′)
end
function select_algorithm(::typeof(left_orth!), A, ::Val{:svd}; trunc = nothing, kwargs...)
    alg = get(kwargs, :svd, nothing)
    alg′ = isnothing(trunc) ? select_algorithm(svd_compact!, A, alg) :
        select_algorithm(svd_trunc!, A, alg; trunc)
    return LeftOrthViaSVD(alg′)
end

function select_algorithm(::typeof(right_orth!), A, ::Val{:lq}; trunc = nothing, kwargs...)
    isnothing(trunc) ||
        throw(ArgumentError("LQ-based `right_orth` is incompatible with specifying `trunc`"))
    alg = select_algorithm(lq_compact!, A, get(kwargs, :lq, nothing))
    return RightOrthViaLQ(alg)
end
function select_algorithm(::typeof(right_orth!), A, ::Val{:polar}; trunc = nothing, kwargs...)
    isnothing(trunc) ||
        throw(ArgumentError("Polar-based `right_orth` is incompatible with specifying `trunc`"))
    alg = select_algorithm(right_polar!, A, get(kwargs, :polar, nothing))
    return RightOrthViaPolar(alg)
end
function select_algorithm(::typeof(right_orth!), A, ::Val{:svd}; trunc = nothing, kwargs...)
    alg = get(kwargs, :svd, nothing)
    alg′ = isnothing(trunc) ? select_algorithm(svd_compact!, A, alg) :
        select_algorithm(svd_trunc!, A, alg; trunc)
    return RightOrthViaSVD(alg′)
end

function select_algorithm(::typeof(left_null!), A, ::Val{:qr}; trunc = nothing, kwargs...)
    isnothing(trunc) ||
        throw(ArgumentError("QR-based `left_null` is incompatible with specifying `trunc`"))
    alg = select_algorithm(qr_null!, A, get(kwargs, :qr, nothing))
    return LeftNullViaQR(alg)
end
function select_algorithm(::typeof(left_null!), A, ::Val{:svd}; trunc = nothing, kwargs...)
    alg_svd = select_algorithm(svd_full!, A, get(kwargs, :svd, nothing))
    alg = TruncatedAlgorithm(alg_svd, select_null_truncation(trunc))
    return LeftNullViaSVD(alg)
end

function select_algorithm(::typeof(right_null!), A, ::Val{:lq}; trunc = nothing, kwargs...)
    isnothing(trunc) ||
        throw(ArgumentError("LQ-based `right_null` is incompatible with specifying `trunc`"))
    alg = select_algorithm(lq_null!, A, get(kwargs, :lq, nothing))
    return RightNullViaLQ(alg)
end
function select_algorithm(::typeof(right_null!), A, ::Val{:svd}; trunc = nothing, kwargs...)
    alg_svd = select_algorithm(svd_full!, A, get(kwargs, :svd, nothing))
    alg = TruncatedAlgorithm(alg_svd, select_null_truncation(trunc))
    return RightNullViaSVD(alg)
end

default_algorithm(::typeof(left_orth!), A::TA; trunc = nothing, kwargs...) where {TA} =
    isnothing(trunc) ? select_algorithm(left_orth!, A, Val(:qr); qr = kwargs) :
    select_algorithm(left_orth!, A, Val(:svd); svd = kwargs)
# disambiguate
default_algorithm(::typeof(left_orth!), ::Type{A}; trunc = nothing, kwargs...) where {A} =
    isnothing(trunc) ? select_algorithm(left_orth!, A, Val(:qr); qr = kwargs) :
    select_algorithm(left_orth!, A, Val(:svd); svd = kwargs)

default_algorithm(::typeof(right_orth!), A::TA; trunc = nothing, kwargs...) where {TA} =
    isnothing(trunc) ? select_algorithm(right_orth!, A, Val(:lq); lq = kwargs) :
    select_algorithm(right_orth!, A, Val(:svd); svd = kwargs)
# disambiguate
default_algorithm(::typeof(right_orth!), ::Type{A}; trunc = nothing, kwargs...) where {A} =
    isnothing(trunc) ? select_algorithm(right_orth!, A, Val(:lq); lq = kwargs) :
    select_algorithm(right_orth!, A, Val(:svd); svd = kwargs)

default_algorithm(::typeof(left_null!), A::TA; trunc = nothing, kwargs...) where {TA} =
    isnothing(trunc) ? select_algorithm(left_null!, A, Val(:qr); qr = kwargs) :
    select_algorithm(left_null!, A, Val(:svd); svd = kwargs, trunc)
# disambiguate
default_algorithm(::typeof(left_null!), ::Type{A}; trunc = nothing, kwargs...) where {A} =
    isnothing(trunc) ? select_algorithm(left_null!, A, Val(:qr); qr = kwargs) :
    select_algorithm(left_null!, A, Val(:svd); svd = kwargs, trunc)

default_algorithm(::typeof(right_null!), A::TA; trunc = nothing, kwargs...) where {TA} =
    isnothing(trunc) ? select_algorithm(right_null!, A, Val(:lq); lq = kwargs) :
    select_algorithm(right_null!, A, Val(:svd); svd = kwargs, trunc)
# disambiguate
default_algorithm(::typeof(right_null!), ::Type{A}; trunc = nothing, kwargs...) where {A} =
    isnothing(trunc) ? select_algorithm(right_null!, A, Val(:lq); lq = kwargs) :
    select_algorithm(right_null!, A, Val(:svd); svd = kwargs, trunc)

left_orth_alg(alg::AbstractAlgorithm) = LeftOrthAlgorithm(alg)
right_orth_alg(alg::AbstractAlgorithm) = RightOrthAlgorithm(alg)
left_null_alg(alg::AbstractAlgorithm) = LeftNullAlgorithm(alg)
right_null_alg(alg::AbstractAlgorithm) = RightNullAlgorithm(alg)
