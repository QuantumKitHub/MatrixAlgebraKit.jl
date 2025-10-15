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
  `alg_qr` keyword. This mode requires `isnothing(trunc)`, and is equivalent to
```julia
        V, C = qr_compact(A; alg_qr...)
```

* `:polar` : Factorize via polar decomposition, with further customizations through the
  `alg_polar` keyword. This mode requires `isnothing(trunc)`, and is equivalent to
```julia
        V, C = left_polar(A; alg_polar...)
```

* `:svd` : Factorize via SVD, with further customizations through the `alg_svd` keyword.
  This mode further allows truncation, which can be selected through the `trunc` argument.
  This mode is roughly equivalent to:
```julia
        V, S, C = svd_trunc(A; trunc, alg_svd...)
        C = S * C
```

### `alg::AbstractAlgorithm`
In this expert mode the algorithm is supplied directly, and the kind of decomposition is
deduced from that. This hinges on the implementation of the algorithm trait
[`MatrixAlgebraKit.left_orth_kind(alg)`](@ref).

---

!!! note
    The bang method `left_orth!` optionally accepts the output structure and possibly
    destroys the input matrix `A`. Always use the return value of the function as it may
    not always be possible to use the provided `CV` as output.

See also [`right_orth(!)`](@ref right_orth), [`left_null(!)`](@ref left_null),
[`right_null(!)`](@ref right_null)
"""
@functiondef left_orth

# helper functions
function left_orth_qr! end
function left_orth_polar! end
function left_orth_svd! end

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
  `alg_lq` keyword. This mode requires `isnothing(trunc)`, and is equivalent to
```julia
        C, Vᴴ = lq_compact(A; alg_lq...)
```

* `:polar` : Factorize via polar decomposition, with further customizations through the
  `alg_polar` keyword. This mode requires `isnothing(trunc)`, and is equivalent to
```julia
        C, Vᴴ = right_polar(A; alg_polar...)
```

* `:svd` : Factorize via SVD, with further customizations through the `alg_svd` keyword.
  This mode further allows truncation, which can be selected through the `trunc` argument.
  This mode is roughly equivalent to:
```julia
        C, S, Vᴴ = svd_trunc(A; trunc, alg_svd...)
        C = C * S
```

### `alg::AbstractAlgorithm`
In this expert mode the algorithm is supplied directly, and the kind of decomposition is
deduced from that. This hinges on the implementation of the algorithm trait
[`MatrixAlgebraKit.right_orth_kind(alg)`](@ref).

---

!!! note
    The bang method `right_orth!` optionally accepts the output structure and possibly
    destroys the input matrix `A`. Always use the return value of the function as it may not
    always be possible to use the provided `CVᴴ` as output.

See also [`left_orth(!)`](@ref left_orth), [`left_null(!)`](@ref left_null),
[`right_null(!)`](@ref right_null)
"""
@functiondef right_orth

# helper functions
function right_orth_lq! end
function right_orth_polar! end
function right_orth_svd! end

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

* `:qr` : Factorize via QR nullspace, with further customizations through the `alg_qr`
  keyword. This mode requires `isnothing(trunc)`, and is equivalent to
```julia
        N = qr_null(A; alg_qr...)
```

* `:svd` : Factorize via SVD, with further customizations through the `alg_svd` keyword.
  This mode further allows truncation, which can be selected through the `trunc` argument.
  It is roughly equivalent to:
```julia
        U, S, _ = svd_trunc(A; trunc, alg_svd...)
        N = truncate(left_null, (U, S), trunc)
```

### `alg::AbstractAlgorithm`
In this expert mode the algorithm is supplied directly, and the kind of decomposition is
deduced from that. This hinges on the implementation of the algorithm trait
[`MatrixAlgebraKit.left_null_kind(alg)`](@ref).

---

!!! note
    The bang method `left_null!` optionally accepts the output structure and possibly
    destroys the input matrix `A`. Always use the return value of the function as it may not
    always be possible to use the provided `N` as output.

See also [`right_null(!)`](@ref right_null), [`left_orth(!)`](@ref left_orth),
[`right_orth(!)`](@ref right_orth)
"""
@functiondef left_null

# helper functions
function left_null_qr! end
function left_null_svd! end

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

* `:lq` : Factorize via LQ nullspace, with further customizations through the `alg_lq`
  keyword. This mode requires `isnothing(trunc)`, and is equivalent to
```julia
        Nᴴ = lq_null(A; alg_qr...)
```

* `:svd` : Factorize via SVD, with further customizations through the `alg_svd` keyword.
  This mode further allows truncation, which can be selected through the `trunc` argument.
  It is roughly equivalent to:
```julia
        _, S, Vᴴ = svd_trunc(A; trunc, alg_svd...)
        Nᴴ = truncate(right_null, (S, Vᴴ), trunc)
```

### `alg::AbstractAlgorithm`
In this expert mode the algorithm is supplied directly, and the kind of decomposition is
deduced from that. This hinges on the implementation of the algorithm trait
[`MatrixAlgebraKit.right_null_kind(alg)`](@ref).

---

!!! note
    The bang method `right_null!` optionally accepts the output structure and possibly
    destroys the input matrix `A`. Always use the return value of the function as it may not
    always be possible to use the provided `Nᴴ` as output.

See also [`left_null(!)`](@ref left_null), [`left_orth(!)`](@ref left_orth),
[`right_orth(!)`](@ref right_orth)
"""
@functiondef right_null

# helper functions
function right_null_lq! end
function right_null_svd! end

# Algorithm selection
# -------------------
# specific override for `alg::Symbol` case, to allow for choosing the kind of factorization.
@inline select_algorithm(::typeof(left_orth!), A, alg::Symbol; trunc = nothing, kwargs...) =
    LeftOrthAlgorithm{alg}(A; alg = get(kwargs, alg, nothing), trunc)
@inline select_algorithm(::typeof(right_orth!), A, alg::Symbol; trunc = nothing, kwargs...) =
    RightOrthAlgorithm{alg}(A; alg = get(kwargs, alg, nothing), trunc)

function LeftOrthViaQR(A; alg = nothing, trunc = nothing, kwargs...)
    isnothing(trunc) ||
        throw(ArgumentError("QR-based `left_orth` is incompatible with specifying `trunc`"))
    alg = select_algorithm(qr_compact!, A, alg; kwargs...)
    return LeftOrthViaQR{typeof(alg)}(alg)
end
function LeftOrthViaPolar(A; alg = nothing, trunc = nothing, kwargs...)
    isnothing(trunc) ||
        throw(ArgumentError("Polar-based `left_orth` is incompatible with specifying `trunc`"))
    alg = select_algorithm(left_polar!, A, alg; kwargs...)
    return LeftOrthViaPolar{typeof(alg)}(alg)
end
function LeftOrthViaSVD(A; alg = nothing, trunc = nothing, kwargs...)
    alg = isnothing(trunc) ? select_algorithm(svd_compact!, A, alg; kwargs...) :
        select_algorithm(svd_trunc!, A, alg; trunc, kwargs...)
    return LeftOrthViaSVD{typeof(alg)}(alg)
end

function RightOrthViaLQ(A; alg = nothing, trunc = nothing, kwargs...)
    isnothing(trunc) ||
        throw(ArgumentError("LQ-based `right_orth` is incompatible with specifying `trunc`"))
    alg = select_algorithm(lq_compact!, A, alg; kwargs...)
    return RightOrthViaLQ{typeof(alg)}(alg)
end
function RightOrthViaPolar(A; alg = nothing, trunc = nothing, kwargs...)
    isnothing(trunc) ||
        throw(ArgumentError("Polar-based `right_orth` is incompatible with specifying `trunc`"))
    alg = select_algorithm(right_polar!, A, alg; kwargs...)
    return RightOrthViaPolar{typeof(alg)}(alg)
end
function RightOrthViaSVD(A; alg = nothing, trunc = nothing, kwargs...)
    alg = isnothing(trunc) ? select_algorithm(svd_compact!, A, alg; kwargs...) :
        select_algorithm(svd_trunc!, A, alg; trunc, kwargs...)
    return RightOrthViaSVD{typeof(alg)}(alg)
end

default_algorithm(::typeof(left_orth!), A::TA; trunc = nothing, kwargs...) where {TA} =
    isnothing(trunc) ? LeftOrthViaQR(A; kwargs...) : LeftOrthViaSVD(A; trunc, kwargs...)
# disambiguate
default_algorithm(::typeof(left_orth!), ::Type{A}; trunc = nothing, kwargs...) where {A} =
    isnothing(trunc) ? LeftOrthViaQR(A; kwargs...) : LeftOrthViaSVD(A; trunc, kwargs...)

default_algorithm(::typeof(right_orth!), A::TA; trunc = nothing, kwargs...) where {TA} =
    isnothing(trunc) ? RightOrthViaLQ(A; kwargs...) : RightOrthViaSVD(A; trunc, kwargs...)
# disambiguate
default_algorithm(::typeof(right_orth!), ::Type{A}; trunc = nothing, kwargs...) where {A} =
    isnothing(trunc) ? RightOrthViaLQ(A; kwargs...) : RightOrthViaSVD(A; trunc, kwargs...)

function select_algorithm(::typeof(left_null!), A, alg::Symbol; trunc = nothing, kwargs...)
    alg === :svd && return select_algorithm(
        left_null_svd!, A, get(kwargs, :alg_svd, nothing); trunc
    )

    isnothing(trunc) || throw(ArgumentError(lazy"alg ($alg) incompatible with truncation"))

    alg === :qr && return select_algorithm(left_null_qr!, A, get(kwargs, :alg_qr, nothing))

    throw(ArgumentError(lazy"unkown alg symbol $alg"))
end

default_algorithm(::typeof(left_null!), A::TA; trunc = nothing, kwargs...) where {TA} =
    isnothing(trunc) ? select_algorithm(left_null_qr!, A; kwargs...) :
    select_algorithm(left_null_svd!, A; trunc, kwargs...)
# disambiguate
default_algorithm(::typeof(left_null!), ::Type{A}; trunc = nothing, kwargs...) where {A} =
    isnothing(trunc) ? select_algorithm(left_null_qr!, A; kwargs...) :
    select_algorithm(left_null_svd!, A; trunc, kwargs...)

select_algorithm(::typeof(left_null_qr!), A, alg = nothing; kwargs...) =
    select_algorithm(qr_null!, A, alg; kwargs...)
function select_algorithm(::typeof(left_null_svd!), A, alg = nothing; trunc = nothing, kwargs...)
    if alg isa TruncatedAlgorithm
        isnothing(trunc) ||
            throw(ArgumentError("`trunc` can't be specified when `alg` is a `TruncatedAlgorithm`"))
        return alg
    else
        alg_svd = select_algorithm(svd_full!, A, alg; kwargs...)
        return TruncatedAlgorithm(alg_svd, select_null_truncation(trunc))
    end
end

function select_algorithm(::typeof(right_null!), A, alg::Symbol; trunc = nothing, kwargs...)
    alg === :svd && return select_algorithm(
        right_null_svd!, A, get(kwargs, :alg_svd, nothing); trunc
    )

    isnothing(trunc) || throw(ArgumentError(lazy"alg ($alg) incompatible with truncation"))

    alg === :lq && return select_algorithm(right_null_lq!, A, get(kwargs, :alg_lq, nothing))

    throw(ArgumentError(lazy"unkown alg symbol $alg"))
end

default_algorithm(::typeof(right_null!), A::TA; trunc = nothing, kwargs...) where {TA} =
    isnothing(trunc) ? select_algorithm(right_null_lq!, A; kwargs...) :
    select_algorithm(right_null_svd!, A; trunc, kwargs...)
default_algorithm(::typeof(right_null!), ::Type{A}; trunc = nothing, kwargs...) where {A} =
    isnothing(trunc) ? select_algorithm(right_null_lq!, A; kwargs...) :
    select_algorithm(right_null_svd!, A; trunc, kwargs...)

select_algorithm(::typeof(right_null_lq!), A, alg = nothing; kwargs...) =
    select_algorithm(lq_null!, A, alg; kwargs...)

function select_algorithm(::typeof(right_null_svd!), A, alg; trunc = nothing, kwargs...)
    if alg isa TruncatedAlgorithm
        isnothing(trunc) ||
            throw(ArgumentError("`trunc` can't be specified when `alg` is a `TruncatedAlgorithm`"))
        return alg
    else
        alg_svd = select_algorithm(svd_full!, A, alg; kwargs...)
        return TruncatedAlgorithm(alg_svd, select_null_truncation(trunc))
    end

end
