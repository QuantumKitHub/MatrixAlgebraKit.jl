# Orth functions
# --------------
"""
    left_orth(A; [alg], [trunc], kwargs...) -> V, C
    left_orth!(A, [VC], [alg]; [trunc], kwargs...) -> V, C

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
decomposition type, which will typically be QR-based for no truncation, or SVD-based for
truncation. The remaining keyword arguments are passed on directly to the algorithm selection
procedure of the chosen decomposition type.

### `alg::Symbol`
Here, the driving selector is `alg`, which is used to select the kind of decomposition. The
remaining keyword arguments are passed on directly to the algorithm selection procedure of
the chosen decomposition type. By default, the supported kinds are:

* `:qr` : Factorize via QR decomposition, with further customizations through the other
  keywords. This mode requires `isnothing(trunc)`, and is equivalent to
```julia
        V, C = qr_compact(A; kwargs...)
```

* `:polar` : Factorize via polar decomposition, with further customizations through the
  other keywords. This mode requires `isnothing(trunc)`, and is equivalent to
```julia
        V, C = left_polar(A; kwargs...)
```

* `:svd` : Factorize via SVD, with further customizations through the other keywords.
  This mode further allows truncation, which can be selected through the `trunc` argument,
  and is roughly equivalent to:
```julia
        V, S, C = svd_trunc(A; trunc, kwargs...)
        C = lmul!(S, C)
```

### `alg::AbstractAlgorithm`
In this expert mode the algorithm is supplied directly, and the kind of decomposition is
deduced from that. This is achieved either directly by providing a
[`LeftOrthAlgorithm{kind}`](@ref LeftOrthAlgorithm), or automatically by attempting to
deduce the decomposition kind with [`left_orth_alg(alg)`](@ref left_orth_alg).

---

!!! note
    The bang method `left_orth!` optionally accepts the output structure and possibly
    destroys the input matrix `A`. Always use the return value of the function as it may
    not always be possible to use the provided `CV` as output.

See also [`right_orth(!)`](@ref right_orth), [`left_null(!)`](@ref left_null) and
[`right_null(!)`](@ref right_null).
"""
@functiondef left_orth

"""
    right_orth(A; [alg], [trunc], kwargs...) -> C, Vᴴ
    right_orth!(A, [CVᴴ], [alg]; [trunc], kwargs...) -> C, Vᴴ

Compute an orthonormal basis `V = adjoint(Vᴴ)` for the coimage of the matrix `A`, i.e. for
the image of `adjoint(A)`, as well as a matrix `C` such that `A` factors as `A = C * Vᴴ`.

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
decomposition type, which will typicall be LQ-based for no truncation, or SVD-based for
truncation. The remaining keyword arguments are passed on directly to the algorithm selection
procedure of the chosen decomposition type.

### `alg::Symbol`
Here, the driving selector is `alg`, which is used to select the kind of decomposition. The
remaining keyword arguments are passed on directly to the algorithm selection procedure of
the chosen decomposition type. By default, the supported kinds are:

* `:lq` : Factorize via LQ decomposition, with further customizations through the other
  keywords. This mode requires `isnothing(trunc)`, and is equivalent to
```julia
        C, Vᴴ = lq_compact(A; kwargs...)
```

* `:polar` : Factorize via polar decomposition, with further customizations through the
  other keywords. This mode requires `isnothing(trunc)`, and is equivalent to
```julia
        C, Vᴴ = right_polar(A; kwargs...)
```

* `:svd` : Factorize via SVD, with further customizations through the other keywords.
  This mode further allows truncation, which can be selected through the `trunc` argument,
  and is roughly equivalent to:
```julia
        C, S, Vᴴ = svd_trunc(A; trunc, kwargs...)
        C = rmul!(C, S)
```

### `alg::AbstractAlgorithm`
In this expert mode the algorithm is supplied directly, and the kind of decomposition is
deduced from that. This is achieved either directly by providing a
[`RightOrthAlgorithm{kind}`](@ref RightOrthAlgorithm), or automatically by attempting to
deduce the decomposition kind with [`right_orth_alg`](@ref).

---

!!! note
    The bang method `right_orth!` optionally accepts the output structure and possibly
    destroys the input matrix `A`. Always use the return value of the function as it may not
    always be possible to use the provided `CVᴴ` as output.

See also [`left_orth(!)`](@ref left_orth), [`left_null(!)`](@ref left_null) and
[`right_null(!)`](@ref right_null).
"""
@functiondef right_orth

# Null functions
# --------------
"""
    left_null(A; [alg], [trunc], kwargs...) -> N
    left_null!(A, [N], [alg]; [trunc], kwargs...) -> N

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
Here, the driving selector is `alg`, which is used to select the kind of decomposition. The
remaining keyword arguments are passed on directly to the algorithm selection procedure of
the chosen decomposition type. By default, the supported kinds are:

* `:qr` : Factorize via QR nullspace, with further customizations through the other
  keywords. This mode requires `isnothing(trunc)`, and is equivalent to
```julia
        N = qr_null(A; kwargs...)
```

* `:svd` : Factorize via SVD, with further customizations through the other keywords.
  This mode further allows truncation, which can be selected through the `trunc` argument.
  It is roughly equivalent to:
```julia
        U, S, _ = svd_full(A; kwargs...)
        N = truncate(left_null, (U, S), trunc)
```

### `alg::AbstractAlgorithm`
In this expert mode the algorithm is supplied directly, and the kind of decomposition is
deduced from that. This is achieved either directly by providing a
[`LeftNullAlgorithm{kind}`](@ref LeftNullAlgorithm), or automatically by attempting to
deduce the decomposition kind with [`left_null_alg(alg)`](@ref left_null_alg).

---

!!! note
    The bang method `left_null!` optionally accepts the output structure and possibly
    destroys the input matrix `A`. Always use the return value of the function as it may not
    always be possible to use the provided `N` as output.

See also [`right_null(!)`](@ref right_null), [`left_orth(!)`](@ref left_orth) and
[`right_orth(!)`](@ref right_orth).
"""
@functiondef left_null

"""
    right_null(A; [alg], [trunc], kwargs...) -> Nᴴ
    right_null!(A, [Nᴴ], [alg]; [trunc], kwargs...) -> Nᴴ

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
Here, the driving selector is `alg`, which is used to select the kind of decomposition. The
remaining keyword arguments are passed on directly to the algorithm selection procedure of
the chosen decomposition type. By default, the supported kinds are:

* `:lq` : Factorize via LQ nullspace, with further customizations through the other
  keywords. This mode requires `isnothing(trunc)`, and is equivalent to
```julia
        Nᴴ = lq_null(A; kwargs...)
```

* `:svd` : Factorize via SVD, with further customizations through the other keywords.
  This mode further allows truncation, which can be selected through the `trunc` argument.
  It is roughly equivalent to:
```julia
        _, S, Vᴴ = svd_full(A; kwargs...)
        Nᴴ = truncate(right_null, (S, Vᴴ), trunc)
```

### `alg::AbstractAlgorithm`
In this expert mode the algorithm is supplied directly, and the kind of decomposition is
deduced from that. This is achieved either directly by providing a
[`RightNullAlgorithm{kind}`](@ref RightNullAlgorithm), or automatically by attempting to
deduce the decomposition kind with [`right_null_alg(alg)`](@ref right_null_alg).

---

!!! note
    The bang method `right_null!` optionally accepts the output structure and possibly
    destroys the input matrix `A`. Always use the return value of the function as it may not
    always be possible to use the provided `Nᴴ` as output.

See also [`left_null(!)`](@ref left_null), [`left_orth(!)`](@ref left_orth) and
[`right_orth(!)`](@ref right_orth).
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
    alg′ = select_algorithm(qr_compact!, A; kwargs...)
    return LeftOrthViaQR(alg′)
end
function select_algorithm(::typeof(left_orth!), A, ::Val{:polar}; trunc = nothing, kwargs...)
    isnothing(trunc) ||
        throw(ArgumentError("Polar-based `left_orth` is incompatible with specifying `trunc`"))
    alg′ = select_algorithm(left_polar!, A; kwargs...)
    return LeftOrthViaPolar(alg′)
end
function select_algorithm(::typeof(left_orth!), A, ::Val{:svd}; trunc = nothing, kwargs...)
    alg′ = isnothing(trunc) ? select_algorithm(svd_compact!, A; kwargs...) :
        select_algorithm(svd_trunc!, A; trunc, kwargs...)
    return LeftOrthViaSVD(alg′)
end

function select_algorithm(::typeof(right_orth!), A, ::Val{:lq}; trunc = nothing, kwargs...)
    isnothing(trunc) ||
        throw(ArgumentError("LQ-based `right_orth` is incompatible with specifying `trunc`"))
    alg = select_algorithm(lq_compact!, A; kwargs...)
    return RightOrthViaLQ(alg)
end
function select_algorithm(::typeof(right_orth!), A, ::Val{:polar}; trunc = nothing, kwargs...)
    isnothing(trunc) ||
        throw(ArgumentError("Polar-based `right_orth` is incompatible with specifying `trunc`"))
    alg = select_algorithm(right_polar!, A; kwargs...)
    return RightOrthViaPolar(alg)
end
function select_algorithm(::typeof(right_orth!), A, ::Val{:svd}; trunc = nothing, kwargs...)
    alg′ = isnothing(trunc) ? select_algorithm(svd_compact!, A; kwargs...) :
        select_algorithm(svd_trunc!, A; trunc, kwargs...)
    return RightOrthViaSVD(alg′)
end

function select_algorithm(::typeof(left_null!), A, ::Val{:qr}; trunc = nothing, kwargs...)
    isnothing(trunc) ||
        throw(ArgumentError("QR-based `left_null` is incompatible with specifying `trunc`"))
    alg = select_algorithm(qr_null!, A; kwargs...)
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
    alg = select_algorithm(lq_null!, A; kwargs...)
    return RightNullViaLQ(alg)
end
function select_algorithm(::typeof(right_null!), A, ::Val{:svd}; trunc = nothing, kwargs...)
    alg_svd = select_algorithm(svd_full!, A; kwargs...)
    alg = TruncatedAlgorithm(alg_svd, select_null_truncation(trunc))
    return RightNullViaSVD(alg)
end

default_algorithm(::typeof(left_orth!), ::Type{A}; trunc = nothing, kwargs...) where {A} =
    isnothing(trunc) ? select_algorithm(left_orth!, A, Val(:qr); kwargs...) :
    select_algorithm(left_orth!, A, Val(:svd); trunc, kwargs...)

default_algorithm(::typeof(right_orth!), ::Type{A}; trunc = nothing, kwargs...) where {A} =
    isnothing(trunc) ? select_algorithm(right_orth!, A, Val(:lq); kwargs...) :
    select_algorithm(right_orth!, A, Val(:svd); trunc, kwargs...)

default_algorithm(::typeof(left_null!), ::Type{A}; trunc = nothing, kwargs...) where {A} =
    isnothing(trunc) ? select_algorithm(left_null!, A, Val(:qr); kwargs...) :
    select_algorithm(left_null!, A, Val(:svd); trunc, kwargs...)

default_algorithm(::typeof(right_null!), ::Type{A}; trunc = nothing, kwargs...) where {A} =
    isnothing(trunc) ? select_algorithm(right_null!, A, Val(:lq); kwargs...) :
    select_algorithm(right_null!, A, Val(:svd); trunc, kwargs...)

"""
    left_orth_alg(alg::AbstractAlgorithm) -> LeftOrthAlgorithm

Convert an algorithm to a [`LeftOrthAlgorithm`](@ref) wrapper for use with [`left_orth`](@ref).

This function attempts to deduce the appropriate factorization kind (`:qr`, `:polar`, or `:svd`)
from the algorithm type and wraps it in a `LeftOrthAlgorithm`. Custom algorithm types can be
registered by defining:

```julia
MatrixAlgebraKit.left_orth_alg(alg::CustomAlgorithm) = LeftOrthAlgorithm{kind}(alg)
```

where `kind` specifies the factorization backend to use.

See also [`LeftOrthAlgorithm`](@ref), [`left_orth`](@ref).
"""
left_orth_alg(alg::AbstractAlgorithm) = error(
    """
    Unknown or invalid `left_orth` algorithm type `$(typeof(alg))`.
    To register the algorithm type for `left_orth`, define

        MatrixAlgebraKit.left_orth_alg(alg::CustomAlgorithm) = LeftOrthAlgorithm{kind}(alg)

    where `kind` selects the factorization type that will be used.
    By default, this is either `:qr`, `:polar` or `:svd`, to select [`qr_compact!`](@ref),
    [`left_polar!`](@ref), [`svd_compact!`](@ref) or [`svd_trunc!`](@ref) respectively.
    """
)
left_orth_alg(alg::LeftOrthAlgorithm) = alg
left_orth_alg(alg::QRAlgorithms) = LeftOrthViaQR(alg)
left_orth_alg(alg::PolarAlgorithms) = LeftOrthViaPolar(alg)
left_orth_alg(alg::SVDAlgorithms) = LeftOrthViaSVD(alg)
left_orth_alg(alg::TruncatedAlgorithm{<:SVDAlgorithms}) = LeftOrthViaSVD(alg)

"""
    right_orth_alg(alg::AbstractAlgorithm) -> RightOrthAlgorithm

Convert an algorithm to a [`RightOrthAlgorithm`](@ref) wrapper for use with [`right_orth`](@ref).

This function attempts to deduce the appropriate factorization kind (`:lq`, `:polar`, or `:svd`)
from the algorithm type and wraps it in a `RightOrthAlgorithm`. Custom algorithm types can be
registered by defining:

```julia
MatrixAlgebraKit.right_orth_alg(alg::CustomAlgorithm) = RightOrthAlgorithm{kind}(alg)
```

where `kind` specifies the factorization backend to use.

See also [`RightOrthAlgorithm`](@ref), [`right_orth`](@ref).
"""
right_orth_alg(alg::AbstractAlgorithm) = error(
    """
    Unknown or invalid `right_orth` algorithm type `$(typeof(alg))`.
    To register the algorithm type for `right_orth`, define

        MatrixAlgebraKit.right_orth_alg(alg::CustomAlgorithm) = RightOrthAlgorithm{kind}(alg)

    where `kind` selects the factorization type that will be used.
    By default, this is either `:lq`, `:polar` or `:svd`, to select [`lq_compact!`](@ref),
    [`right_polar!`](@ref), [`svd_compact!`](@ref) or [`svd_trunc!`](@ref) respectively.
    """
)
right_orth_alg(alg::RightOrthAlgorithm) = alg
right_orth_alg(alg::LQAlgorithms) = RightOrthViaLQ(alg)
right_orth_alg(alg::PolarAlgorithms) = RightOrthViaPolar(alg)
right_orth_alg(alg::SVDAlgorithms) = RightOrthViaSVD(alg)
right_orth_alg(alg::TruncatedAlgorithm{<:SVDAlgorithms}) = RightOrthViaSVD(alg)

"""
    left_null_alg(alg::AbstractAlgorithm) -> LeftNullAlgorithm

Convert an algorithm to a [`LeftNullAlgorithm`](@ref) wrapper for use with [`left_null`](@ref).

This function attempts to deduce the appropriate factorization kind (`:qr` or `:svd`) from
the algorithm type and wraps it in a `LeftNullAlgorithm`. Custom algorithm types can be
registered by defining:

```julia
MatrixAlgebraKit.left_null_alg(alg::CustomAlgorithm) = LeftNullAlgorithm{kind}(alg)
```

where `kind` specifies the factorization backend to use.

See also [`LeftNullAlgorithm`](@ref), [`left_null`](@ref).
"""
left_null_alg(alg::AbstractAlgorithm) = error(
    """
    Unknown or invalid `left_null` algorithm type `$(typeof(alg))`.
    To register the algorithm type for `left_null`, define

        MatrixAlgebraKit.left_null_alg(alg::CustomAlgorithm) = LeftNullAlgorithm{kind}(alg)

    where `kind` selects the factorization type that will be used.
    By default, this is either `:qr` or `:svd`, to select [`qr_null!`](@ref),
    [`svd_compact!`](@ref) or [`svd_trunc!`](@ref) respectively.
    """
)
left_null_alg(alg::LeftNullAlgorithm) = alg
left_null_alg(alg::QRAlgorithms) = LeftNullViaQR(alg)
left_null_alg(alg::SVDAlgorithms) = LeftNullViaSVD(alg)
left_null_alg(alg::TruncatedAlgorithm{<:SVDAlgorithms}) = LeftNullViaSVD(alg)

"""
    right_null_alg(alg::AbstractAlgorithm) -> RightNullAlgorithm

Convert an algorithm to a [`RightNullAlgorithm`](@ref) wrapper for use with [`right_null`](@ref).

This function attempts to deduce the appropriate factorization kind (`:lq` or `:svd`) from
the algorithm type and wraps it in a `RightNullAlgorithm`. Custom algorithm types can be
registered by defining:

```julia
MatrixAlgebraKit.right_null_alg(alg::CustomAlgorithm) = RightNullAlgorithm{kind}(alg)
```

where `kind` specifies the factorization backend to use.

See also [`RightNullAlgorithm`](@ref), [`right_null`](@ref).
"""
right_null_alg(alg::AbstractAlgorithm) = error(
    """
    Unknown or invalid `right_null` algorithm type `$(typeof(alg))`.
    To register the algorithm type for `right_null`, define

        MatrixAlgebraKit.right_null_alg(alg::CustomAlgorithm) = RightNullAlgorithm{kind}(alg)

    where `kind` selects the factorization type that will be used.
    By default, this is either `:lq` or `:svd`, to select [`lq_null!`](@ref),
    [`svd_compact!`](@ref) or [`svd_trunc!`](@ref) respectively.
    """
)
right_null_alg(alg::RightNullAlgorithm) = alg
right_null_alg(alg::LQAlgorithms) = RightNullViaLQ(alg)
right_null_alg(alg::SVDAlgorithms) = RightNullViaSVD(alg)
right_null_alg(alg::TruncatedAlgorithm{<:SVDAlgorithms}) = RightNullViaSVD(alg)
