# Simplified API
# --------------
function orth(A::AbstractMatrix; kwargs...)
    return left_orth(A; kwargs...)
end
function null(A::AbstractMatrix; kwargs...)
    return adjoint(right_null(A; kwargs...))
end

# TODO: do we need an advanced interface for the simplified API
function orth!(A::AbstractMatrix, args...; kwargs...)
    return left_orth!(A, args...; kwargs...)
end

function null!(A::AbstractMatrix, args...; kwargs...)
    return right_null!(A, args...; kwargs...)
end

# Orth functions
# --------------
"""
    left_orth(A; [kind::Symbol, atol::Real=0, rtol::Real=0, alg]) -> V, C
    left_orth!(A, [VC]; [kind::Symbol, atol::Real=0, rtol::Real=0, alg]) -> V, C

Compute an orthonormal basis `V` for the image of the matrix `A` of size `(m, n)`,
as well as a  matrix `C` (the corestriction) such that `A` factors as `A = V * C`.
The keyword argument `kind` can be used to specify the specific orthogonal decomposition
that should be used to factor `A`, whereas `atol` and `rtol` can be used to control the
precision in determining the rank of `A` via its singular values.

This is a high-level wrapper and will use one of the decompositions
`qr!`, `svd!`, and `left_polar!` to compute the orthogonal basis `V`, as controlled
by the keyword arguments.

When `kind` is provided, its possible values are

*   `kind == :qrpos`: `V` and `C` are computed using the positive QR decomposition.
    This requires `iszero(atol) && iszero(rtol)` and `left_orth!(A, [VC])` is equivalent to
    `qr_compact!(A, [VC], alg)` with a default value `alg = select_algorithm(qr_compact!, A; positive=true)`

*   `kind == :qr`: `V` and `C` are computed using the QR decomposition,
    This requires `iszero(atol) && iszero(rtol)` and `left_orth!(A, [VC])` is equivalent to
    `qr_compact!(A, [VC], alg)` with a default value `alg = select_algorithm(qr_compact!, A)`

*   `kind == :polar`: `V` and `C` are computed using the polar decomposition,
    This requires `iszero(atol) && iszero(rtol)` and `left_orth!(A, [VC])` is equivalent to
    `left_polar!(A, [VC], alg)` with a default value `alg = select_algorithm(left_polar!, A)`

*   `kind == :svd`: `V` and `C` are computed using the singular value decomposition `svd_trunc!`,
    where `V` will contain the left singular vectors corresponding to the singular values that
    are larger than `max(atol, rtol * σ₁)`, where `σ₁` is the largest singular value of `A`.
    `C` is computed as the product of the singular values and the right singular vectors,
    i.e. with `U, S, Vᴴ = svd_trunc!(A)`, we have `V = U` and `C = S * Vᴴ`.

When `kind` is not provided, the default value is `:qrpos` when `iszero(atol) && iszero(rtol)`
and `:svd` otherwise. Finally, finer control is obtained by providing an explicit algorithm
using the `alg` keyword argument, which should be compatible with the chosen or default value
of `kind`.

!!! note
    The bang method `left_orth!` optionally accepts the output structure and possibly destroys
    the input matrix `A`. Always use the return value of the function as it may not always be
    possible to use the provided `CV` as output.

See also [`right_orth(!)`](@ref right_orth), [`left_null(!)`](@ref left_null), [`right_null(!)`](@ref right_null)
"""
function left_orth end
function left_orth! end
function left_orth!(A::AbstractMatrix; kwargs...)
    return left_orth!(A, initialize_output(left_orth!, A); kwargs...)
end
function left_orth(A::AbstractMatrix; kwargs...)
    return left_orth!(copy_input(left_orth, A); kwargs...)
end

"""
    right_orth(A; [kind::Symbol, atol::Real=0, rtol::Real=0, alg]) -> C, Vᴴ
    right_orth!(A, [CVᴴ]; [kind::Symbol, atol::Real=0, rtol::Real=0, alg]) -> C, Vᴴ

Compute an orthonormal basis `V = adjoint(Vᴴ)` for the coimage of the matrix `A`, i.e.
for the image of `adjoint(A)`, as well as a matrix `C` such that `A = C * Vᴴ`.
The keyword argument `kind` can be used to specify the specific orthogonal decomposition
that should be used to factor `A`, whereas `atol` and `rtol` can be used to control the
precision in determining the rank of `A` via its singular values.

This is a high-level wrapper and will use call one of the decompositions
`qr!`, `svd!`, and `left_polar!` to compute the orthogonal basis `V`, as controlled
by the keyword arguments.

When `kind` is provided, its possible values are

*   `kind == :lqpos`: `C` and `Vᴴ` are computed using the positive QR decomposition.
    This requires `iszero(atol) && iszero(rtol)` and `right_orth!(A, [CVᴴ])` is equivalent to
    `lq_compact!(A, [CVᴴ], alg)` with a default value `alg = select_algorithm(lq_compact!, A; positive=true)`

*   `kind == :lq`: `C` and `Vᴴ` are computed using the QR decomposition,
    This requires `iszero(atol) && iszero(rtol)` and `right_orth!(A, [CVᴴ])` is equivalent to
    `lq_compact!(A, [CVᴴ], alg)` with a default value `alg = select_algorithm(lq_compact!, A))`

*   `kind == :polar`: `C` and `Vᴴ` are computed using the polar decomposition,
    This requires `iszero(atol) && iszero(rtol)` and `right_orth!(A, [CVᴴ])` is equivalent to
    `right_polar!(A, [CVᴴ], alg)` with a default value `alg = select_algorithm(right_polar!, A))`

*   `kind == :svd`: `C` and `Vᴴ` are computed using the singular value decomposition `svd_trunc!`,
    where `V = adjoint(Vᴴ)` will contain the right singular vectors corresponding to the singular
    values that are larger than `max(atol, rtol * σ₁)`, where `σ₁` is the largest singular value of `A`.
    `C` is computed as the product of the singular values and the right singular vectors,
    i.e. with `U, S, Vᴴ = svd_trunc!(A)`, we have `C = rmul!(U, S)` and `Vᴴ = Vᴴ`.

When `kind` is not provided, the default value is `:lqpos` when `iszero(atol) && iszero(rtol)`
and `:svd` otherwise. Finally, finer control is obtained by providing an explicit algorithm
using the `alg` keyword argument, which should be compatible with the chosen or default value
of `kind`.

!!! note
    The bang method `right_orth!` optionally accepts the output structure and possibly destroys
    the input matrix `A`. Always use the return value of the function as it may not always be
    possible to use the provided `CVᴴ` as output.

See also [`left_orth(!)`](@ref left_orth), [`left_null(!)`](@ref left_null), [`right_null(!)`](@ref right_null)
"""
function right_orth end
function right_orth! end
function right_orth!(A::AbstractMatrix; kwargs...)
    return right_orth!(A, initialize_output(right_orth!, A); kwargs...)
end
function right_orth(A::AbstractMatrix; kwargs...)
    return right_orth!(copy_input(right_orth, A); kwargs...)
end

# Null functions
# --------------
"""
    left_null(A; [kind::Symbol, atol::Real=0, rtol::Real=0, alg]) -> N
    left_null!(A, [N]; [kind::Symbol, atol::Real=0, rtol::Real=0, alg]) -> N

Compute an orthonormal basis `N` for the cokernel of the matrix `A` of size `(m, n)`, i.e.
the nullspace of `adjoint(A)`, such that `adjoint(A)*N ≈ 0` and `N'*N ≈ I`.
The keyword argument `kind` can be used to specify the specific orthogonal decomposition
that should be used to factor `A`, whereas `atol` and `rtol` can be used to control the
precision in determining the rank of `A` via its singular values.

This is a high-level wrapper and will use one of the decompositions `qr!` or `svd!`
to compute the orthogonal basis `N`, as controlled by the keyword arguments.

When `kind` is provided, its possible values are

*   `kind == :qrpos`: `N` is computed using the positive QR decomposition.
    This requires `iszero(atol) && iszero(rtol)` and `left_null!(A, [N], kind=:qrpos)` is equivalent to
    `qr_null!(A, [N], alg)` with a default value `alg = select_algorithm(qr_compact!, A; positive=true)`

*   `kind == :qr`: `N` is computed using the (nonpositive) QR decomposition.
    This requires `iszero(atol) && iszero(rtol)` and `left_null!(A, [N], kind=:qr)` is equivalent to
    `qr_null!(A, [N], alg)` with a default value `alg = select_algorithm(qr_compact!, A)`

*   `kind == :svd`: `N` is computed using the singular value decomposition and will contain 
    the left singular vectors corresponding to the singular values that
    are smaller than `max(atol, rtol * σ₁)`, where `σ₁` is the largest singular value of `A`.

When `kind` is not provided, the default value is `:qrpos` when `iszero(atol) && iszero(rtol)`
and `:svd` otherwise. Finally, finer control is obtained by providing an explicit algorithm
using the `alg` keyword argument, which should be compatible with the chosen or default value
of `kind`.

!!! note
    The bang method `left_null!` optionally accepts the output structure and possibly destroys
    the input matrix `A`. Always use the return value of the function as it may not always be
    possible to use the provided `N` as output.

See also [`right_null(!)`](@ref right_null), [`left_orth(!)`](@ref left_orth), [`right_orth(!)`](@ref right_orth)
"""
function left_null end
function left_null! end
function left_null!(A::AbstractMatrix; kwargs...)
    return left_null!(A, initialize_output(left_null!, A); kwargs...)
end
function left_null(A::AbstractMatrix; kwargs...)
    return left_null!(copy_input(left_null, A); kwargs...)
end

"""
    right_null(A; [kind::Symbol, atol::Real=0, rtol::Real=0, alg]) -> Nᴴ
    right_null!(A, [Nᴴ]; [kind::Symbol, atol::Real=0, rtol::Real=0, alg]) -> Nᴴ

Compute an orthonormal basis `N = adjoint(Nᴴ)` for the kernel or nullspace of the matrix `A`
of size `(m, n)`, such that `A*adjoint(Nᴴ) ≈ 0` and `Nᴴ*adjoint(Nᴴ) ≈ I`.
The keyword argument `kind` can be used to specify the specific orthogonal decomposition
that should be used to factor `A`, whereas `atol` and `rtol` can be used to control the
precision in determining the rank of `A` via its singular values.

This is a high-level wrapper and will use one of the decompositions `lq!` or `svd!`
to compute the orthogonal basis `Nᴴ`, as controlled by the keyword arguments.

When `kind` is provided, its possible values are

*   `kind == :lqpos`: `Nᴴ` is computed using the positive LQ decomposition.
    This requires `iszero(atol) && iszero(rtol)` and `right_null!(A, [Nᴴ], kind=:lqpos)` is equivalent to
    `lq_null!(A, [Nᴴ], alg)` with a default value `alg = select_algorithm(lq_compact!, A; positive=true)`

*   `kind == :lq`: `Nᴴ` is computed using the (nonpositive) LQ decomposition.
    This requires `iszero(atol) && iszero(rtol)` and `right_null!(A, [Nᴴ], kind=:lq)` is equivalent to
    `lq_null!(A, [Nᴴ], alg)` with a default value `alg = select_algorithm(lq_compact!, A)`

*   `kind == :svd`: `N` is computed using the singular value decomposition and will contain 
    the left singular vectors corresponding to the singular values that
    are smaller than `max(atol, rtol * σ₁)`, where `σ₁` is the largest singular value of `A`.

When `kind` is not provided, the default value is `:lqpos` when `iszero(atol) && iszero(rtol)`
and `:svd` otherwise. Finally, finer control is obtained by providing an explicit algorithm
using the `alg` keyword argument, which should be compatible with the chosen or default value
of `kind`.

!!! note
    The bang method `right_null!` optionally accepts the output structure and possibly destroys
    the input matrix `A`. Always use the return value of the function as it may not always be
    possible to use the provided `Nᴴ` as output.

See also [`left_null(!)`](@ref left_null), [`left_orth(!)`](@ref left_orth), [`right_orth(!)`](@ref right_orth)
"""
function right_null end
function right_null! end
function right_null!(A::AbstractMatrix; kwargs...)
    return right_null!(A, initialize_output(right_null!, A); kwargs...)
end
function right_null(A::AbstractMatrix; kwargs...)
    return right_null!(copy_input(right_null, A); kwargs...)
end
