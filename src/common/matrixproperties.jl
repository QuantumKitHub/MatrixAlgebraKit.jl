"""
    isisometric(A; side=:left, isapprox_kwargs...) -> Bool

Test whether a linear map is an isometry, where the type of isometry is controlled by `kind`:

- `side = :left` : `A' * A ≈ I`. 
- `side = :right` : `A * A' ≈ I`.

The `isapprox_kwargs` are passed on to `isapprox` to control the tolerances.

New specializations should overload [`MatrixAlgebraKit.is_left_isometric`](@ref) and
[`MatrixAlgebraKit.is_right_isometric`](@ref).

See also [`isunitary`](@ref).
"""
function isisometric(A; side::Symbol = :left, isapprox_kwargs...)
    side === :left && return is_left_isometric(A; isapprox_kwargs...)
    side === :right && return is_right_isometric(A; isapprox_kwargs...)

    throw(ArgumentError(lazy"Invalid isometry side: $side"))
end

"""
    isunitary(A; isapprox_kwargs...)

Test whether a linear map is unitary, i.e. `A * A' ≈ I ≈ A' * A`.
The `isapprox_kwargs` are passed on to `isapprox` to control the tolerances.

See also [`isisometric`](@ref).
"""
function isunitary(A; isapprox_kwargs...)
    return is_left_isometric(A; isapprox_kwargs...) &&
        is_right_isometric(A; isapprox_kwargs...)
end
function isunitary(A::AbstractMatrix; isapprox_kwargs...)
    size(A, 1) == size(A, 2) || return false
    return is_left_isometric(A; isapprox_kwargs...)
end

@doc """
    is_left_isometric(A; isapprox_kwargs...) -> Bool

Test whether a linear map is a (left) isometry, i.e. `A' * A ≈ I`.
The `isapprox_kwargs` can be used to control the tolerances of the equality.

See also [`isisometric`](@ref) and [`MatrixAlgebraKit.is_right_isometric`](@ref).
""" is_left_isometric

function is_left_isometric(A::AbstractMatrix; atol::Real = 0, rtol::Real = defaulttol(A), norm = LinearAlgebra.norm)
    P = A' * A
    nP = norm(P) # isapprox would use `rtol * max(norm(P), norm(I))`
    diagview(P) .-= 1
    return norm(P) <= max(atol, rtol * nP) # assume that the norm of I is `sqrt(n)`
end

@doc """
    is_right_isometric(A; isapprox_kwargs...) -> Bool

Test whether a linear map is a (right) isometry, i.e. `A * A' ≈ I`.
The `isapprox_kwargs` can be used to control the tolerances of the equality.

See also [`isisometric`](@ref) and [`MatrixAlgebraKit.is_left_isometric`](@ref).
""" is_right_isometric
is_right_isometric(A; kwargs...) = is_left_isometric(A'; kwargs...)

"""
    ishermitian(A; isapprox_kwargs...)

Test whether a linear map is Hermitian, i.e. `A = A'`.
The `isapprox_kwargs` can be used to control the tolerances of the equality.
"""
function ishermitian(A; atol::Real = 0, rtol::Real = 0, kwargs...)
    if iszero(atol) && iszero(rtol)
        return ishermitian_exact(A; kwargs...)
    else
        return ishermitian_approx(A; atol, rtol, kwargs...)
    end
end

ishermitian_exact(A) = A == A'
ishermitian_exact(A::StridedMatrix; kwargs...) = strided_ishermitian_exact(A, Val(false); kwargs...)
ishermitian_exact(A::Diagonal) = diagonal_ishermitian_exact(A, Val(false))

function ishermitian_approx(A; atol, rtol, kwargs...)
    return norm(project_antihermitian(A; kwargs...)) ≤ max(atol, rtol * norm(A))
end
ishermitian_approx(A::StridedMatrix; kwargs...) = strided_ishermitian_approx(A, Val(false); kwargs...)
ishermitian_approx(A::Diagonal; kwargs...) = diagonal_ishermitian_approx(A, Val(false); kwargs...)

"""
    isantihermitian(A; isapprox_kwargs...)

Test whether a linear map is anti-Hermitian, i.e. `A = -A'`.
The `isapprox_kwargs` can be used to control the tolerances of the equality.
"""
function isantihermitian(A; atol::Real = 0, rtol::Real = 0, kwargs...)
    if iszero(atol) && iszero(rtol)
        return isantihermitian_exact(A; kwargs...)
    else
        return isantihermitian_approx(A; atol, rtol, kwargs...)
    end
end
isantihermitian_exact(A) = A == -A'
isantihermitian_exact(A::StridedMatrix; kwargs...) = strided_ishermitian_exact(A, Val(true); kwargs...)
isantihermitian_exact(A::Diagonal) = diagonal_ishermitian_exact(A, Val(true))

function isantihermitian_approx(A; atol, rtol, kwargs...)
    return norm(project_hermitian(A; kwargs...)) ≤ max(atol, rtol * norm(A))
end
isantihermitian_approx(A::StridedMatrix; kwargs...) = strided_ishermitian_approx(A, Val(true); kwargs...)
isantihermitian_approx(A::Diagonal; kwargs...) = diagonal_ishermitian_approx(A, Val(true); kwargs...)

# blocked implementation of exact checks for strided matrices
# -----------------------------------------------------------
function strided_ishermitian_exact(A::AbstractMatrix, anti::Val; blocksize = 32)
    n = size(A, 1)
    for j in 1:blocksize:n
        jb = min(blocksize, n - j + 1)
        _ishermitian_exact_diag(view(A, j:(j + jb - 1), j:(j + jb - 1)), anti) || return false
        for i in 1:blocksize:(j - 1)
            ib = blocksize
            _ishermitian_exact_offdiag(
                view(A, i:(i + ib - 1), j:(j + jb - 1)),
                view(A, j:(j + jb - 1), i:(i + ib - 1)),
                anti
            ) || return false
        end
    end
    return true
end
function _ishermitian_exact_diag(A, ::Val{anti}) where {anti}
    n = size(A, 1)
    @inbounds for j in 1:n
        @simd for i in 1:j
            A[i, j] == (anti ? -adjoint(A[j, i]) : adjoint(A[j, i])) || return false
        end
    end
    return true
end
function _ishermitian_exact_offdiag(Al, Au, ::Val{anti}) where {anti}
    m, n = size(Al) # == reverse(size(Al))
    @inbounds for j in 1:n
        @simd for i in 1:m
            Al[i, j] == (anti ? -adjoint(Au[j, i]) : adjoint(Au[j, i])) || return false
        end
    end
    return true
end

function strided_ishermitian_approx(
        A::AbstractMatrix, anti::Val;
        blocksize = 32, atol::Real = default_hermitian_tol(A), rtol::Real = 0
    )
    n = size(A, 1)
    ϵ² = abs2(zero(eltype(A)))
    ϵ²max = oftype(ϵ², rtol > 0 ? max(atol, rtol * norm(A)) : atol)^2
    for j in 1:blocksize:n
        jb = min(blocksize, n - j + 1)
        ϵ² += _ishermitian_approx_diag(view(A, j:(j + jb - 1), j:(j + jb - 1)), anti)
        ϵ² < ϵ²max || return false
        for i in 1:blocksize:(j - 1)
            ib = blocksize
            ϵ² += 2 * _ishermitian_approx_offdiag(
                view(A, i:(i + ib - 1), j:(j + jb - 1)),
                view(A, j:(j + jb - 1), i:(i + ib - 1)),
                anti
            )
            ϵ² < ϵ²max || return false
        end
    end
    return true
end

function _ishermitian_approx_diag(A, ::Val{anti}) where {anti}
    n = size(A, 1)
    ϵ² = abs2(zero(eltype(A)))
    @inbounds for j in 1:n
        @simd for i in 1:j
            val = _project_hermitian(A[i, j], A[j, i], !anti)
            ϵ² += abs2(val) * (1 + Int(i < j))
        end
    end
    return ϵ²
end
function _ishermitian_approx_offdiag(Al, Au, ::Val{anti}) where {anti}
    m, n = size(Al) # == reverse(size(Al))
    ϵ² = abs2(zero(eltype(Al)))
    @inbounds for j in 1:n
        @simd for i in 1:m
            val = _project_hermitian(Al[i, j], Au[j, i], !anti)
            ϵ² += abs2(val)
        end
    end
    return ϵ²
end

diagonal_ishermitian_exact(A, ::Val{anti}) where {anti} = all(iszero ∘ (anti ? real : imag), diagview(A))

function diagonal_ishermitian_approx(
        A, ::Val{anti}; atol::Real = default_hermitian_tol(A), rtol::Real = 0
    ) where {anti}
    m, n = size(A)
    m == n || throw(DimensionMismatch())
    init = abs2(zero(eltype(A)))
    ϵ² = sum(abs2 ∘ (anti ? real : imag), diagview(A); init)
    ϵ²max = oftype(ϵ², rtol > 0 ? max(atol, rtol * norm(A)) : atol)^2
    return ϵ² ≤ ϵ²max
end
