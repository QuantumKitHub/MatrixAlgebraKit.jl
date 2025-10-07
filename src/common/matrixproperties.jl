"""
    isisometry(A; side=:left, isapprox_kwargs...) -> Bool

Test whether a linear map is an isometry, where the type of isometry is controlled by `kind`:

- `side = :left` : `A' * A ≈ I`. 
- `side = :right` : `A * A' ≈ I`.

The `isapprox_kwargs` are passed on to `isapprox` to control the tolerances.

New specializations should overload [`is_left_isometry`](@ref) and [`is_right_isometry`](@ref).

See also [`isunitary`](@ref).
"""
function isisometry(A; side::Symbol = :left, isapprox_kwargs...)
    side === :left && return is_left_isometry(A; isapprox_kwargs...)
    side === :right && return is_right_isometry(A; isapprox_kwargs...)

    throw(ArgumentError(lazy"Invalid isometry side: $side"))
end

"""
    isunitary(A; isapprox_kwargs...)

Test whether a linear map is unitary, i.e. `A * A' ≈ I ≈ A' * A`.
The `isapprox_kwargs` are passed on to `isapprox` to control the tolerances.

See also [`isisometry`](@ref).
"""
function isunitary(A; isapprox_kwargs...)
    return is_left_isometry(A; isapprox_kwargs...) &&
        is_right_isometry(A; isapprox_kwargs...)
end
function isunitary(A::AbstractMatrix; isapprox_kwargs...)
    size(A, 1) == size(A, 2) || return false
    return is_left_isometry(A; isapprox_kwargs...)
end

@doc """
    is_left_isometry(A; isapprox_kwargs...) -> Bool

Test whether a linear map is a left isometry, i.e. `A' * A ≈ I`.
The `isapprox_kwargs` can be used to control the tolerances of the equality.

See also [`isisometry`](@ref) and [`is_right_isometry`](@ref).
""" is_left_isometry

function is_left_isometry(A::AbstractMatrix; atol::Real = 0, rtol::Real = defaulttol(A), norm = LinearAlgebra.norm)
    P = A' * A
    nP = norm(P) # isapprox would use `rtol * max(norm(P), norm(I))`
    diagview(P) .-= 1
    return norm(P) <= max(atol, rtol * nP) # assume that the norm of I is `sqrt(n)`
end

@doc """
    is_right_isometry(A; isapprox_kwargs...) -> Bool

Test whether a linear map is a right isometry, i.e. `A * A' ≈ I`.
The `isapprox_kwargs` can be used to control the tolerances of the equality.

See also [`isisometry`](@ref) and [`is_left_isometry`](@ref).
""" is_right_isometry

function is_right_isometry(A::AbstractMatrix; atol::Real = 0, rtol::Real = defaulttol(A), norm = LinearAlgebra.norm)
    P = A * A'
    nP = norm(P) # isapprox would use `rtol * max(norm(P), norm(I))`
    diagview(P) .-= 1
    return norm(P) <= max(atol, rtol * nP) # assume that the norm of I is `sqrt(n)`
end

"""
    ishermitian(A; isapprox_kwargs...)

Test whether a linear map is Hermitian, i.e. `A = A'`.
The `isapprox_kwargs` can be used to control the tolerances of the equality.
"""
function ishermitian(A; atol::Real = 0, rtol::Real = 0, norm = LinearAlgebra.norm, kwargs...)
    if iszero(atol) && iszero(rtol)
        return ishermitian_exact(A; kwargs...)
    else
        return 2 * norm(project_antihermitian(A; kwargs...)) ≤ max(atol, rtol * norm(A))
    end
end
function ishermitian_exact(A)
    return A == A'
end
function ishermitian_exact(A::StridedMatrix; kwargs...)
    return strided_ishermitian_exact(A, Val(false); kwargs...)
end

"""
    isantihermitian(A; isapprox_kwargs...)

Test whether a linear map is anti-Hermitian, i.e. `A = -A'`.
The `isapprox_kwargs` can be used to control the tolerances of the equality.
"""
function isantihermitian(A; atol::Real = 0, rtol::Real = 0, norm = LinearAlgebra.norm, kwargs...)
    if iszero(atol) && iszero(rtol)
        return isantihermitian_exact(A; kwargs...)
    else
        return 2 * norm(project_hermitian(A; kwargs...)) ≤ max(atol, rtol * norm(A))
    end
end
function isantihermitian_exact(A)
    return A == -A'
end
function isantihermitian_exact(A::StridedMatrix; kwargs...)
    return strided_ishermitian_exact(A, Val(true); kwargs...)
end

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
