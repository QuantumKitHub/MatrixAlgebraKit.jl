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

@doc """
    is_left_isometry(A; isapprox_kwargs...) -> Bool

Test whether a linear map is a left isometry, i.e. `A' * A ≈ I`.
The `isapprox_kwargs` can be used to control the tolerances of the equality.

See also [`isisometry`](@ref) and [`is_right_isometry`](@ref).
""" is_left_isometry

function is_left_isometry(A::AbstractMatrix; isapprox_kwargs...)
    return isapprox(A' * A, LinearAlgebra.I; isapprox_kwargs...)
end

@doc """
    is_right_isometry(A; isapprox_kwargs...) -> Bool

Test whether a linear map is a right isometry, i.e. `A * A' ≈ I`.
The `isapprox_kwargs` can be used to control the tolerances of the equality.

See also [`isisometry`](@ref) and [`is_left_isometry`](@ref).
""" is_right_isometry

function is_right_isometry(A::AbstractMatrix; isapprox_kwargs...)
    return isapprox(A * A', LinearAlgebra.I; isapprox_kwargs...)
end

"""
    ishermitian(A; isapprox_kwargs...)

Test whether a linear map is Hermitian, i.e. `A = A'`.
The `isapprox_kwargs` can be used to control the tolerances of the equality.
"""
function ishermitian(A; atol::Real = 0, rtol::Real = 0, kwargs...)
    return (atol == rtol == 0) ? (A == A') : isapprox(A, A'; atol, rtol, kwargs...)
end
function ishermitian(A::AbstractMatrix; atol::Real = 0, rtol::Real = 0, norm = LinearAlgebra.norm, kwargs...)
    Base.require_one_based_indexing(A)
    m, n = size(A)
    m == n || return false
    if atol == rtol == 0
        for j in 1:n
            for i in 1:j
                A[i, j] == adjoint(A[j, i]) || return false
            end
        end
    elseif norm === LinearAlgebra.norm
        atol = max(atol, rtol * norm(A))
        for j in 1:n
            for i in 1:j
                isapprox(A[i, j], adjoint(A[j, i]); atol, abs, kwargs...)  || return false
            end
        end
    else
        return isapprox(A, A'; atol, rtol, norm, kwargs...)
    end
    return true
end

"""
    isantihermitian(A; isapprox_kwargs...)

Test whether a linear map is anti-Hermitian, i.e. `A = -A'`.
The `isapprox_kwargs` can be used to control the tolerances of the equality.
"""
function isantihermitian(A; atol::Real = 0, rtol::Real = 0, kwargs...)
    return (atol == 0 & rtol == 0) ? (A == -A') : isapprox(A, -A'; atol, rtol, kwargs...)
end
function isantihermitian(A::AbstractMatrix; atol::Real = 0, rtol::Real = 0, norm = LinearAlgebra.norm, kwargs...)
    Base.require_one_based_indexing(A)
    m, n = size(A)
    m == n || return false
    if atol == rtol == 0
        @inbounds for j in 1:n
            for i in 1:j
                A[i, j] == -adjoint(A[j, i]) || return false
            end
        end
    elseif norm === LinearAlgebra.norm
        atol = max(atol, rtol * norm(A))
        @inbounds for j in 1:n
            for i in 1:j
                isapprox(A[i, j], -adjoint(A[j, i]); atol, abs, kwargs...)  || return false
            end
        end
    else
        return isapprox(A, -A'; atol, rtol, norm, kwargs...)
    end
    return true
end
