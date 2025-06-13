"""
    isisometry(A; side=:left, isapprox_kwargs...) -> Bool

Test whether a linear map is an isometry, where the type of isometry is controlled by `kind`:

- `side = :left` : `A' * A ≈ I`. 
- `side = :right` : `A * A` ≈ I`.

The `isapprox_kwargs` are passed on to `isapprox` to control the tolerances.

New specializations should overload [`is_left_isometry`](@ref) and [`is_right_isometry`](@ref).

See also [`isunitary`](@ref).
"""
function isisometry(A; side::Symbol=:left, isapprox_kwargs...)
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
