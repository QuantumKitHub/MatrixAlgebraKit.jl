# Exponential functions
# --------------

"""
    exponential(A; kwargs...) -> expA
    exponential(A, alg::AbstractAlgorithm) -> expA
    exponential!(A, [expA]; kwargs...) -> expA
    exponential!(A, [expA], alg::AbstractAlgorithm) -> expA

Compute the exponential of the square matrix `A`,

!!! note
    The bang method `exponential!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `expA` as output.

See also [`exponentiali(!)`](@ref exponentiali).
"""
@functiondef exponential

"""
    exponentiali(τ, A; kwargs...) -> expiτA
    exponentiali(τ, A, alg::AbstractAlgorithm) -> expiτA
    exponentiali!(τ, A, [expiτA]; kwargs...) -> expiτA
    exponentiali!(τ, A, [expiτA], alg::AbstractAlgorithm) -> expiτA

Compute the exponential of `i*τ*A`, where `i` is the imaginary unit, `τ` is a scalar, and `A` is a square matrix.
This allows the user to use the hermitian eigendecomposition when `A` is hermitian, even when `i*τ*A` is not.

!!! note
    The bang method `exponentiali!` optionally accepts the output structure and
    possibly destroys the input matrix `A`.
    Always use the return value of the function as it may not always be
    possible to use the provided `expiτA` as output.

See also [`exponential(!)`](@ref exponential).
"""
@functiondef n_args = 2 exponentiali

# Algorithm selection
# -------------------
default_exponential_algorithm(A; kwargs...) = default_exponential_algorithm(typeof(A); kwargs...)
function default_exponential_algorithm(T::Type; kwargs...)
    return MatrixFunctionViaLA(; kwargs...)
end
function default_exponential_algorithm(::Type{T}; kwargs...) where {T <: Diagonal}
    return DiagonalAlgorithm(; kwargs...)
end

for f in (:exponential!,)
    @eval function default_algorithm(::typeof($f), ::Type{A}; kwargs...) where {A}
        return default_exponential_algorithm(A; kwargs...)
    end
end

for f in (:exponentiali!,)
    @eval function default_algorithm(::typeof($f), ::Tuple{A, B}; kwargs...) where {A, B}
        return default_exponential_algorithm(B; kwargs...)
    end
end
