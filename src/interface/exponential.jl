# Exponential functions
# --------------

"""
    exponential(A; kwargs...) -> expA
    exponential(A, alg::AbstractAlgorithm) -> expA
    exponential!(A, [expA]; kwargs...) -> expA
    exponential!(A, [expA], alg::AbstractAlgorithm) -> expA
    exponential((τ, A); kwargs...) -> expτA
    exponential((τ, A), alg::AbstractAlgorithm) -> expτA
    exponential!((τ, A), [expA]; kwargs...) -> expτA
    exponential!((τ, A), [expA], alg::AbstractAlgorithm) -> expτA

Compute the exponential of the square matrix `A` or `τ * A`,

!!! note
    The bang method `exponential!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `expA` as output.
"""
@functiondef exponential

# Algorithm selection
# -------------------
default_exponential_algorithm(A; kwargs...) = default_exponential_algorithm(typeof(A); kwargs...)
function default_exponential_algorithm(T::Type; kwargs...)
    return MatrixFunctionViaTaylor(; kwargs...)
end
function default_exponential_algorithm(::Type{T}; kwargs...) where {T <: Diagonal}
    return DiagonalAlgorithm(; kwargs...)
end

function default_algorithm(::typeof(exponential!), ::Type{A}; kwargs...) where {A}
    return default_exponential_algorithm(A; kwargs...)
end

function default_algorithm(::typeof(exponential!), ::Tuple{A, B}; kwargs...) where {A, B}
    return default_algorithm(exponential!, B; kwargs...)
end

function default_algorithm(::typeof(exponential!), ::Type{Tuple{A, B}}; kwargs...) where {A, B}
    return default_algorithm(exponential!, B; kwargs...)
end
