# Logarithm
# ---------

"""
    logarithm(A; kwargs...) -> logA
    logarithm(A, alg::AbstractAlgorithm) -> logA
    logarithm!(A, [logA]; kwargs...) -> logA
    logarithm!(A, [logA], alg::AbstractAlgorithm) -> logA

Compute the principal logarithm `logA` of the square matrix `A`, i.e. the logarithm
whose eigenvalues have imaginary part in `(-π, π]`.

The scalar type of the output matches that of the input.
As a consequence, a real matrix with eigenvalues on the negative real axis, for which
the principal logarithm is complex, leads to a `DomainError`; pass a complex matrix
to obtain the principal value.
A matrix with (numerically) zero eigenvalues has no logarithm and also leads to a
`DomainError`.
Both checks use a tolerance `domain_atol`, which can be specified for the algorithms
that support it and defaults to [`default_domain_atol`](@ref).

!!! note
    The bang method `logarithm!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `logA` as output.
"""
@functiondef logarithm

# Algorithm selection
# -------------------
default_logarithm_algorithm(A; kwargs...) = default_logarithm_algorithm(typeof(A); kwargs...)
function default_logarithm_algorithm(T::Type; kwargs...)
    return MatrixFunctionViaLA(; kwargs...)
end
function default_logarithm_algorithm(::Type{T}; kwargs...) where {T <: Diagonal}
    return DiagonalAlgorithm(; kwargs...)
end

function default_algorithm(::typeof(logarithm!), ::Type{A}; kwargs...) where {A}
    return default_logarithm_algorithm(A; kwargs...)
end
