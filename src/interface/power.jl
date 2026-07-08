# Power
# -----

"""
    power(A, p::Real; kwargs...) -> powA
    power(A, p::Real, alg::AbstractAlgorithm) -> powA
    power!(A, p::Real, [powA]; kwargs...) -> powA
    power!(A, p::Real, [powA], alg::AbstractAlgorithm) -> powA

Compute the matrix power `powA = A^p` of the square matrix `A`.
For integer `p` this is defined for any square matrix (invertible if `p < 0`);
for fractional `p` the principal power `exp(p * log(A))` is computed.

The scalar type of the output matches that of the input.
As a consequence, for fractional `p`, a real matrix with eigenvalues on the negative
real axis, for which the principal power is complex, leads to a `DomainError`; pass a
complex matrix to obtain the principal value.
Real eigenvalues that are negative within a tolerance `domain_atol` are treated as
rounding artifacts and clamped to zero, where `domain_atol` can be specified for the
algorithms that support it and defaults to [`default_domain_atol`](@ref).
For negative fractional `p`, (numerically) zero eigenvalues also lead to a `DomainError`.

!!! note
    The bang method `power!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `powA` as output.
"""
@functiondef n_args = 2 power

# Algorithm selection
# -------------------
default_power_algorithm(A; kwargs...) = default_power_algorithm(typeof(A); kwargs...)
function default_power_algorithm(T::Type; kwargs...)
    return MatrixFunctionViaLA(; kwargs...)
end
function default_power_algorithm(::Type{T}; kwargs...) where {T <: Diagonal}
    return DiagonalAlgorithm(; kwargs...)
end

function default_algorithm(::typeof(power!), ::Tuple{A, P}; kwargs...) where {A, P}
    return default_power_algorithm(A; kwargs...)
end
