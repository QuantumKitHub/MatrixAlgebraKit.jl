# Square root
# -----------

"""
    squareroot(A; kwargs...) -> sqrtA
    squareroot(A, alg::AbstractAlgorithm) -> sqrtA
    squareroot!(A, [sqrtA]; kwargs...) -> sqrtA
    squareroot!(A, [sqrtA], alg::AbstractAlgorithm) -> sqrtA

Compute the principal square root `sqrtA` of the square matrix `A`, i.e. the square root
whose eigenvalues have nonnegative real part.

The scalar type of the output matches that of the input.
As a consequence, a real matrix with eigenvalues on the negative real axis, for which
the principal square root is complex, leads to a `DomainError`; pass a complex matrix
to obtain the principal value.
For the algorithms that have access to the spectrum, real eigenvalues that are negative within a
tolerance `domain_atol` are treated as rounding artifacts and clamped to zero, so that raising
`domain_atol` accepts more matrices. It defaults to [`default_domain_atol`](@ref), and is
supported by every algorithm except [`MatrixFunctionViaLA`](@ref), which has no access to the
spectrum; see [Domain considerations](@ref sec_matrixfunction_domain).

!!! note
    The bang method `squareroot!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `sqrtA` as output.
"""
@functiondef squareroot

# Algorithm selection
# -------------------
default_squareroot_algorithm(A; kwargs...) = default_squareroot_algorithm(typeof(A); kwargs...)
function default_squareroot_algorithm(T::Type; kwargs...)
    return MatrixFunctionViaSchur(; kwargs...)
end
function default_squareroot_algorithm(::Type{T}; kwargs...) where {T <: Diagonal}
    return DiagonalAlgorithm(; kwargs...)
end

function default_algorithm(::typeof(squareroot!), ::Type{A}; kwargs...) where {A}
    return default_squareroot_algorithm(A; kwargs...)
end
