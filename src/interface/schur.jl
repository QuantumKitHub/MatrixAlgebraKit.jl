# Schur API
# ---------
# TODO: export? or not export but mark as public ?
function schur!(A::AbstractMatrix, args...; kwargs...)
    return schur_full!(A, args...; kwargs...)
end
function schur(A::AbstractMatrix, args...; kwargs...)
    return schur_full(A, args...; kwargs...)
end

# Schur functions
# -------------
"""
    schur_full(A; kwargs...) -> T, Z, vals
    schur_full(A, alg::AbstractAlgorithm) -> T, Z, vals
    schur_full!(A, [TZv]; kwargs...) -> T, Z, vals
    schur_full!(A, [TZv], alg::AbstractAlgorithm) -> T, Z, vals

Compute the full Schur decomposition of the square matrix `A`,
such that `A * Z = Z * T`, where the orthogonal or unitary matrix `Z` contains the
Schur vectors and the square matrix `T` is upper triangular (in the complex case)
or quasi-upper triangular (in the real case). The list `vals` contains the (complex-valued)
eigenvalues of `A`, as extracted from the (quasi-)diagonal of `T`.

!!! note
    The bang method `schur_full!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `TZv` as output.
"""
@functiondef schur_full

# TODO: is this useful? Is there any difference with simply `eig_vals`?
"""
    schur_vals(A; kwargs...) -> vals
    schur_vals(A, alg::AbstractAlgorithm) -> vals
    schur_vals!(A, [vals]; kwargs...) -> vals
    schur_vals!(A, [vals], alg::AbstractAlgorithm) -> vals

Compute the list of eigenvalues of `A` by computing the Schur decomposition of `A`.

!!! note
    The bang method `schur_vals!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `vals` as output.

See also [`eig_full(!)`](@ref eig_full) and [`eig_trunc(!)`](@ref eig_trunc).
"""
@functiondef schur_vals

# TODO: partial or truncated schur? Do we ever want or use this?

# Algorithm selection
# -------------------
function default_algorithm(::typeof(schur_full!), ::Type{A};
                           kwargs...) where {A<:YALAPACK.BlasMat}
    return default_algorithm(eig_full!, A; kwargs...)
end
function default_algorithm(::typeof(schur_vals!), ::Type{A};
                           kwargs...) where {A<:YALAPACK.BlasMat}
    return default_algorithm(eig_vals!, A; kwargs...)
end
