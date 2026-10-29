# Batched EIGH functions
# ----------------------
"""
    batched_eigh_full(A; kwargs...) -> Ds, Vs
    batched_eigh_full(A, alg::AbstractAlgorithm) -> Ds, Vs
    batched_eigh_full!(A, [DV]; kwargs...) -> Ds, Vs
    batched_eigh_full!(A, [DV], alg::AbstractAlgorithm) -> Ds, Vs

Compute the *batched* full Hermitian eigenvalue decompositions (EIGH) of the square
matrices `A` of size `(n, n)`, such that `A[:, :, i] * V[:, :, i] = V[:, :, i] * Diagonal(D[:, i])`.
Here, `V[:, :, i]` are unitary matrices of size `(n, n)`, and `D` is a real matrix of size
`(n, batch_size)` holding the eigenvalues. For a vector of matrices `A`, the outputs are
vectors of `Diagonal` eigenvalue matrices and eigenvector matrices instead.

!!! note
    The bang method `batched_eigh_full!` optionally accepts the output structure and
    possibly destroys the input matrices `A`. Always use the return value of the function
    as it may not always be possible to use the provided `DV` as output.

See also [`batched_eigh_vals(!)`](@ref batched_eigh_vals).
"""
@functiondef batched_eigh_full

"""
    batched_eigh_vals(A; kwargs...) -> Ds
    batched_eigh_vals(A, alg::AbstractAlgorithm) -> Ds
    batched_eigh_vals!(A, [D]; kwargs...) -> Ds
    batched_eigh_vals!(A, [D], alg::AbstractAlgorithm) -> Ds

Compute the *batched* vector of eigenvalues of `A`, such that for an `n x n x batch_size`
array `A`, `D[:, i]` is a vector of size `n` and contains the eigenvalues of `A[:, :, i]`.

See also [`batched_eigh_full(!)`](@ref batched_eigh_full).
"""
@functiondef batched_eigh_vals

# Algorithm selection
# -------------------
for f in (:batched_eigh_full!, :batched_eigh_vals!)
    @eval function default_algorithm(::typeof($f), ::Type{A}; kwargs...) where {A}
        return default_eigh_algorithm(A; kwargs...)
    end
end
default_eigh_algorithm(::Type{V}; kwargs...) where {M <: AbstractMatrix, V <: AbstractVector{M}} = default_eigh_algorithm(M; kwargs...)
