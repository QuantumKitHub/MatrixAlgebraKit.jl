# Batched EIGH functions
# -------------
"""
    batched_eigh_full(A; kwargs...) -> Ds, Vs
    batched_eigh_full(A, alg::AbstractAlgorithm) -> Ds, Vs
    batched_eigh_full!(A, [DV]; kwargs...) -> Ds, Vs
    batched_eigh_full!(A, [DV], alg::AbstractAlgorithm) -> Ds, Vs

Compute the *batched* full Hermitian eigenvalue decompositions (EIGH) of the square 
matrices `A` of size `(n, n)`, such that `A[:, :, i] = D[:, :, i] * V[:, :, i]`.
Here, `V[:, :, i]` are unitary matrices of size `(n, n)`, and `D[:, :, i]` is
a real diagonal matrix of size `(n, n)`.

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
