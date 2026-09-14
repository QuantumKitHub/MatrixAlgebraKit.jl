# Batched SVD functions
# -------------
"""
    batched_svd_full(A; kwargs...) -> Us, Ss, Vᴴs
    batched_svd_full(A, alg::AbstractAlgorithm) -> Us, Ss, Vᴴs
    batched_svd_full!(A, [USVᴴ]; kwargs...) -> Us, Ss, Vᴴs
    batched_svd_full!(A, [USVᴴ], alg::AbstractAlgorithm) -> Us, Ss, Vᴴs

Compute the *batched* full singular value decompositions (SVD) of the rectangular
matrices `A` of size `(m, n)`, such that `A[:, :, i] = U[:, :, i] * S[:, :, i] * Vᴴ[:, :, i]`.
Here, `U[:, :, i]` and `Vᴴ[:, :, i]` are unitary matrices of size
`(m, m)` and `(n, n)` respectively, and `S[:, :, i]` is a diagonal matrix of size `(m, n)`.

!!! note
    The bang method `batched_svd_full!` optionally accepts the output structure and
    possibly destroys the input matrices `A`. Always use the return value of the function
    as it may not always be possible to use the provided `USVᴴ` as output.

See also [`batched_svd_compact(!)`](@ref batched_svd_compact) and
[`batched_svd_vals(!)`](@ref batched_svd_vals).
"""
@functiondef batched_svd_full

"""
    batched_svd_compact(A; kwargs...) -> Us, Ss, Vᴴs
    batched_svd_compact(A, alg::AbstractAlgorithm) -> Us, Ss, Vᴴs
    batched_svd_compact!(A, [USVᴴ]; kwargs...) -> Us, Ss, Vᴴs
    batched_svd_compact!(A, [USVᴴ], alg::AbstractAlgorithm) -> Us, Ss, Vᴴs

Compute the *batched* compact singular value decomposition (SVD) of the rectangular
matrices `A` of size `(m, n)`, such that `A[:, :, i] = U[:, :, i] * S[:, :, i] * Vᴴ[:, :, i]`.
Here, `U[:, :, i]` is an isometric matrix (orthonormal columns) of size `(m, k)`, whereas 
`Vᴴ[:, :, i]` is a matrix of size `(k, n)` with orthonormal rows and `S[:, :, i]`
is a square diagonal matrix of size `(k, k)`, with `k = min(m, n)`.

!!! note
    The bang method `batched_svd_compact!` optionally accepts the output structure and
    possibly destroys the input matrices `A`. Always use the return value of the function
    as it may not always be possible to use the provided `USVᴴ` as output.

See also [`batched_svd_full(!)`](@ref batched_svd_full) and
[`batched_svd_vals(!)`](@ref batched_svd_vals). 
"""
@functiondef batched_svd_compact

"""
    batched_svd_vals(A; kwargs...) -> Ss
    batched_svd_vals(A, alg::AbstractAlgorithm) -> Ss
    batched_svd_vals!(A, [S]; kwargs...) -> Ss
    batched_svd_vals!(A, [S], alg::AbstractAlgorithm) -> Ss

Compute the *batched* vector of singular values of `A`, such that for an M×N matrix `A`,
`S` is a vector of size `K = min(M, N)`, the number of kept singular values.

See also [`batched_svd_full(!)`](@ref batched_svd_full),
[`batched_svd_compact(!)`](@ref batched_svd_compact).
"""
@functiondef batched_svd_vals

# Algorithm selection
# -------------------
for f in (:batched_svd_full!, :batched_svd_compact!, :batched_svd_vals!)
    @eval function default_algorithm(::typeof($f), ::Type{A}; kwargs...) where {A}
        return default_svd_algorithm(A; kwargs...)
    end
end
