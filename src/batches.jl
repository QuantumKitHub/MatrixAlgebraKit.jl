_isempty_batch(A::AbstractArray{<:Any, 3}) = isempty(A)
_isempty_batch(A::AbstractVector{<:AbstractMatrix}) = all(isempty, A)

# Adjoint of every matrix in a batch, i.e. `dst[:, :, i] = adjoint(src[:, :, i])`.
function batched_adjoint!(dst::AbstractArray{<:Any, 3}, src::AbstractArray{<:Any, 3})
    isempty(dst) && return dst
    permutedims!(dst, src, (2, 1, 3))
    eltype(dst) <: Real || (dst .= conj.(dst))
    return dst
end
function batched_adjoint(A::AbstractArray{<:Any, 3})
    return batched_adjoint!(similar(A, (size(A, 2), size(A, 1), size(A, 3))), A)
end
batched_adjoint(A::AbstractVector{<:AbstractMatrix}) = map(a -> adjoint!(similar(a'), a), A)

# Ragged batches
# --------------

"""
    supports_ragged_batch(f!, alg::AbstractAlgorithm, driver::Driver, T::Type) -> Bool

Whether the algorithm `alg` running on `driver` accepts a *ragged* batch of matrices
of type `T` which do not have uniform size for function `f!`. `false` by default.
"""
supports_ragged_batch(f!, alg::AbstractAlgorithm, driver::Driver, ::Type) = false
supports_ragged_batch(f!, alg::AbstractAlgorithm, ::DefaultDriver, ::Type{TA}) where {TA} =
    supports_ragged_batch(f!, alg, default_driver(alg, TA), TA)

"""
    max_batched_blocksize(alg, driver::Driver, T::Type) -> Int

Largest matrix dimension that the `driver` for the batched version of `alg` accepts for arrays
of type `T`. Larger matrices in a ragged batch are decomposed one at a time instead.
Unlimited by default.
"""
max_batched_blocksize(alg::AbstractAlgorithm, driver::Driver, ::Type) = typemax(Int)
max_batched_blocksize(alg::AbstractAlgorithm, ::DefaultDriver, ::Type{TA}) where {TA} =
    max_batched_blocksize(alg, default_driver(alg, TA), TA)

"""
    supports_pointer_batch(alg, driver::Driver, T::Type) -> Bool

Whether the low-level batched `driver` for `alg` accepts a batch of matrices of type `T`
as an `AbstractVector` of separately allocated matrices. Such a group of matrices is handed
to the driver as a vector of pointers, instead of being copied into one contiguous 3D array.
`false` by default.
"""
supports_pointer_batch(::AbstractAlgorithm, driver::Driver, ::Type) = false
supports_pointer_batch(alg::AbstractAlgorithm, ::DefaultDriver, ::Type{TA}) where {TA} =
    supports_pointer_batch(alg, default_driver(alg, TA), TA)

# Split a ragged batch into batches the driver can handle: matrices of equal size are
# batched together, and, if `pad`, whatever is left over is zero-padded into one more batch.
# Returns the batches as `(indices, (m, n))` pairs, and the indices of the matrices that have
# to be decomposed one at a time.
# TODO: should everything be padded into ONE batch?
function _ragged_batches(A::AbstractVector{<:AbstractMatrix}, alg::AbstractAlgorithm; pad::Bool = true)
    batches = Tuple{Vector{Int}, Tuple{Int, Int}}[]
    rest = Int[]
    isempty(A) && return batches, rest
    driver = get(alg.kwargs, :driver, DefaultDriver())
    batch_size_limit = max_batched_blocksize(alg, driver, typeof(first(A)))
    needs_tall = requires_tall(alg)
    groups = Dict{Tuple{Int, Int}, Vector{Int}}()
    for i in eachindex(A)
        push!(get!(Vector{Int}, groups, size(A[i])), i)
    end
    for ((m, n), inds) in groups
        if length(inds) >= BATCHED_SVD_THRESHOLD && max(m, n) <= batch_size_limit && (!needs_tall || m >= n)
            push!(batches, (inds, (m, n)))
        else
            append!(rest, inds)
        end
    end
    pad || return batches, rest
    # Zero padding leaves the leading `min(m, n)` singular values and vectors of every input
    # untouched. Pad to a square only when the algorithm requires `m ≥ n`
    # (currently only `QRIteration`).
    m = maximum(i -> size(A[i], 1), rest; init = 0)
    n = maximum(i -> size(A[i], 2), rest; init = 0)
    padded = needs_tall ? (max(m, n), max(m, n)) : (m, n)
    if length(rest) >= BATCHED_SVD_THRESHOLD && maximum(padded) <= batch_size_limit
        push!(batches, (rest, padded))
        rest = Int[]
    end
    return batches, rest
end

# Outputs for a batch that `_ragged_pack` produced, which is either a contiguous `(m, n, b)`
# array or, for a pointer-batch driver, a view of `b` equally sized matrices. Either way the
# outputs are packed into the 3D arrays the batched drivers write into.
_packed_output(f!, A::AbstractArray{<:Any, 3}, alg::AbstractAlgorithm) = initialize_output(f!, A, alg)

# Gather `A[inds]` into a single `(m, n, length(inds))` batch, zero-padding where needed.
function _ragged_pack(A::AbstractVector{<:AbstractMatrix}, inds, m::Int, n::Int, alg::AbstractAlgorithm)
    driver = get(alg.kwargs, :driver, DefaultDriver())
    uniform = all(i -> size(A[i]) == (m, n), inds)
    uniform && supports_pointer_batch(alg, driver, typeof(A[first(inds)])) && return view(A, inds)
    # `stack` can't zero-pad
    # On the GPU it falls back to scalar indexing for matrices that are views
    uniform && A isa AbstractVector{<:Array} && return stack(view(A, inds))
    Ab = similar(A[first(inds)], (m, n, length(inds)))
    uniform || zero!(Ab) # only need to zero if matrices are ragged
    for (j, i) in enumerate(inds)
        copyto!(view(Ab, axes(A[i])..., j), A[i])
    end
    return Ab
end
