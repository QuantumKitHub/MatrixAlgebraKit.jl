# LQ functions
# -------------
"""
    lq_full(A; kwargs...) -> L, Q
    lq_full(A, alg::AbstractAlgorithm) -> L, Q
    lq_full!(A, [LQ]; kwargs...) -> L, Q
    lq_full!(A, [LQ], alg::AbstractAlgorithm) -> L, Q

Compute the full LQ decomposition of the rectangular matrix `A`, such that `A = L * Q`
where `Q` is a square unitary matrix with the same number of rows as `A` and `L` is a
lower triangular matrix with the same size as `A`.

!!! note
    The bang method `lq_full!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `LQ` as output.

See also [`lq_compact(!)`](@ref lq_compact).
"""
@functiondef lq_full

"""
    lq_compact(A; kwargs...) -> L, Q
    lq_compact(A, alg::AbstractAlgorithm) -> L, Q
    lq_compact!(A, [LQ]; kwargs...) -> L, Q
    lq_compact!(A, [LQ], alg::AbstractAlgorithm) -> L, Q

Compute the compact LQ decomposition of the rectangular matrix `A` of size `(m,n)`,
such that `A = L * Q` where the matrix `Q` of size `(min(m,n), n)` has orthogonal rows
spanning the image of `A'`, and the matrix `L` of size `(m, min(m,n))` is lower triangular.

!!! note
    The bang method `lq_compact!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `LQ` as output.

!!! note
    The compact QR decomposition is equivalent to the full LQ decomposition when `m >= n`.
    Some algorithms may require `m <= n`.

See also [`lq_full(!)`](@ref lq_full).
"""
@functiondef lq_compact

"""
    lq_null(A; kwargs...) -> Nᴴ
    lq_null(A, alg::AbstractAlgorithm) -> Nᴴ
    lq_null!(A, [Nᴴ]; kwargs...) -> Nᴴ
    lq_null!(A, [Nᴴ], alg::AbstractAlgorithm) -> Nᴴ

For a (m, n) matrix A, compute the matrix `Nᴴ` corresponding the final `n - min(m, n)` rows 
oft the unitary `Q` factor in the full LQ decomposition of `A`, i.e. the rows that are not
present in the `Q` factor of the compact LQ decomposition. The matrix `Nᴴ` is such that the
isometric matrix `N = adjoint(Nᴴ)` contains an orthonormal basis for the kernel (null space)
of `A` as its columns, i.e. `A * N = 0` or thus `A * adjoint(Nᴴ) = 0`.

!!! note
    The bang method `lq_null!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `Nᴴ` argument as output.

!!! note
    The matrix `Nᴴ` is empty when `m >= n`.

See also [`qr_full(!)`](@ref lq_full) and [`qr_compact(!)`](@ref lq_compact).
"""
@functiondef lq_null

# Algorithm selection
# -------------------
for f in (:lq_full, :lq_compact, :lq_null)
    f! = Symbol(f, :!)
    @eval begin
        function select_algorithm(::typeof($f), A; kwargs...)
            return select_algorithm($f!, A; kwargs...)
        end
        function select_algorithm(::typeof($f!), A; alg=nothing, kwargs...)
            if alg isa AbstractAlgorithm
                return alg
            elseif alg isa Symbol
                return Algorithm{alg}(; kwargs...)
            else
                isnothing(alg) || throw(ArgumentError("Unknown alg $alg"))
                return default_lq_algorithm(A; kwargs...)
            end
        end
    end
end

# Default to LAPACK 
function default_lq_algorithm(A::StridedMatrix{<:BlasFloat}; kwargs...)
    return LAPACK_HouseholderLQ(; kwargs...)
end
