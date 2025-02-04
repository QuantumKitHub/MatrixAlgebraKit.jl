# QR functions
# -------------
"""
    qr_full(A; kwargs...) -> Q, R
    qr_full(A, alg::AbstractAlgorithm) -> Q, R
    qr_full!(A, [QR]; kwargs...) -> Q, R
    qr_full!(A, [QR], alg::AbstractAlgorithm) -> Q, R

Compute the full QR decomposition of the rectangular matrix `A`, such that `A = Q * R`
where `Q` is a square unitary matrix with the same number of rows as `A` and `R` is an
upper triangular matrix with the same size as `A`.

!!! note
    The bang method `qr_full!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `QR` as output.

See also [`qr_compact(!)`](@ref qr_compact).
"""
@functiondef qr_full

"""
    qr_compact(A; kwargs...) -> Q, R
    qr_compact(A, alg::AbstractAlgorithm) -> Q, R
    qr_compact!(A, [QR]; kwargs...) -> Q, R
    qr_compact!(A, [QR], alg::AbstractAlgorithm) -> Q, R

Compute the compact QR decomposition of the rectangular matrix `A` of size `(m,n)`,
such that `A = Q * R` where the isometric matrix `Q` of size `(m, min(m,n))` has
orthogonal columns spanning the image of `A`, and the matrix `R` of size `(min(m,n), n)`
is upper triangular.

!!! note
    The bang method `qr_compact!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `QR` as output.

!!! note
    The compact QR decomposition is equivalent to the full QR decomposition when `m >= n`.
    Some algorithms may require `m >= n`.

See also [`qr_full(!)`](@ref qr_full).
"""
@functiondef qr_compact

# Algorithm selection
# -------------------
for f in (:qr_full, :qr_compact)
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
                return default_qr_algorithm(A; kwargs...)
            end
        end
    end
end

# Default to LAPACK 
function default_qr_algorithm(A::StridedMatrix{<:BlasFloat}; kwargs...)
    return LAPACK_HouseholderQR(; kwargs...)
end
