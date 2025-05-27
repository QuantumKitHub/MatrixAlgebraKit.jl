# Polar API
# ---------
function polar!(A::AbstractMatrix, args...; kwargs...)
    return left_polar!(A, args...; kwargs...)
end
function polar(A::AbstractMatrix, args...; kwargs...)
    return left_polar(A, args...; kwargs...)
end

# Polar functions
# ---------------
"""
    left_polar(A; kwargs...) -> W, P
    left_polar(A, alg::AbstractAlgorithm) -> W, P
    left_polar!(A, [WP]; kwargs...) -> W, P
    left_polar!(A, [WP], alg::AbstractAlgorithm) -> W, P

Compute the full polar decomposition of the rectangular matrix `A` of size `(m, n)`
with `m >= n`, such that `A = W * P`. Here, `W` is an isometric matrix (orthonormal columns)
of size `(m, n)`, whereas `P` is a positive (semi)definite matrix of size `(n, n)`.

!!! note
    The bang method `left_polar!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `WP` as output.

See also [`right_polar(!)`](@ref right_polar).
"""
@functiondef left_polar

"""
    right_polar(A; kwargs...) -> P, Wᴴ
    right_polar(A, alg::AbstractAlgorithm) -> P, Wᴴ
    right_polar!(A, [PWᴴ]; kwargs...) -> P, Wᴴ
    right_polar!(A, [PWᴴ], alg::AbstractAlgorithm) -> P, Wᴴ

Compute the full polar decomposition of the rectangular matrix `A` of size `(m, n)`
with `n >= m`, such that `A = P * Wᴴ`. Here, `P` is a positive (semi)definite matrix
of size `(m, m)`, whereas `Wᴴ` is a matrix with orthonormal rows (its adjoint is isometric)
of size `(n, m)`.

!!! note
    The bang method `right_polar!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `WP` as output.

See also [`left_polar(!)`](@ref left_polar).
"""
@functiondef right_polar

"""
    PolarViaSVD(svdalg)
    
Algorithm for computing the polar decomposition of a matrix `A` via the singular value
decomposition (SVD) of `A`. The `svdalg` argument specifies the SVD algorithm to use.
"""
struct PolarViaSVD{SVDAlg} <: AbstractAlgorithm
    svdalg::SVDAlg
end

# Algorithm selection
# -------------------
function default_algorithm(::typeof(left_polar!), ::Type{A};
                           kwargs...) where {A<:YALAPACK.BlasMat}
    return PolarViaSVD(default_algorithm(svd_compact!, A; kwargs...))
        end
function default_algorithm(::typeof(right_polar!), ::Type{A};
                           kwargs...) where {A<:YALAPACK.BlasMat}
    return PolarViaSVD(default_algorithm(svd_compact!, A; kwargs...))
end
