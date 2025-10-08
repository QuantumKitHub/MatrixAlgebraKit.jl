@doc """
    project_hermitian(A; kwargs...)
    project_hermitian(A, alg)
    project_hermitian!(A, [Aₕ]; kwargs...)
    project_hermitian!(A, [Aₕ], alg)

Compute the hermitian part of a (square) matrix `A`, defined as `(A + A') / 2`.
For real matrices this corresponds to the symmetric part of `A`. In the bang method,
the output storage can be provided via the optional argument `Aₕ`; by default it is
equal to `A` and so the input matrix `A` is replaced by its hermitian projection.

See also [`project_antihermitian`](@ref).
"""
@functiondef project_hermitian

@doc """
    project_antihermitian(A; kwargs...)
    project_antihermitian(A, alg)
    project_antihermitian!(A, [Aₐ]; kwargs...)
    project_antihermitian!(A, [Aₐ], alg)

Compute the anti-hermitian part of a (square) matrix `A`, defined as `(A - A') / 2`.
For real matrices this corresponds to the antisymmetric part of `A`. In the bang method,
the output storage can be provided via the optional argument `Aₐ``; by default it is
equal to `A` and so the input matrix `A` is replaced by its antihermitian projection.

See also [`project_hermitian`](@ref).
"""
@functiondef project_antihermitian

@doc """
    project_isometric(A; kwargs...)
    project_isometric(A, alg)
    project_isometric!(A, [W]; kwargs...)
    project_isometric!(A, [W], alg)

Compute the projection of `A` onto the manifold of isometric matrices, i.e. matrices
satisfying `A' * A ≈ I`. This projection is computed via the polar decomposition, i.e.
`W` corresponds to the first return value of `left_polar!`, but avoids computing the
positive definite factor explicitly.

!!! note
    The bang method `project_isometric!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `W` as output.
"""
@functiondef project_isometric

"""
NativeBlocked(; blocksize = 32)

Algorithm type to denote a native blocked algorithm with given `blocksize` for computing
the hermitian or anti-hermitian part of a matrix.
"""
@algdef NativeBlocked
# TODO: multithreaded? numthreads keyword?

default_hermitian_algorithm(A; kwargs...) = default_hermitian_algorithm(typeof(A); kwargs...)
function default_hermitian_algorithm(::Type{A}; kwargs...) where {A <: AbstractMatrix}
    return NativeBlocked(; kwargs...)
end

for f in (:project_hermitian!, :project_antihermitian!)
    @eval function default_algorithm(::typeof($f), ::Type{A}; kwargs...) where {A}
        return default_hermitian_algorithm(A; kwargs...)
    end
end

default_algorithm(::typeof(project_isometric!), ::Type{A}; kwargs...) where {A} =
    default_polar_algorithm(A; kwargs...)
