@doc """
    project_hermitian(A; kwargs...)
    project_hermitian(A, alg)
    project_hermitian!(A; kwargs...)
    project_hermitian!(A, alg)

Compute the hermitian part of a (square) matrix `A`, defined as `(A + A') / 2`.
For real matrices this corresponds to the symmetric part of `A`.

See also [`project_antihermitian`](@ref).
"""
@functiondef project_hermitian

@doc """
    project_antihermitian(A; kwargs...)
    project_antihermitian(A, alg)
    project_antihermitian!(A; kwargs...)
    project_antihermitian!(A, alg)

Compute the anti-hermitian part of a (square) matrix `A`, defined as `(A - A') / 2`.
For real matrices this corresponds to the antisymmetric part of `A`.

See also [`project_hermitian`](@ref).
"""
@functiondef project_antihermitian

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
