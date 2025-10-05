@doc """
    hermitianpart(A; kwargs...)
    hermitianpart(A, alg)
    hermitianpart!(A; kwargs...)
    hermitianpart!(A, alg)

Compute the hermitian part of a (square) matrix `A`, defined as `(A + A') / 2`.
For real matrices this corresponds to the symmetric part of `A`.

See also [`antihermitianpart`](@ref).
"""
@functiondef hermitianpart

@doc """
    antihermitianpart(A; kwargs...)
    antihermitianpart(A, alg)
    antihermitianpart!(A; kwargs...)
    antihermitianpart!(A, alg)

Compute the anti-hermitian part of a (square) matrix `A`, defined as `(A - A') / 2`.
For real matrices this corresponds to the antisymmetric part of `A`.

See also [`hermitianpart`](@ref).
"""
@functiondef antihermitianpart

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

for f in (:hermitianpart!, :antihermitianpart!)
    @eval function default_algorithm(::typeof($f), ::Type{A}; kwargs...) where {A}
        return default_hermitian_algorithm(A; kwargs...)
    end
end
