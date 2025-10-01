# TODO: Consider using zerovector! if using VectorInterface.jl
function zero!(A::AbstractArray)
    A .= zero(eltype(A))
    return A
end

function one!(A::AbstractMatrix)
    length(A) > 0 || return A
    zero!(A)
    diagview(A) .= one(eltype(A))
    return A
end

function uppertriangular!(A::AbstractMatrix)
    Base.require_one_based_indexing(A)
    m, n = size(A)
    for i in 1:n
        r = (i + 1):m
        zero!(view(A, r, i))
    end
    return A
end

function lowertriangular!(A::AbstractMatrix)
    Base.require_one_based_indexing(A)
    m, n = size(A)
    for i in 2:n
        r = 1:min(i - 1, m)
        zero!(view(A, r, i))
    end
    return A
end

@doc """
    hermitianpart(A)
    hermitianpart!(A)

In-place implementation of the Hermitian part of a (square) matrix `A`, defined as `(A + A') / 2`.
For real matrices this is also called the symmetric part of `A`.

See also [`antihermitianpart`](@ref).
""" hermitianpart, hermitianpart!

hermitianpart(A) = hermitianpart!(copy(A))
function hermitianpart!(A::AbstractMatrix)
    Base.require_one_based_indexing(A)
    n = LinearAlgebra.checksquare(A)
    @inbounds for j in 1:n
        A[j, j] = real(A[j, j])
        for i in 1:(j - 1)
            val = (A[i, j] + adjoint(A[j, i])) / 2
            A[i, j] = val
            A[j, i] = adjoint(val)
        end
    end
    return A
end

@doc """
    antihermitianpart(A)
    antihermitianpart!(A)

In-place implementation of the anti-Hermitian part of a (square) matrix `A`, defined as `(A - A') / 2`.
For real matrices this is also called the anti-symmetric part of `A`.

See also [`hermitianpart`](@ref).
""" antihermitianpart, antihermitianpart!

antihermitianpart(A) = antihermitianpart!(copy(A))
function antihermitianpart!(A::AbstractMatrix)
    Base.require_one_based_indexing(A)
    n = LinearAlgebra.checksquare(A)
    @inbounds for j in 1:n
        A[j, j] = imag(A[j, j]) * im
        for i in 1:(j - 1)
            val = (A[i, j] - adjoint(A[j, i])) / 2
            A[i, j] = val
            A[j, i] = -adjoint(val)
        end
    end
    return A
end
