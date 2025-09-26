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
