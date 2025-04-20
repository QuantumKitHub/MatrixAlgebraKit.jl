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
