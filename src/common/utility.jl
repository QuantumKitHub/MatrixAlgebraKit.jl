function has_equal_storage(A::Diagonal, B::Diagonal)
    return diagview(A) === diagview(B)
end
function has_equal_storage(A::AbstractMatrix, B::AbstractMatrix)
    return A === B
end

function has_equal_storage(A::Diagonal, B::AbstractVector)
    return diagview(A) === B
end
function has_equal_storage(A::AbstractVector, B::Diagonal)
    return A === diagview(B)
end
has_equal_storage(A::AbstractMatrix, B::AbstractVector) = false
has_equal_storage(A::AbstractVector, B::AbstractMatrix) = false
