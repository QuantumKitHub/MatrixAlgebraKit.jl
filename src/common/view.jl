# diagind: provided by LinearAlgebra.jl
@doc """
    diagview(D)

Return a view of the diagonal elements of a matrix `D`.

See also [`diagonal`](@ref).
""" diagview

diagview(D::Diagonal) = D.diag
diagview(D::AbstractMatrix) = view(D, diagind(D))

@doc """
    diagonal(v)

Construct a diagonal matrix view for the given diagonal vector.

See also [`diagview`](@ref).
""" diagonal

diagonal(v::AbstractVector) = Diagonal(v)

# triangularind
function lowertriangularind(A::AbstractMatrix)
    Base.require_one_based_indexing(A)
    m, n = size(A)
    minmn = min(m, n)
    I = Vector{Int}(undef, div(minmn * (minmn - 1), 2) + minmn * (m - minmn))
    offset = 0
    for j in 1:n
        r = (j + 1):m
        I[offset .- j .+ r] = (j - 1) * m .+ r
        offset += length(r)
    end
    return I
end

function uppertriangularind(A::AbstractMatrix)
    Base.require_one_based_indexing(A)
    m, n = size(A)
    minmn = min(m, n)
    I = Vector{Int}(undef, div(minmn * (minmn - 1), 2) + minmn * (n - minmn))
    offset = 0
    for i in 1:m
        r = (i + 1):n
        I[offset .- i .+ r] = i .+ m .* (r .- 1)
        offset += length(r)
    end
    return I
end
