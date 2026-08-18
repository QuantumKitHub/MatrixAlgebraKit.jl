"""
    iszerotangent(x)

Return true if `x` is of a type that the different AD engines use to communicate
a (co)tangent that is identically zero. By overloading this method, and writing
pullback definitions in term of it, we will be able to hook into different AD
ecosystems
"""
function iszerotangent end

iszerotangent(::Any) = false
iszerotangent(::Nothing) = true

# fallback
_sylvester(A, B, C) = LinearAlgebra.sylvester(A, B, C)

"""
    select_indices(r::AbstractRange, ind)

Compute `r[ind]` without iterating over `ind`, so that this also works for an `ind` that
lives on a device.
"""
select_indices(r::AbstractRange, ind) = r[ind]
select_indices(r::AbstractRange, ind::AbstractRange{<:Integer}) = r[ind]
function select_indices(r::AbstractRange, ind::AbstractVector{<:Integer})
    checkbounds(r, ind)
    return first(r) .+ step(r) .* (ind .- 1)
end

"""
    is_leading_index(ind, p::Int)

Check whether `ind` selects the first `p` values in order, i.e. whether `ind == 1:p`, without
iterating over `ind`, so that this also works for an `ind` that lives on a device.
"""
is_leading_index(ind::AbstractRange, p::Int) = ind == 1:p
is_leading_index(ind::AbstractVector, p::Int) = length(ind) == p && all(ind .== 1:p)
