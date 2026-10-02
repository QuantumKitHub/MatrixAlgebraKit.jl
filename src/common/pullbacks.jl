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

"""
    _smith_iteration!(X, Xₙ, G, w, atol, maxiter)

Solve `X = B + G * X * Diagonal(w)` by summing the Neumann series
`X = Σₖ Gᵏ * B * Diagonal(w)ᵏ` by doubling (Smith's method), i.e. by repeatedly adding
`G^(2ʲ) * X * Diagonal(w)^(2ʲ)` to `X` until the norm of that increment drops below `atol`,
for at most `maxiter` steps.

On entry, `X` contains `B`, and it is overwritten with the result. `Xₙ` is used as a buffer,
and `G` and `w` are overwritten. It is assumed that `w` is normalized such that
`maximum(abs, w) == 1`, so that squaring it can only shrink it.
"""
function _smith_iteration!(X, Xₙ, G, w, atol, maxiter)
    Gₙ = similar(G)
    for k in 1:maxiter
        Xₙ = rmul!(mul!(Xₙ, G, X), Diagonal(w))
        if maximum(abs, Xₙ) < atol
            break
        end
        X .+= Xₙ
        if k == maxiter
            @warn "Sylvester iteration did not converge after $k iterations, final norm of X: $(maximum(abs, X))"
            break
        end
        w .= w .^ 2
        Gₙ = mul!(Gₙ, G, G)
        G, Gₙ = Gₙ, G
    end
    return X
end
