"""
    gaugefix!(f_eig, V)
    gaugefix!(f_svd, U, Vᴴ)

Fix the residual gauge degrees of freedom in the eigen or singular vectors, that are
obtained from the decomposition performed by `f_eig` or `f_svd`.
This is achieved by ensuring that, in every column of `V` or `U`, the first entry whose
magnitude is within a relative tolerance of the largest magnitude is real and positive
(see [`gauge_index`](@ref)).
""" gaugefix!

# Helper functions
_argmaxabs(x) = reduce(_largest, x; init = zero(eltype(x)))
_largest(x::Real, y::Real) = abs(x) < abs(y) ? y : x
_largest(x::Complex, y::Complex) = abs2(x) < abs2(y) ? y : x

default_gauge_rtol(V) = sqrt(eps(real(float(eltype(V)))))

"""
    gauge_index(V; dims = 1, rtol = sqrt(eps(real(eltype(V)))))

Indices of the entries of `V` used to fix the gauge along dimension `dims`: for every slice, the
first entry whose magnitude is at least `(1 - rtol)` times the largest magnitude in that slice.
"""
function gauge_index(V::AbstractArray; dims::Int = 1, rtol::Real = default_gauge_rtol(V))
    absV = abs.(V)
    thr = maximum(absV; dims) .* (1 - rtol)
    # `findmax` returns the first index among equal maxima (also for GPU arrays)
    return last(findmax(absV .>= thr; dims))
end

function gaugefix!(::typeof(qr_householder!), Q, R, Rd)
    ax = Base.OneTo(length(Rd))
    if !isempty(Q)
        Qf = view(Q, axes(Q, 1), ax)
        Qf .*= sign_safe.(transpose(Rd))
    end
    if !isempty(R)
        Rf = view(R, ax, axes(R, 2))
        Rf .*= conj.(sign_safe.(Rd))
    end
    return Q, R
end

function gaugefix!(::typeof(lq_householder!), L, Q, Ld)
    ax = Base.OneTo(length(Ld))
    if !isempty(Q)
        Qf = view(Q, ax, axes(Q, 2))
        Qf .*= sign_safe.(Ld)
    end
    if !isempty(L)
        Lf = view(L, axes(L, 1), ax)
        Lf .*= conj.(sign_safe.(transpose(Ld)))
    end
    return L, Q
end

function gaugefix!(::Union{typeof(eig_full!), typeof(eigh_full!), typeof(gen_eig_full!)}, V::AbstractMatrix)
    isempty(V) && return V
    signs = sign.(V[gauge_index(V; dims = 1)])
    @. V = V * conj(signs)
    return V
end

# TODO: can do the multiplication @. Vᴴk = signs_t * Vᴴk in one fell swoop
# earlier, or compute signs in the n > m case entirely with reference to V?
# This might require modifying forward-mode AD rules.
function gaugefix!(::typeof(svd_full!), U, Vᴴ)
    m, n = size(U, 2), size(Vᴴ, 1)
    k = min(m, n)
    isempty(U) && return (U, Vᴴ)
    signs = sign.(U[gauge_index(U; dims = 1)])
    @. U = U * conj(signs)
    Vᴴk = view(Vᴴ, axes(Vᴴ, 1)[1:k], :)
    signs_t = transpose(view(signs, :, axes(signs, 2)[1:k]))
    @. Vᴴk = signs_t * Vᴴk
    if n > m
        Vᴴr = view(Vᴴ, axes(Vᴴ, 1)[(k + 1):n], :)
        signs_r = sign.(Vᴴr[gauge_index(Vᴴr; dims = 2)])
        @. Vᴴr = conj(signs_r) * Vᴴr
    end
    return (U, Vᴴ)
end

function gaugefix!(::Union{typeof(svd_compact!), typeof(svd_trunc!)}, U, Vᴴ)
    @assert axes(U, 2) == axes(Vᴴ, 1)
    isempty(U) && return (U, Vᴴ)
    signs = sign.(U[gauge_index(U; dims = 1)])
    signs_t = transpose(signs)
    @. U = U * conj(signs)
    @. Vᴴ = signs_t * Vᴴ
    return (U, Vᴴ)
end

function gaugefix!(::typeof(batched_svd_compact!), U, Vᴴ)
    for (u, vᴴ) in zip(eachslice(U, dims = 3), eachslice(Vᴴ, dims = 3))
        gaugefix!(svd_compact!, u, vᴴ)
    end
    return U, Vᴴ
end

function gaugefix!(::typeof(batched_svd_full!), U, Vᴴ)
    for (u, vᴴ) in zip(eachslice(U, dims = 3), eachslice(Vᴴ, dims = 3))
        gaugefix!(svd_full!, u, vᴴ)
    end
    return U, Vᴴ
end
