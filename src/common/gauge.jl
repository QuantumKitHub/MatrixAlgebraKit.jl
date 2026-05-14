"""
    gaugefix!(f_eig, V)
    gaugefix!(f_svd, U, Vᴴ)

Fix the residual gauge degrees of freedom in the eigen or singular vectors, that are
obtained from the decomposition performed by `f_eig` or `f_svd`.
This is achieved by ensuring that the entry with the largest magnitude in `V` or `U`
is real and positive.
""" gaugefix!

# Helper functions
_argmaxabs(x) = reduce(_largest, x; init = zero(eltype(x)))
_largest(x, y) = abs(x) < abs(y) ? y : x

function gaugefix!(::typeof(qr_householder!), Q, R, Rd)
    ax = Base.OneTo(length(Rd))
    Qf = view(Q, axes(Q, 1), ax)
    Qf .*= sign_safe.(transpose(Rd))
    if !isnothing(R)
        Rf = view(R, ax, axes(R, 2))
        Rf .*= conj.(sign_safe.(Rd))
    end
    return Q, R
end

function gaugefix!(::typeof(lq_householder!), L, Q, Ld)
    ax = Base.OneTo(length(Ld))
    Qf = view(Q, ax, axes(Q, 2))
    Qf .*= sign_safe.(Ld)
    if !isnothing(L)
        Lf = view(L, axes(L, 1), ax)
        Lf .*= conj.(sign_safe.(transpose(Ld)))
    end
    return L, Q
end

function gaugefix!(::Union{typeof(eig_full!), typeof(eigh_full!), typeof(gen_eig_full!)}, V::AbstractMatrix)
    for j in axes(V, 2)
        v = view(V, :, j)
        s = sign(_argmaxabs(v))
        @inbounds v .*= conj(s)
    end
    return V
end

function gaugefix!(::typeof(svd_full!), U, Vᴴ)
    m, n = size(U, 2), size(Vᴴ, 1)
    for j in 1:max(m, n)
        if j <= min(m, n)
            u = view(U, :, j)
            v = view(Vᴴ, j, :)
            s = sign(_argmaxabs(u))
            u .*= conj(s)
            v .*= s
        elseif j <= m
            u = view(U, :, j)
            s = sign(_argmaxabs(u))
            u .*= conj(s)
        else
            v = view(Vᴴ, j, :)
            s = sign(_argmaxabs(v))
            v .*= conj(s)
        end
    end
    return (U, Vᴴ)
end

function gaugefix!(::Union{typeof(svd_compact!), typeof(svd_trunc!)}, U, Vᴴ)
    @assert axes(U, 2) == axes(Vᴴ, 1)
    for j in axes(U, 2)
        u = view(U, :, j)
        v = view(Vᴴ, j, :)
        s = sign(_argmaxabs(u))
        u .*= conj(s)
        v .*= s
    end
    return (U, Vᴴ)
end
