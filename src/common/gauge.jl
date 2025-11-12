"""
    gaugefix!(f_eig, V)
    gaugefix!(f_svd, U, Vᴴ)

Fix the residual gauge degrees of freedom in the eigen or singular vectors, that are
obtained from the decomposition performed by `f_eig` or `f_svd`.
This is achieved by ensuring that the entry with the largest magnitude in `V` or `U`
is real and positive.
""" gaugefix!

function gaugefix!(V::AbstractMatrix)
    for j in axes(V, 2)
        v = view(V, :, j)
        s = conj(sign(_argmaxabs(v)))
        @inbounds v .*= s
    end
    return V
end

function gaugefix!(::typeof(svd_full!), U, Vᴴ, m::Int, n::Int)
    for j in 1:max(m, n)
        if j <= min(m, n)
            u = view(U, :, j)
            v = view(Vᴴ, j, :)
            s = conj(sign(_argmaxabs(u)))
            u .*= s
            v .*= conj(s)
        elseif j <= m
            u = view(U, :, j)
            s = conj(sign(_argmaxabs(u)))
            u .*= s
        else
            v = view(Vᴴ, j, :)
            s = conj(sign(_argmaxabs(v)))
            v .*= s
        end
    end
    return (U, Vᴴ)
end

function gaugefix!(::typeof(svd_compact!), U, Vᴴ, m::Int, n::Int)
    for j in 1:size(U, 2)
        u = view(U, :, j)
        v = view(Vᴴ, j, :)
        s = conj(sign(_argmaxabs(u)))
        u .*= s
        v .*= conj(s)
    end
    return (U, Vᴴ)
end

function gaugefix!(::typeof(svd_trunc!), U, Vᴴ, m::Int, n::Int)
    for j in 1:min(m, n)
        u = view(U, :, j)
        v = view(Vᴴ, j, :)
        s = conj(sign(_argmaxabs(u)))
        u .*= s
        v .*= conj(s)
    end
    return (U, Vᴴ)
end
