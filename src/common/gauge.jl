function gaugefix!(V::AbstractMatrix)
    for j in axes(V, 2)
        v = view(V, :, j)
        s = conj(sign(argmax(abs, v)))
        @inbounds v .*= s
    end
    return V
end

function gaugefix!(::Val{:full}, U, S, Vᴴ, m::Int, n::Int)
    for j in 1:max(m, n)
        if j <= min(m, n)
            u = view(U, :, j)
            v = view(Vᴴ, j, :)
            s = conj(sign(argmax(abs, u)))
            u .*= s
            v .*= conj(s)
        elseif j <= m
            u = view(U, :, j)
            s = conj(sign(argmax(abs, u)))
            u .*= s
        else
            v = view(Vᴴ, j, :)
            s = conj(sign(argmax(abs, v)))
            v .*= s
        end
    end
    return (U, S, Vᴴ)
end

function gaugefix!(::Val{:compact}, U, S, Vᴴ, m::Int, n::Int)
    for j in 1:size(U, 2)
        u = view(U, :, j)
        v = view(Vᴴ, j, :)
        s = conj(sign(argmax(abs, u)))
        u .*= s
        v .*= conj(s)
    end
    return (U, S, Vᴴ)
end
