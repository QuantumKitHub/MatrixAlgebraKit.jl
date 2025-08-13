function gaugefix!(V::AbstractMatrix)
    for j in axes(V, 2)
        v = view(V, :, j)
        s = conj(sign(argmax(abs, v)))
        @inbounds v .*= s
    end
    return V
end
