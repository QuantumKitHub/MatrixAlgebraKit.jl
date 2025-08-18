function gaugefix!(V::AbstractMatrix)
    for j in axes(V, 2)
        v = view(V, :, j)
        s = conj(sign(_argmaxabs(v)))
        @inbounds v .*= s
    end
    return V
end
