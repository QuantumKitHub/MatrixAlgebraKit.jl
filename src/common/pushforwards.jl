function _diagonal_qr_lq_pushforward!(Δa, q, l_or_r, Δq, Δl_or_r, p::Int)
    Base.mightalias(Δa, Δl_or_r) && (Δa = copy(Δa))
    Δa₁ = view(Δa, 1:p)
    q₁ = view(q, 1:p)
    l_or_r₁ = view(l_or_r, 1:p)
    Δq₁ = view(Δq, 1:p)
    Δl_or_r₁ = view(Δl_or_r, 1:p)
    @. Δl_or_r₁ = real(conj(q₁) * Δa₁)
    @. Δq₁ = (Δa₁ - q₁ * Δl_or_r₁) / l_or_r₁
    zero!(view(Δq, (p + 1):length(Δq)))
    zero!(view(Δl_or_r, (p + 1):length(Δl_or_r)))
    return Δq, Δl_or_r
end
