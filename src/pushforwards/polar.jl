function left_polar_pushforward!(ΔA, A, WP, ΔWP; kwargs...)
    W, P = WP
    ΔW, ΔP = ΔWP
    WᴴdA = adjoint(W) * ΔA
    K̇ = _sylvester(P, P, -(WᴴdA - adjoint(WᴴdA)))
    dAiP = ΔA * inv(P)
    WᴴdAiP = W' * dAiP
    L̇ = mul!(dAiP, W, WᴴdAiP, -1, +1)
    ΔW .= W * K̇ + L̇
    ΔP .= WᴴdA - K̇ * P
    return (ΔW, ΔP)
end

function right_polar_pushforward!(ΔA, A, PWᴴ, ΔPWᴴ; kwargs...)
    P, Wᴴ = PWᴴ
    ΔP, ΔWᴴ = ΔPWᴴ
    dAW = ΔA * adjoint(Wᴴ)
    K̇ = _sylvester(P, P, -(dAW - adjoint(dAW)))
    iPdA = inv(P) * ΔA
    iPdAW = iPdA * Wᴴ'
    L̇ = mul!(iPdA, iPdAW, Wᴴ, -1, +1)
    ΔWᴴ .= K̇ * Wᴴ + L̇
    ΔP .= dAW - P * K̇
    return (ΔWᴴ, ΔP)
end
