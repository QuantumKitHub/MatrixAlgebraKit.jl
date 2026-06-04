function left_polar_pushforward!(ΔA, A, WP, ΔWP; kwargs...)
    W, P = WP
    ΔW, ΔP = ΔWP
    mul!(ΔP, adjoint(W), ΔA, +1, 0)
    K̇ = _sylvester(P, P, adjoint(ΔP) - ΔP)
    dAiP = ΔA * inv(P)
    WᴴdAiP = W' * dAiP
    L̇ = mul!(dAiP, W, WᴴdAiP, -1, +1)
    ΔW .= W * K̇ + L̇
    ΔP = mul!(ΔP, K̇, P, -1, +1)
    return (ΔW, ΔP)
end

function right_polar_pushforward!(ΔA, A, PWᴴ, ΔPWᴴ; kwargs...)
    P, Wᴴ = PWᴴ
    ΔP, ΔWᴴ = ΔPWᴴ
    mul!(ΔP, ΔA, adjoint(Wᴴ), +1, 0)
    K̇ = _sylvester(P, P, adjoint(ΔP) - ΔP)
    iPdA = inv(P) * ΔA
    iPdAW = iPdA * Wᴴ'
    L̇ = mul!(iPdA, iPdAW, Wᴴ, -1, +1)
    ΔWᴴ .= K̇ * Wᴴ + L̇
    ΔP = mul!(ΔP, P, K̇, -1, +1)
    return (ΔWᴴ, ΔP)
end
