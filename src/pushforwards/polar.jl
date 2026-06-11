function left_polar_pushforward!(ΔA, A, WP, ΔWP; kwargs...)
    W, P = WP
    ΔW, ΔP = ΔWP
    mul!(ΔP, adjoint(W), ΔA, +1, 0)
    K̇ = _sylvester(P, P, adjoint(ΔP) - ΔP)
    mul!(ΔW, ΔA, inv(P), +1, 0)
    WᴴdAiP = W' * ΔW
    mul!(ΔW, W, WᴴdAiP, -1, +1)
    ΔW = mul!(ΔW, W, K̇, +1, +1)
    ΔP = mul!(ΔP, K̇, P, -1, +1)
    return (ΔW, ΔP)
end

function right_polar_pushforward!(ΔA, A, PWᴴ, ΔPWᴴ; kwargs...)
    P, Wᴴ = PWᴴ
    ΔP, ΔWᴴ = ΔPWᴴ
    mul!(ΔP, ΔA, adjoint(Wᴴ), +1, 0)
    K̇ = _sylvester(P, P, adjoint(ΔP) - ΔP)
    mul!(ΔWᴴ, inv(P), ΔA, +1, 0)
    iPdAW = ΔWᴴ * Wᴴ'
    mul!(ΔWᴴ, iPdAW, Wᴴ, -1, +1)
    ΔWᴴ = mul!(ΔWᴴ, K̇, Wᴴ, +1, +1)
    ΔP = mul!(ΔP, P, K̇, -1, +1)
    return (ΔWᴴ, ΔP)
end
