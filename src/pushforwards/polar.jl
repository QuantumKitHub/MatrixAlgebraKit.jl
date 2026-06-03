function left_polar_pushforward!(ΔA, A, WP, ΔWP; kwargs...)
    W, P = WP
    ΔW, ΔP = ΔWP
    WᴴdA = adjoint(W) * ΔA
    K̇ = _sylvester(P, P, -(WᴴdA - adjoint(WᴴdA)))
    L̇ = (LinearAlgebra.UniformScaling(1) - W * adjoint(W)) * ΔA * inv(P)
    ΔW .= W * K̇ + L̇
    ΔP .= WᴴdA - K̇ * P
    return (ΔW, ΔP)
end

function right_polar_pushforward!(ΔA, A, PWᴴ, ΔPWᴴ; kwargs...)
    P, Wᴴ = PWᴴ
    ΔP, ΔWᴴ = ΔPWᴴ
    dAW = ΔA * adjoint(Wᴴ)
    K̇ = _sylvester(P, P, -(dAW - adjoint(dAW)))
    L̇ = inv(P) * ΔA * (LinearAlgebra.UniformScaling(1) - adjoint(Wᴴ) * Wᴴ)
    ΔWᴴ .= K̇ * Wᴴ + L̇
    ΔP .= dAW - P * K̇
    return (ΔWᴴ, ΔP)
end
