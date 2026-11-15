function left_polar_pushforward!(
        ΔA, A, WP, ΔWP; rank_atol::Real = default_pullback_rank_atol(WP[2]), kwargs...
    )
    W, P = WP
    ΔW, ΔP = ΔWP
    D, U = eigh_full(P)
    d = diagview(D)
    mul!(ΔP, adjoint(W), ΔA, +1, 0)
    K̇, = _sylvester_psd(U, d, ΔP - adjoint(ΔP), rank_atol)
    # ΔA * pinv(P)
    ΔAU = ΔA * U
    ΔAU .*= transpose(inv_safe.(d, rank_atol))
    mul!(ΔW, ΔAU, adjoint(U), +1, 0)
    WᴴdAiP = W' * ΔW
    mul!(ΔW, W, WᴴdAiP, -1, +1)
    ΔW = mul!(ΔW, W, K̇, +1, +1)
    ΔP = mul!(ΔP, K̇, P, -1, +1)
    return (ΔW, ΔP)
end

function right_polar_pushforward!(
        ΔA, A, PWᴴ, ΔPWᴴ; rank_atol::Real = default_pullback_rank_atol(PWᴴ[1]), kwargs...
    )
    P, Wᴴ = PWᴴ
    ΔP, ΔWᴴ = ΔPWᴴ
    D, U = eigh_full(P)
    d = diagview(D)
    mul!(ΔP, ΔA, adjoint(Wᴴ), +1, 0)
    K̇, = _sylvester_psd(U, d, ΔP - adjoint(ΔP), rank_atol)
    # pinv(P) * ΔA
    UᴴΔA = U' * ΔA
    UᴴΔA .*= inv_safe.(d, rank_atol)
    mul!(ΔWᴴ, U, UᴴΔA, +1, 0)
    iPdAW = ΔWᴴ * Wᴴ'
    mul!(ΔWᴴ, iPdAW, Wᴴ, -1, +1)
    ΔWᴴ = mul!(ΔWᴴ, K̇, Wᴴ, +1, +1)
    ΔP = mul!(ΔP, P, K̇, -1, +1)
    return (ΔWᴴ, ΔP)
end
