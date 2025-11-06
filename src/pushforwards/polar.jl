function left_polar_pushforward!(ΔA, A, WP, ΔWP; kwargs...)
    W, P   = WP
    ΔW, ΔP = ΔWP
    aWdA   = adjoint(W) * ΔA
    K̇      = sylvester(P, P, -(aWdA - adjoint(aWdA)))
    L̇      = (Diagonal(ones(eltype(W), size(W, 1))) - W*adjoint(W))*ΔA*inv(P)
    ΔW     .= W * K̇ + L̇
    ΔP     .= aWdA - K̇*P
    MatrixAlgebraKit.zero!(ΔA)
    return (ΔW, ΔP)
end

function right_polar_pushforward!(ΔA, A, PWᴴ, ΔPWᴴ; kwargs...)
    P, Wᴴ   = PWᴴ
    ΔP, ΔWᴴ = ΔPWᴴ
    dAW     = ΔA * adjoint(Wᴴ)
    K̇       = sylvester(P, P, -(dAW - adjoint(dAW)))
    ImW     = (Diagonal(ones(eltype(Wᴴ), size(Wᴴ, 2))) - adjoint(Wᴴ) * Wᴴ)
    @show size(P), size(ΔA), size(ImW), size(Wᴴ)
    L̇       = inv(P)*ΔA*ImW
    ΔWᴴ    .= K̇ * Wᴴ + L̇
    ΔP     .= dAW - P * K̇
    MatrixAlgebraKit.zero!(ΔA)
    return (ΔWᴴ, ΔP)
end
