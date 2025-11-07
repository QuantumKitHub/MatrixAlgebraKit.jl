function eig_pushforward!(ΔA, A, DV, ΔDV; kwargs...)
    D, V     = DV
    ΔD, ΔV   = ΔDV
    iVΔAV    = inv(V) * ΔA * V
    diagview(ΔD) .= diagview(iVΔAV)
    F        = 1 ./ (transpose(diagview(D)) .- diagview(D))
    fill!(diagview(F), zero(eltype(F)))
    K̇        = F .* iVΔAV
    mul!(ΔV, V, K̇, 1, 0)
    zero!(ΔA)
    return ΔDV
end
