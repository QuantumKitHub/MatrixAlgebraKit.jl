function eig_pushforward!(
        ΔA, A, DV, ΔDV;
        degeneracy_atol::Real = default_pullback_rank_atol(DV[1]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔDV[2])
    )
    D, V = DV
    ΔD, ΔV = ΔDV
    iVΔAV = inv(V) * ΔA * V
    if !iszerotangent(ΔD)
        diagview(ΔD) .= diagview(iVΔAV)
    end
    if !iszerotangent(ΔV)
        F = inv_safe.(transpose(diagview(D)) .- diagview(D), degeneracy_atol)
        K̇ = F .* iVΔAV
        mul!(ΔV, V, K̇, 1, 0)
    end
    return ΔDV
end

function eig_vals_pushforward!(ΔA, A, DV, ΔD; kwargs...)
    return eig_pushforward!(ΔA, A, DV, (Diagonal(ΔD), nothing); kwargs...)
end
