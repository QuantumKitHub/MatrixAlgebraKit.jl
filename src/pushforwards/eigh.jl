function eigh_pushforward!(
        ΔA, A, DV, ΔDV;
        degeneracy_atol::Real = default_pullback_rank_atol(DV[1]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔDV[2])
    )
    D, V = DV
    ΔD, ΔV = ΔDV
    ΔAV = mul!(ΔV, ΔA, V)
    ∂K = V' * ΔAV
    if !iszerotangent(ΔD)
        diagview(ΔD) .= real.(diagview(∂K))
    end
    if !iszerotangent(ΔV)
        ∂K .*= inv_safe.(transpose(diagview(D)) .- diagview(D), degeneracy_atol)
        ΔV = mul!(ΔV, V, ∂K)
    end
    return (ΔD, ΔV)
end

function eigh_vals_pushforward!(ΔA, A, DV, ΔD; kwargs...)
    return eigh_pushforward!(ΔA, A, DV, (Diagonal(ΔD), nothing); kwargs...)
end
