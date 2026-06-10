function eigh_pushforward!(
        dA, A, DV, dDV;
        degeneracy_atol::Real = default_pullback_rank_atol(DV[1]),
        gauge_atol::Real = default_pullback_gauge_atol(dDV[2])
    )
    D, V = DV
    dD, dV = dDV
    dAV = mul!(dV, dA, V)
    ∂K = V' * dAV
    ∂Kdiag = diag(∂K)
    if !iszerotangent(dD)
        diagview(dD) .= real.(∂Kdiag)
    end
    if !iszerotangent(dV)
        ∂K .*= inv_safe.(transpose(diagview(D)) .- diagview(D), degeneracy_atol)
        dV = mul!(dV, V, ∂K)
    end
    return (dD, dV)
end

function eigh_vals_pushforward!(ΔA, A, DV, ΔD; kwargs...)
    return eigh_pushforward!(ΔA, A, DV, (Diagonal(ΔD), nothing); kwargs...)
end
