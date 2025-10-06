function eig_full_fwd(dA, DV, dDV)
    D, V     = DV
    dD, dV   = dDV
    ∂K       = inv(V) * dA * V
    ∂Kdiag   = diagview(∂K)
    dD.diag .= ∂Kdiag
    ∂K     ./= transpose(diagview(D)) .- diagview(D)
    fill!(∂Kdiag, zero(eltype(D)))
    mul!(dV, V, ∂K, 1, 0)
    dA      .= zero(eltype(dA))
    return dDV
end
