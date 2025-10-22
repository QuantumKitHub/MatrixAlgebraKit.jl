function eigh_pushforward!(dA, A, DV, dDV; kwargs...)
    tmpV         = V \ dA
    ∂K           = tmpV * V
    ∂Kdiag       = diag(∂K)
    dD.diag     .= real.(∂Kdiag)
    dDD          = transpose(diagview(D)) .- diagview(D)
    F            = one(eltype(dDD)) ./ dDD
    diagview(F) .= zero(eltype(F))
    ∂K         .*= F
    ∂V           = mul!(tmpV, V, ∂K) 
    copyto!(dV, ∂V)
    dA          .= zero(eltype(A))
    return (dD, dV)
end
