function svd_pushforward!(dA, A, USVᴴ, dUSVᴴ;
        tol::Real = default_pullback_gaugetol(USVᴴ[2]),
        rank_atol::Real = tol,
        degeneracy_atol::Real = tol,
        gauge_atol::Real = tol
    )
    U, S, Vᴴ = USVᴴ
    dU, dS, dVᴴ = dUSVᴴ
    V       = adjoint(Vᴴ)
    UdAV     = U' * dA * V
    copyto!(diagview(dS), diag(real.(UdAV)))
    m, n    = size(A)
    F       = one(eltype(S)) ./ (diagview(S)' .- diagview(S))
    G       = one(eltype(S)) ./ (diagview(S)' .+ diagview(S))
    diagview(F) .= zero(eltype(F))
    invSdiag = zeros(eltype(S), length(diagview(S)))
    for i in 1:length(diagview(S))
        @inbounds invSdiag[i] = inv(diagview(S)[i])
    end
    invS = Diagonal(invSdiag)
    #∂U = U * (F .* (U' * dA * V * S + S * Vᴴ * dA' * U)) + (LinearAlgebra.diagm(ones(eltype(U), m)) - U*U') * dA * V * invS
    #∂V = V * (F .* (S * U' * dA * V + Vᴴ * dA' * U * S)) + (LinearAlgebra.diagm(ones(eltype(V), n)) - V*Vᴴ) * dA' * U * invS
    hUdAV = F .* project_hermitian(UdAV)
    aUdAV = G .* project_antihermitian(UdAV)
    ∂U  = U * (hUdAV + aUdAV)
    ∂U += (LinearAlgebra.diagm(ones(eltype(U), m)) - U*U') * dA * V * invS
    ∂V  = V * (hUdAV - aUdAV)
    ∂V += (LinearAlgebra.diagm(ones(eltype(U), n)) - V*V') * dA' * U * invS
    copyto!(dU, ∂U)
    adjoint!(dVᴴ, ∂V)
    dA .= zero(eltype(A))
    return (dU, dS, dVᴴ)
end
