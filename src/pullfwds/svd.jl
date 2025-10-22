function svd_pullfwd!(dA, A, USVᴴ, dUSVᴴ; kwargs...)
    U, S, Vᴴ = USVᴴ
    dU, dS, dVᴴ = dUSVᴴ
    V       = adjoint(Vᴴ)
    copyto!(dS.diag, diag(real.(U' * dA * V)))
    m, n    = size(A)
    F       = one(eltype(S)) ./ (diagview(S)' .- diagview(S))
    G       = one(eltype(S)) ./ (diagview(S)' .+ diagview(S))
    diagview(F) .= zero(eltype(F))
    invSdiag = zeros(eltype(S), length(S.diag))
    for i in 1:length(S.diag)
        @inbounds invSdiag[i] = inv(diagview(S)[i])
    end
    invS = Diagonal(invSdiag)
    ∂U = U * (F .* (U' * dA * V * S + S * Vᴴ * dA' * U)) + (diagm(ones(eltype(U), m)) - U*U') * dA * V * invS
    ∂V = V * (F .* (S * U' * dA * V + Vᴴ * dA' * U * S)) + (diagm(ones(eltype(V), n)) - V*Vᴴ) * dA' * U * invS
    copyto!(dU, ∂U)
    adjoint!(dVᴴ, ∂V)
    dA .= zero(eltype(A))
    return (dU, dS, dVᴴ)
end
