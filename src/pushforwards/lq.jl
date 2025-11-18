function lq_pushforward!(dA, A, LQ, dLQ; tol::Real = default_pullback_gauge_atol(LQ[1]), rank_atol::Real = tol, gauge_atol::Real = tol)
    return qr_pushforward!(adjoint(dA), adjoint(A), adjoint.(reverse(LQ)), adjoint.(reverse(dLQ)); tol, rank_atol, gauge_atol)
end

function lq_null_pushforward!(dA, A, Nᴴ, dNᴴ; tol::Real = default_pullback_gauge_atol(Nᴴ), rank_atol::Real = tol, gauge_atol::Real = tol)
    return iszero(min(size(Nᴴ)...)) && return # nothing to do
end
