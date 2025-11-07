#=function lq_pushforward!(dA, A, LQ, dLQ; tol::Real=MatrixAlgebraKit.default_pullback_gaugetol(LQ[1]), rank_atol::Real=tol, gauge_atol::Real=tol)
    L, Q  = LQ
    dL, dQ = dLQ
    m     = size(L, 1)
    n     = size(Q, 2)
    minmn = min(m, n)
    Ld    = diagview(L)
    p     = findlast(>=(rank_atol) ∘ abs, Ld)

    if p == minmn && size(L,1) == size(L,2) # full-rank
        invL = inv(L)
        dQ .= invL * (dA - dL * Q)
        dL  = invL * dA * Q'
        return (dL, dQ)
    end

    n1 = p
    n2 = minmn - p
    n3 = n - minmn
    m1 = p
    m2 = m - p

    #####
    Q1  = view(Q, 1:m1, 1:n) # full rank portion
    Q2  = view(Q, n1+1:n1+n2, 1:n)
    L11 = view(L, 1:m1, 1:n1)
    L21 = view(L, (m1+1):m, 1:n1)

    dA1 = view(dA, 1:m1, 1:n)
    dA2 = view(dA, (m1+1):m, 1:n)

    dQ1    = view(dQ, 1:n1, 1:n)
    dQ2    = view(dQ, n1+1:n1+n2, 1:n)
    dL11   = view(dL, 1:m1, 1:n1)
    dL21   = view(dL, (m1+1):m, 1:n1)
    dL22   = view(dL, (m1+1):m, n1+1:(n1+n2) )

    # fwd rule for Q1 and R11 -- for a non-rank redeficient QR, this is all we need
    invL11 = inv(L11)
    tmp    = invL11 * dA1 * Q1'
    Ltmp   = tmp + tmp'
    diagview(Ltmp) ./= 2
    utLtmp = view(Ltmp, MatrixAlgebraKit.uppertriangularind(Ltmp))
    dL11  .= L11 * Ltmp
    dQ1   .= invL11 * dA1 - invL11 * dL11 * Q1

    dL21  .= (dA2 - L21 * dQ1) * adjoint(Q1)
    dQ2   .= -(dQ2 * Q1') * Q1
    if size(Q2, 1) > 0
        dQ2  .+= Q2 * (Q2' * dQ2)
    end
    if n3 > 0 && size(dQ2, 1) > 0
        # only present for qr_full or rank-deficient qr_compact
        Q3    = view(Q, (n1+n2+1):n, 1:n)
        dQ2 .+= Q3 * (Q3' * dQ2)
    end
    if !isempty(dL22)
        _, l22 = qr_full(dA2 - L21 * dQ1 - dL12 * Q1, MatrixAlgebraKit.LAPACK_HouseholderQR(; positive=true))
        dL22  .= view(l22, 1:size(dL22, 1), 1:size(dL22, 2))
    end
    return (dL, dQ)
end=#

function lq_pushforward!(dA, A, LQ, dLQ; kwargs...)
    qr_pushforward!(dA, A, (adjoint(LQ[2]), adjoint(LQ[1])), (adjoint(dLQ[2]), adjoint(dLQ[1])); kwargs...)
end

function lq_null_pushforward!(dA, A, LQ, dLQ; tol::Real=MatrixAlgebraKit.default_pullback_gaugetol(LQ[1]), rank_atol::Real=tol, gauge_atol::Real=tol) end
