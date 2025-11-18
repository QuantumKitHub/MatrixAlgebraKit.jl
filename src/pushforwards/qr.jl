function qr_pushforward!(dA, A, QR, dQR; tol::Real = default_pullback_gauge_atol(QR[2]), rank_atol::Real = tol, gauge_atol::Real = tol)
    Q, R = QR
    m = size(A, 1)
    n = size(A, 2)
    minmn = min(m, n)
    Rd = diagview(R)
    p = findlast(>=(rank_atol) ∘ abs, Rd)

    m1 = p
    m2 = minmn - p
    m3 = m - minmn
    n1 = p
    n2 = n - p

    Q1 = view(Q, 1:m, 1:m1) # full rank portion
    Q2 = view(Q, 1:m, (m1 + 1):(m2 + m1))
    R11 = view(R, 1:m1, 1:n1)
    R12 = view(R, 1:m1, (n1 + 1):n)

    dA1 = view(dA, 1:m, 1:n1)
    dA2 = view(dA, 1:m, (n1 + 1):n)

    dQ, dR = dQR
    dQ1 = view(dQ, 1:m, 1:m1)
    dQ2 = view(dQ, 1:m, (m1 + 1):(m2 + m1))
    dQ3 = minmn + 1 < size(dQ, 2) ? view(dQ, :, (minmn + 1):size(dQ, 2)) : similar(dQ, eltype(dQ), (0, 0))
    dR11 = view(dR, 1:m1, 1:n1)
    dR12 = view(dR, 1:m1, (n1 + 1):n)
    dR22 = view(dR, (m1 + 1):(m1 + m2), (n1 + 1):n)

    # fwd rule for Q1 and R11 -- for a non-rank redeficient QR, this is all we need
    invR11 = inv(R11)
    tmp = Q1' * dA1 * invR11
    Rtmp = tmp + tmp'
    diagview(Rtmp) ./= 2
    ltRtmp = view(Rtmp, lowertriangularind(Rtmp))
    ltRtmp .= zero(eltype(Rtmp))
    dR11 .= Rtmp * R11
    dQ1 .= dA1 * invR11 - Q1 * dR11 * invR11
    dR12 .= adjoint(Q1) * (dA2 - dQ1 * R12)
    if size(Q2, 2) > 0
        dQ2 .= -Q1 * (Q1' * Q2)
        dQ2 .+= Q2 * (Q2' * dQ2)
    end
    if m3 > 0 && size(Q, 2) > minmn
        # only present for qr_full or rank-deficient qr_compact
        Q′ = view(Q, :, 1:minmn)
        Q3 = view(Q, :, (minmn + 1):m)
        #dQ3  .= Q′ * (Q′' * Q3)
        dQ3 .= Q3
    end
    if !isempty(dR22)
        _, r22 = qr_compact(dA2 - dQ1 * R12 - Q1 * dR12; positive = true)
        dR22 .= view(r22, 1:size(dR22, 1), 1:size(dR22, 2))
    end
    return (dQ, dR)
end

function qr_null_pushforward!(dA, A, N, dN; tol::Real = default_pullback_gauge_atol(N), rank_atol::Real = tol, gauge_atol::Real = tol)
    return iszero(min(size(N)...)) && return # nothing to do
end
