"""
    qr_pushforward!(
        ΔA, A, QR, ΔQR;
        rank_atol::Real = default_pullback_rank_atol(QR[2])
    )

Computes the pushforward `ΔQR` of the QR decomposition `QR` of `qr_compact(A;
positive = true)` or `qr_full(A; positive = true)` given the tangent `ΔA` of `A`.

If the original matrix `A` is rank-deficient (rank `r < min(size(A)...)`), only the first `r`
columns of `Q` and the first `r` rows of `R` are differentiable, and the tangents of the
remaining columns of `Q` and rows of `R` are set to zero. Similarly, for `qr_full` the extra
columns of `Q` are only determined up to a unitary rotation, and only their gauge-invariant
tangent component along the first `r` columns of `Q` is computed.
"""
function qr_pushforward!(
        ΔA, A, QR, ΔQR;
        rank_atol::Real = default_pullback_rank_atol(QR[2]), kwargs...
    )
    Q, R = QR
    ΔQ, ΔR = ΔQR
    m = size(Q, 1)
    n = size(R, 2)
    minmn = min(m, n)
    p = qr_rank(R; rank_atol)
    (m, n) == size(ΔA) || throw(DimensionMismatch("size of ΔA ($(size(ΔA))) does not match size of Q*R ($m, $n)"))

    Q₁ = view(Q, :, 1:p)
    R₁₁ = UpperTriangular(view(R, 1:p, 1:p))
    R₁₂ = view(R, 1:p, (p + 1):n)

    ΔA₁ = view(ΔA, :, 1:p)
    ΔA₂ = view(ΔA, :, (p + 1):n)

    ΔQ₁ = ΔA₁ / R₁₁
    Q₁ᴴΔQ₁ = Q₁' * ΔQ₁
    M = Q₁ᴴΔQ₁ + Q₁ᴴΔQ₁'
    diagview(M) ./= 2
    view(M, lowertriangularind(M)) .= zero(eltype(M))
    ΔR₁₁ = M * R₁₁
    ΔQ₁ = mul!(ΔQ₁, Q₁, M, -1, 1)
    ΔR₁₂ = Q₁' * ΔA₂
    ΔR₁₂ = mul!(ΔR₁₂, ΔQ₁' * Q₁, R₁₂, 1, 1)

    zero!(ΔQ)
    zero!(ΔR)
    view(ΔQ, :, 1:p) .= ΔQ₁
    view(ΔR, 1:p, 1:p) .= ΔR₁₁
    view(ΔR, 1:p, (p + 1):n) .= ΔR₁₂
    if p == minmn && size(Q, 2) > minmn
        # extra columns in the case of qr_full, orthogonality to Q₁ fixes their component along Q₁
        Q₃ = view(Q, :, (minmn + 1):size(Q, 2))
        ΔQ₃ = view(ΔQ, :, (minmn + 1):size(Q, 2))
        mul!(ΔQ₃, Q₁, ΔQ₁' * Q₃, -1, 0)
    end
    return ΔQ, ΔR
end

"""
    qr_null_pushforward!(ΔA, A, N, ΔN; kwargs...)

Compute the pushforward `ΔN` of the nullspace basis `N` of `qr_null(A)` given the
tangent `ΔA` of `A`.

See also [`qr_pushforward!`](@ref).
"""
function qr_null_pushforward!(ΔA, A, N, ΔN; kwargs...)
    if size(N, 2) == 0
        zero!(ΔN)
        return ΔN
    end
    Q, R = qr_compact(A; positive = true)
    X = ldiv!(UpperTriangular(R)', ΔA' * N)
    return mul!(ΔN, Q, X, -1, 0)
end
