function check_qr_cotangents(Q, R, ΔQ, ΔR, minmn::Int, p::Int; gauge_atol::Real = default_pullback_gauge_atol(ΔQ))
    if minmn > p # case where A is rank-deficient
        Δgauge = abs(zero(eltype(Q)))
        if !iszerotangent(ΔQ)
            # in this case the number Householder reflections will
            # change upon small variations, and all of the remaining
            # columns of ΔQ should be zero for a gauge-invariant
            # cost function
            ΔQ2 = view(ΔQ, :, (p + 1):size(Q, 2))
            Δgauge = max(Δgauge, norm(ΔQ2, Inf))
        end
        if !iszerotangent(ΔR)
            ΔR22 = view(ΔR, (p + 1):minmn, (p + 1):size(R, 2))
            Δgauge = max(Δgauge, norm(ΔR22, Inf))
        end
        Δgauge ≤ gauge_atol ||
            @warn "`qr` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
    end
    return
end

function check_qr_full_cotangents(Q1, ΔQ2, Q1dΔQ2; gauge_atol::Real = default_pullback_gauge_atol(ΔQ2))
    # in the case where A is full rank, but there are more columns in Q than in A
    # (the case of `qr_full`), there is gauge-invariant information in the
    # projection of ΔQ2 onto the column space of Q1, by virtue of Q being a unitary
    # matrix. As the number of Householder reflections is in fixed in the full rank
    # case, Q is expected to rotate smoothly (we might even be able to predict) also
    # how the full Q2 will change, but this we omit for now, and we consider
    # Q2' * ΔQ2 as a gauge dependent quantity.
    Δgauge = norm(mul!(copy(ΔQ2), Q1, Q1dΔQ2, -1, 1), Inf)
    Δgauge ≤ gauge_atol ||
        @warn "`qr` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
    return
end

"""
    qr_pullback!(
        ΔA, A, QR, ΔQR;
        tol::Real = default_pullback_gaugetol(QR[2]),
        rank_atol::Real = default_pullback_rank_atol(QR[2]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔQR[1])
    )

Adds the pullback from the QR decomposition of `A` to `ΔA` given the output `QR` and
cotangent `ΔQR` of `qr_compact(A; positive = true)` or `qr_full(A; positive = true)`.

In the case where the rank `r` of the original matrix `A ≈ Q * R` (as determined by
`rank_atol`) is less then the minimum of the number of rows and columns, the cotangents `ΔQ`
and `ΔR`, only the first `r` columns of `Q` and the first `r` rows of `R` are well-defined,
and also the adjoint variables `ΔQ` and `ΔR` should have nonzero values only in the first
`r` columns and rows respectively. If nonzero values in the remaining columns or rows exceed
`gauge_atol`, a warning will be printed.
"""
function qr_pullback!(
        ΔA::AbstractMatrix, A, QR, ΔQR;
        rank_atol::Real = default_pullback_rank_atol(QR[2]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔQR[1])
    )
    # process
    Q, R = QR
    m = size(Q, 1)
    n = size(R, 2)
    minmn = min(m, n)
    Rd = diagview(R)
    p = @something findlast(>=(rank_atol) ∘ abs, Rd) 0

    ΔQ, ΔR = ΔQR

    Q1 = view(Q, :, 1:p)
    Q2 = view(Q, :, (p + 1):size(Q, 2))
    R11 = view(R, 1:p, 1:p)
    ΔA1 = view(ΔA, :, 1:p)
    ΔA2 = view(ΔA, :, (p + 1):n)

    check_qr_cotangents(Q, R, ΔQ, ΔR, minmn, p; gauge_atol)

    ΔQ̃ = zero!(similar(Q, (m, p)))
    if !iszerotangent(ΔQ)
        copy!(ΔQ̃, view(ΔQ, :, 1:p))
        if p < size(Q, 2)
            Q2 = view(Q, :, (p + 1):size(Q, 2))
            ΔQ2 = view(ΔQ, :, (p + 1):size(Q, 2))
            Q1dΔQ2 = Q1' * ΔQ2
            check_qr_full_cotangents(Q1, ΔQ2, Q1dΔQ2; gauge_atol)
            ΔQ̃ = mul!(ΔQ̃, Q2, Q1dΔQ2', -1, 1)
        end
    end
    if !iszerotangent(ΔR) && n > p
        R12 = view(R, 1:p, (p + 1):n)
        ΔR12 = view(ΔR, 1:p, (p + 1):n)
        ΔQ̃ = mul!(ΔQ̃, Q1, ΔR12 * R12', -1, 1)
        # Adding ΔA2 contribution
        ΔA2 = mul!(ΔA2, Q1, ΔR12, 1, 1)
    end

    # construct M
    M = zero!(similar(R, (p, p)))
    if !iszerotangent(ΔR)
        ΔR11 = view(ΔR, 1:p, 1:p)
        M = mul!(M, ΔR11, R11', 1, 1)
    end
    M = mul!(M, Q1', ΔQ̃, -1, 1)
    view(M, lowertriangularind(M)) .= conj.(view(M, uppertriangularind(M)))
    if eltype(M) <: Complex
        Md = diagview(M)
        Md .= real.(Md)
    end
    rdiv!(M, UpperTriangular(R11)')
    rdiv!(ΔQ̃, UpperTriangular(R11)')
    ΔA1 = mul!(ΔA1, Q1, M, +1, 1)
    ΔA1 .+= ΔQ̃
    return ΔA
end

function check_qr_null_cotangents(N, ΔN; gauge_atol::Real = default_pullback_gauge_atol(ΔN))
    aNᴴΔN = project_antihermitian!(N' * ΔN)
    Δgauge = norm(aNᴴΔN)
    Δgauge ≤ gauge_atol ||
        @warn "`qr_null` cotangent sensitive to gauge choice: (|Δgauge| = $Δgauge)"
    return
end

"""
    qr_null_pullback!(
        ΔA::AbstractMatrix, A, N, ΔN;
        gauge_atol::Real = default_pullback_gauge_atol(ΔN)
    )

Adds the pullback from the right nullspace of `A` to `ΔA`, given the nullspace basis
`N` and its cotangent `ΔN` of `qr_null(A)`.

See also [`qr_pullback!`](@ref).
"""
function qr_null_pullback!(
        ΔA::AbstractMatrix, A, N, ΔN;
        gauge_atol::Real = default_pullback_gauge_atol(ΔN)
    )
    if !iszerotangent(ΔN) && size(N, 2) > 0
        check_qr_null_cotangents(N, ΔN; gauge_atol)
        Q, R = qr_compact(A; positive = true)
        X = rdiv!(ΔN' * Q, UpperTriangular(R)')
        ΔA = mul!(ΔA, N, X, -1, 1)
    end
    return ΔA
end
