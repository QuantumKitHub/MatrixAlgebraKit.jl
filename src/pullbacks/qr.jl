qr_rank(R; rank_atol = default_pullback_rank_atol(R)) =
    @something findlast(>=(rank_atol) ∘ abs, diagview(R)) 0

function check_and_prepare_qr_cotangents(
        Q, R, ΔQ, ΔR, p::Int;
        gauge_atol::Real = default_pullback_gauge_atol(ΔQ)
    )
    m, n = size(Q, 1), size(R, 2)
    minmn = min(m, n)
    Δgauge = abs(zero(eltype(Q)))
    Q₁ = view(Q, :, 1:p)
    ΔQ₁ = zero!(similar(Q₁))
    if !iszerotangent(ΔQ)
        size(ΔQ) == size(Q) || throw(DimensionMismatch("ΔQ must have the same size as Q"))
        ΔQ₁ .= view(ΔQ, 1:m, 1:p)
        if p == minmn # full rank case, ΔQ₃ contains gauge-invariant information along Q₁
            ΔQ₃ = copy(view(ΔQ, :, (minmn + 1):size(Q, 2))) # extra columns in the case of qr_full
            Q₃ = view(Q, :, (minmn + 1):size(Q, 2))
            Q₁ᴴΔQ₃ = Q₁' * ΔQ₃
            mul!(ΔQ₃, Q₁, Q₁ᴴΔQ₃, -1, 1)
            Δgauge_Q = norm(ΔQ₃, Inf)
            mul!(ΔQ₁, Q₃, Q₁ᴴΔQ₃', -1, 1)
        else
            ΔQ₂₃ = view(ΔQ, :, (p + 1):size(Q, 2))
            Δgauge_Q = norm(ΔQ₂₃, Inf)
        end
        Δgauge = max(Δgauge, Δgauge_Q)
    end
    if !iszerotangent(ΔR)
        size(ΔR) == size(R) || throw(DimensionMismatch("ΔR must have the same size as R"))
        ΔR₁₁ = UpperTriangular(view(ΔR, 1:p, 1:p))
        ΔR₁₂ = view(ΔR, 1:p, (p + 1):n)
        ΔR₂₂ = view(ΔR, (p + 1):minmn, (p + 1):n)
        if p < minmn # otherwise ΔR₂₂ is empty
            # uppertriangularind generates linear indices
            # compute the appropriate offset in ΔR so we aren't
            # operating on a view-of-view, which doesn't work
            # for GPU arrays
            I = uppertriangularind(ΔR₂₂)
            upper_inds = view(LinearIndices(ΔR), (p + 1):minmn, (p + 1):n)[I]
            ΔR₂₂upper = view(ΔR, upper_inds)
            Δgauge_R = norm(ΔR₂₂upper, Inf)
            Δgauge_R = max(Δgauge_R, norm(view(ΔR₂₂, diagind(ΔR₂₂)), Inf))
            Δgauge = max(Δgauge, Δgauge_R)
        end
    else
        ΔR₁₁ = nothing
        ΔR₁₂ = nothing
    end
    Δgauge ≤ gauge_atol ||
        @warn "`qr` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
    return ΔQ₁, ΔR₁₁, ΔR₁₂
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
    p = qr_rank(R; rank_atol)
    (m, n) == size(ΔA) || throw(DimensionMismatch("size of ΔA ($(size(ΔA))) does not match size of Q*R ($m, $n)"))


    Q₁ = view(Q, :, 1:p)
    R₁₁ = UpperTriangular(R[1:p, 1:p])
    R₁₂ = view(R, 1:p, (p + 1):n)

    ΔA₁ = view(ΔA, :, 1:p)
    ΔA₂ = view(ΔA, :, (p + 1):n)

    ΔQ, ΔR = ΔQR
    ΔQ₁, ΔR₁₁, ΔR₁₂ = check_and_prepare_qr_cotangents(Q, R, ΔQ, ΔR, p; gauge_atol)

    if !iszerotangent(ΔR) && n > p
        ΔQ₁ = mul!(ΔQ₁, Q₁, ΔR₁₂ * R₁₂', -1, 1)
        # Adding ΔA₂ contribution
        ΔA₂ = mul!(ΔA₂, Q₁, ΔR₁₂, 1, 1)
    end

    # construct M
    M = zero!(similar(R, (p, p)))
    if !iszerotangent(ΔR)
        M = mul!(M, ΔR₁₁, R₁₁', 1, 1)
    end
    M = mul!(M, Q₁', ΔQ₁, -1, 1)
    view(M, lowertriangularind(M)) .= conj.(view(M, uppertriangularind(M)))
    if eltype(M) <: Complex
        Md = diagview(M)
        Md .= real.(Md)
    end
    mul!(ΔQ₁, Q₁, M, +1, 1)
    ΔA₁ .+= rdiv!(ΔQ₁, R₁₁')
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

"""
    remove_qr_gauge_dependence!(ΔQ, ΔR, A, Q, R; rank_atol = ...)

Remove the gauge-dependent part from the cotangents `ΔQ` and `ΔR` of the QR factors `Q` and
`R`. For the full QR decomposition, the extra columns of `Q` beyond the rank `r` are not
uniquely determined by `A`, so the corresponding part of `ΔQ` is projected to remove this
ambiguity. Additionally, rows of `ΔR` beyond the rank are zeroed out.
"""
function remove_qr_gauge_dependence!(ΔQ, ΔR, A, Q, R; rank_atol = MatrixAlgebraKit.default_pullback_rank_atol(R))
    r = MatrixAlgebraKit.qr_rank(R; rank_atol)
    m, n = size(A, 1), size(A, 2)
    minmn = min(m, n)
    Q₁ = view(Q, :, 1:r)
    ΔQ₂ = view(ΔQ, :, (r + 1):minmn)
    zero!(ΔQ₂)
    ΔQ₃ = view(ΔQ, :, (minmn + 1):size(ΔQ, 2)) # extra columns in the case of qr_full
    if r == minmn # full rank case, ΔQ₃ contains gauge-invariant information along Q₁
        Q₁ᴴΔQ₃ = Q₁' * ΔQ₃
        mul!(ΔQ₃, Q₁, Q₁ᴴΔQ₃)
    else # rank-deficient case, no gauge-invariant information
        zero!(ΔQ₃)
    end
    ΔR₂₂ = view(ΔR, (r + 1):minmn, (r + 1):size(R, 2))
    zero!(diagview(ΔR₂₂))
    if r < minmn
        # uppertriangularind generates linear indices
        # compute the appropriate offset in ΔR so we aren't
        # operating on a view-of-view, which doesn't work
        # for GPU arrays
        # uppertriangularind generates linear indices
        # compute the appropriate offset in ΔR so we aren't
        # operating on a view-of-view, which doesn't work
        # for GPU arrays
        I = uppertriangularind(ΔR₂₂)
        upper_inds = view(LinearIndices(ΔR), (r + 1):minmn, (r + 1):n)[I]
        ΔR₂₂upper = view(ΔR, upper_inds)
        zero!(ΔR₂₂upper)
    end
    return ΔQ, ΔR
end

"""
    remove_qr_null_gauge_dependence!(ΔN, A, N)

Remove the gauge-dependent part from the cotangent `ΔN` of the QR null space `N`. The null
space is only determined up to a unitary rotation, so `ΔN` is projected onto the column span
of the compact QR factor `Q₁`.
"""
function remove_qr_null_gauge_dependence!(ΔN, A, N)
    return mul!(ΔN, N, N' * ΔN, -1, 1)
end

"""
    remove_left_null_gauge_dependence!(ΔN, A, N)

Remove the gauge-dependent part from the cotangent `ΔN` of the left null space `N`. The null
space basis is only determined up to a unitary rotation, so `ΔN` is projected onto the column
span of the compact QR factor `Q₁` of `A`.
"""
remove_left_null_gauge_dependence!(ΔN, A, N) = remove_qr_null_gauge_dependence!(ΔN, A, N)
