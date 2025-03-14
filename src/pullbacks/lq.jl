# TODO: we should somewhere check that we only call this when performing a positive LQ

"""
    lq_compact_pullback!(ΔA, (L, Q), (ΔL, ΔQ);
                            tol::Real=default_pullback_gaugetol(R),
                            rank_atol::Real=tol,
                            gauge_atol::Real=tol)

Adds the pullback from the LQ decomposition of `A` to `ΔA` given the output `(L, Q)` and
cotangent `(ΔL, ΔQ)` of `lq_compact(A; positive=true)` or `lq_full(A; positive=true)`.

In the case where the rank `r` of the original matrix `A ≈ L * Q` (as determined
by `rank_atol`) is less then the  minimum of the number of rows and columns ,
the cotangents `ΔL` and `ΔQ`, only the first `r` columns of `L` and the first `r` rows
of `Q` are well-defined, and also the adjoint variables `ΔL` and `ΔQ` should have nonzero
values only in the first `r` columns and rows respectively. If nonzero values in the
remaining columns or rows exceed `gauge_atol`, a warning will be printed.
"""
function lq_compact_pullback!(ΔA::AbstractMatrix, LQ, ΔLQ;
                              tol::Real=default_pullback_gaugetol(LQ[1]),
                              rank_atol::Real=tol,
                              gauge_atol::Real=tol)
    # process
    L, Q = LQ
    m = size(L, 1)
    n = size(Q, 2)
    minmn = min(m, n)
    Ld = diagview(L)
    p = findlast(>=(rank_atol) ∘ abs, Ld)

    ΔL, ΔQ = ΔLQ

    Q1 = view(Q, 1:p, :)
    Q2 = view(Q, (p + 1):size(Q, 1), :)
    L11 = view(L, 1:p, 1:p)
    ΔA1 = view(ΔA, 1:p, :)
    ΔA2 = view(ΔA, (p + 1):m, :)

    if minmn > p # case where A is rank-deficient
        Δgauge = abs(zero(eltype(Q)))
        if !iszerotangent(ΔQ)
            # in this case the number Householder reflections will
            # change upon small variations, and all of the remaining
            # columns of ΔQ should be zero for a gauge-invariant
            # cost function
            ΔQ2 = view(ΔQ, (p + 1):size(Q, 1), :)
            Δgauge = max(Δgauge, norm(ΔQ2, Inf))
        end
        if !iszerotangent(ΔL)
            ΔL22 = view(ΔL, (p + 1):m, (p + 1):minmn)
            Δgauge = max(Δgauge, norm(ΔL22, Inf))
        end
        Δgauge < gauge_atol ||
            @warn "`lq` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
    end

    ΔQ̃ = zero!(similar(Q, (p, n)))
    if !iszerotangent(ΔQ)
        ΔQ1 = view(ΔQ, 1:p, :)
        copy!(ΔQ̃, ΔQ1)
        if p < size(Q, 1)
            Q2 = view(Q, (p + 1):size(Q, 1), :)
            ΔQ2 = view(ΔQ, (p + 1):size(Q, 1), :)
            # in the case where A is full rank, but there are more columns in Q than in A
            # (the case of `qr_full`), there is gauge-invariant information in the
            # projection of ΔQ2 onto the column space of Q1, by virtue of Q being a unitary
            # matrix. As the number of Householder reflections is in fixed in the full rank
            # case, Q is expected to rotate smoothly (we might even be able to predict) also
            # how the full Q2 will change, but this we omit for now, and we consider
            # Q2' * ΔQ2 as a gauge dependent quantity.
            ΔQ2Q1d = ΔQ2 * Q1'
            Δgauge = norm(mul!(copy(ΔQ2), ΔQ2Q1d, Q1, -1, 1), Inf)
            Δgauge < tol ||
                @warn "`qr` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
            ΔQ̃ = mul!(ΔQ̃, ΔQ2Q1d', Q2, -1, 1)
        end
    end
    if !iszerotangent(ΔL) && m > p
        L21 = view(L, (p + 1):m, 1:p)
        ΔL21 = view(ΔL, (p + 1):m, 1:p)
        ΔQ̃ = mul!(ΔQ̃, L21' * ΔL21, Q1, -1, 1)
        # Adding ΔA2 contribution
        ΔA2 = mul!(ΔA2, ΔL21, Q1, 1, 1)
    end

    # construct M
    M = zero!(similar(L, (p, p)))
    if !iszerotangent(ΔL)
        ΔL11 = view(ΔL, 1:p, 1:p)
        M = mul!(M, L11', ΔL11, 1, 1)
    end
    M = mul!(M, ΔQ̃, Q1', -1, 1)
    view(M, uppertriangularind(M)) .= conj.(view(M, lowertriangularind(M)))
    if eltype(M) <: Complex
        Md = diagview(M)
        Md .= real.(Md)
    end
    ldiv!(LowerTriangular(L11)', M)
    ldiv!(LowerTriangular(L11)', ΔQ̃)
    ΔA1 = mul!(ΔA1, M, Q1, +1, 1)
    ΔA1 .+= ΔQ̃
    return ΔA
end
