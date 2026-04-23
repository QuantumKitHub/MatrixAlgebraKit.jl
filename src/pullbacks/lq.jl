lq_rank(L; kwargs...) = qr_rank(L; kwargs...)

function check_and_prepare_lq_cotangents(
        L, Q, ΔL, ΔQ, p::Int;
        gauge_atol::Real = default_pullback_gauge_atol(ΔQ)
    )
    m, n = size(L, 1), size(Q, 2)
    minmn = min(m, n)
    Δgauge = abs(zero(eltype(Q)))
    Q₁ = view(Q, 1:p, :)
    ΔQ₁ = zero!(similar(Q₁))
    if !iszerotangent(ΔQ)
        size(ΔQ) == size(Q) || throw(DimensionMismatch("ΔQ must have the same size as Q"))
        ΔQ₁ .= view(ΔQ, 1:p, 1:n)
        if p == minmn # full rank case, ΔQ₃ contains gauge-invariant information along Q₁
            Q₃ = view(Q, (minmn + 1):size(Q, 1), :)
            ΔQ₃ = view(ΔQ, (minmn + 1):size(Q, 1), :)
            ΔQ₃Q₁ᴴ = ΔQ₃ * Q₁'
            mul!(ΔQ₃, ΔQ₃Q₁ᴴ, Q₁, -1, 1)
            Δgauge_Q = norm(ΔQ₃, Inf)
            mul!(ΔQ₁, ΔQ₃Q₁ᴴ', Q₃, -1, 1)
        else
            ΔQ₂ = view(ΔQ, (p + 1):size(ΔQ, 1), :)
            Δgauge_Q = norm(ΔQ₂, Inf)
        end
        Δgauge = max(Δgauge, Δgauge_Q)
    end
    if !iszerotangent(ΔL)
        size(ΔL) == size(L) || throw(DimensionMismatch("ΔL must have the same size as L"))
        ΔL₁₁ = LowerTriangular(view(ΔL, 1:p, 1:p))
        ΔL₂₁ = view(ΔL, (p + 1):size(ΔL, 1), 1:p)
        ΔL₂₂ = view(ΔL, (p + 1):size(ΔL, 1), (p + 1):minmn)
        Δgauge_L = norm(view(ΔL₂₂, lowertriangularind(ΔL₂₂)), Inf)
        Δgauge_L = max(Δgauge_L, norm(view(ΔL₂₂, diagind(ΔL₂₂)), Inf))
        Δgauge = max(Δgauge, Δgauge_L)
    else
        ΔL₁₁ = nothing
        ΔL₂₁ = nothing
    end
    Δgauge ≤ gauge_atol ||
        @warn "`lq` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
    return ΔL₁₁, ΔL₂₁, ΔQ₁
end

"""
    lq_pullback!(
        ΔA, A, LQ, ΔLQ;
        rank_atol::Real = default_pullback_rank_atol(LQ[1]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔLQ[2])
    )

Adds the pullback from the LQ decomposition of `A` to `ΔA` given the output `LQ` and
cotangent `ΔLQ` of `lq_compact(A; positive = true)` or `lq_full(A; positive = true)`.

In the case where the rank `r` of the original matrix `A ≈ L * Q` (as determined by
`rank_atol`) is less then the minimum of the number of rows and columns of the cotangents
`ΔL` and `ΔQ`, only the first `r` columns of `L` and the first `r` rows of `Q` are
well-defined, and also the adjoint variables `ΔL` and `ΔQ` should have nonzero values only
in the first `r` columns and rows respectively. If nonzero values in the remaining columns
or rows exceed `gauge_atol`, a warning will be printed.
"""
function lq_pullback!(
        ΔA::AbstractMatrix, A, LQ, ΔLQ;
        rank_atol::Real = default_pullback_rank_atol(LQ[1]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔLQ[2])
    )
    # process
    L, Q = LQ
    m = size(L, 1)
    n = size(Q, 2)
    p = lq_rank(L; rank_atol)
    (m, n) == size(ΔA) || throw(DimensionMismatch("size of ΔA ($(size(ΔA))) does not match size of L*Q ($m, $n)"))

    L₁₁ = LowerTriangular(view(L, 1:p, 1:p))
    L₂₁ = view(L, (p + 1):m, 1:p)
    Q₁ = view(Q, 1:p, :)

    ΔA₁ = view(ΔA, 1:p, :)
    ΔA₂ = view(ΔA, (p + 1):m, :)

    ΔL, ΔQ = ΔLQ
    ΔL₁₁, ΔL₂₁, ΔQ₁ = check_and_prepare_lq_cotangents(L, Q, ΔL, ΔQ, p; gauge_atol)

    if !iszerotangent(ΔL) && m > p
        ΔQ₁ = mul!(ΔQ₁, L₂₁' * ΔL₂₁, Q₁, -1, 1)
        # Adding ΔA₂ contribution
        ΔA₂ = mul!(ΔA₂, ΔL₂₁, Q₁, 1, 1)
    end

    # construct M
    M = zero!(similar(L, (p, p)))
    if !iszerotangent(ΔL)
        M = mul!(M, L₁₁', ΔL₁₁, 1, 1)
    end
    M = mul!(M, ΔQ₁, Q₁', -1, 1)
    view(M, uppertriangularind(M)) .= conj.(view(M, lowertriangularind(M)))
    if eltype(M) <: Complex
        Md = diagview(M)
        Md .= real.(Md)
    end
    ΔA₁ .+= ldiv!(L₁₁', mul!(ΔQ₁, M, Q₁, +1, 1))
    return ΔA
end

function check_lq_null_cotangents(Nᴴ, ΔNᴴ; gauge_atol::Real = default_pullback_gauge_atol(ΔNᴴ))
    aNᴴΔN = project_antihermitian!(Nᴴ * ΔNᴴ')
    Δgauge = norm(aNᴴΔN)
    Δgauge ≤ gauge_atol ||
        @warn "`lq_null` cotangent sensitive to gauge choice: (|Δgauge| = $Δgauge)"
    return
end

"""
    lq_null_pullback!(
        ΔA::AbstractMatrix, A, Nᴴ, ΔNᴴ;
        gauge_atol::Real = default_pullback_gauge_atol(ΔNᴴ)
    )

Adds the pullback from the left nullspace of `A` to `ΔA`, given the nullspace basis
 `Nᴴ` and its cotangent `ΔNᴴ` of `lq_null(A)`.

See also [`lq_pullback!`](@ref).
"""
function lq_null_pullback!(
        ΔA::AbstractMatrix, A, Nᴴ, ΔNᴴ;
        gauge_atol::Real = default_pullback_gauge_atol(ΔNᴴ)
    )
    if !iszerotangent(ΔNᴴ) && size(Nᴴ, 1) > 0
        check_lq_null_cotangents(Nᴴ, ΔNᴴ; gauge_atol)
        L, Q = lq_compact(A; positive = true) # should we be able to provide algorithm here?
        X = ldiv!(LowerTriangular(L)', Q * ΔNᴴ')
        ΔA = mul!(ΔA, X, Nᴴ, -1, 1)
    end
    return ΔA
end


"""
    remove_lq_gauge_dependence!(ΔL, ΔQ, A, L, Q; rank_atol = ...)

Remove the gauge-dependent part from the cotangents `ΔL` and `ΔQ` of the LQ factors `L` and
`Q`. For the full LQ decomposition, the extra rows of `Q` beyond the rank `r` are not uniquely
determined by `A`, so the corresponding part of `ΔQ` is projected to remove this ambiguity.
Additionally, columns of `ΔL` beyond the rank are zeroed out.
"""
function remove_lq_gauge_dependence!(ΔL, ΔQ, A, L, Q; rank_atol = MatrixAlgebraKit.default_pullback_rank_atol(L))
    r = MatrixAlgebraKit.lq_rank(L; rank_atol)
    minmn = min(size(A)...)
    Q₁ = view(Q, 1:r, :)
    ΔQ₂ = view(ΔQ, (r + 1):minmn, :)
    zero!(ΔQ₂)
    ΔQ₃ = view(ΔQ, (minmn + 1):size(ΔQ, 1), :) # extra rows in the case of lq_full
    if r == minmn
        ΔQ₃Q₁ᴴ = ΔQ₃ * Q₁'
        mul!(ΔQ₃, ΔQ₃Q₁ᴴ, Q₁)
    else # rank-deficient case, no gauge-invariant information
        zero!(ΔQ₃)
    end
    ΔL₂₂ = view(ΔL, (r + 1):size(ΔL, 1), (r + 1):minmn)
    zero!(diagview(ΔL₂₂))
    zero!(view(ΔL₂₂, lowertriangularind(ΔL₂₂)))
    return ΔL, ΔQ
end

"""
    remove_lq_null_gauge_dependence!(ΔNᴴ, A, Nᴴ)

Remove the gauge-dependent part from the cotangent `ΔNᴴ` of the LQ null space `Nᴴ`. The null
space is only determined up to a unitary rotation, so `ΔNᴴ` is projected onto the row span of
the compact LQ factor `Q₁`.
"""
function remove_lq_null_gauge_dependence!(ΔNᴴ, A, Nᴴ)
    return mul!(ΔNᴴ, ΔNᴴ * Nᴴ', Nᴴ, -1, 1)
end

"""
    remove_right_null_gauge_dependence!(ΔNᴴ, A, Nᴴ)

Remove the gauge-dependent part from the cotangent `ΔNᴴ` of the right null space `Nᴴ`. The
null space basis is only determined up to a unitary rotation, so `ΔNᴴ` is projected onto the
row span of the compact LQ factor `Q₁` of `A`.
"""
remove_right_null_gauge_dependence!(ΔNᴴ, A, Nᴴ) = remove_lq_null_gauge_dependence!(ΔNᴴ, A, Nᴴ)
