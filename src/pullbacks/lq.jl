lq_rank(L; kwargs...) = qr_rank(L; kwargs...)

function check_lq_cotangents(
        L, Q, ΔL, ΔQ, p::Int;
        gauge_atol::Real = default_pullback_gauge_atol(ΔQ)
    )
    minmn = min(size(L, 1), size(Q, 2))
    Δgauge = abs(zero(eltype(Q)))
    if !iszerotangent(ΔQ)
        ΔQ₂ = view(ΔQ, (p + 1):minmn, :)
        ΔQ₃ = ΔQ[(minmn + 1):size(Q, 1), :]
        Δgauge_Q = norm(ΔQ₂, Inf)
        Q₁ = view(Q, 1:p, :)
        ΔQ₃Q₁ᴴ = ΔQ₃ * Q₁'
        mul!(ΔQ₃, ΔQ₃Q₁ᴴ, Q₁, -1, 1)
        Δgauge_Q = max(Δgauge_Q, norm(ΔQ₃, Inf))
        Δgauge = max(Δgauge, Δgauge_Q)
    end
    if !iszerotangent(ΔL)
        ΔL22 = view(ΔL, (p + 1):size(ΔL, 1), (p + 1):minmn)
        Δgauge_L = norm(view(ΔL22, lowertriangularind(ΔL22)), Inf)
        Δgauge_L = max(Δgauge_L, norm(view(ΔL22, diagind(ΔL22)), Inf))
        Δgauge = max(Δgauge, Δgauge_L)
    end
    Δgauge ≤ gauge_atol ||
        @warn "`lq` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
    return nothing
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
    minmn = min(m, n)
    p = lq_rank(L; rank_atol)

    ΔL, ΔQ = ΔLQ

    Q₁ = view(Q, 1:p, :)
    L₁₁ = LowerTriangular(view(L, 1:p, 1:p))
    ΔA₁ = view(ΔA, 1:p, :)
    ΔA₂ = view(ΔA, (p + 1):m, :)

    check_lq_cotangents(L, Q, ΔL, ΔQ, p; gauge_atol)

    ΔQ̃ = zero!(similar(Q, (p, n)))
    if !iszerotangent(ΔQ)
        ΔQ₁ = view(ΔQ, 1:p, :)
        copy!(ΔQ̃, ΔQ₁)
        if minmn < size(Q, 1)
            ΔQ₃ = view(ΔQ, (minmn + 1):size(ΔQ, 1), :)
            Q₃ = view(Q, (minmn + 1):size(Q, 1), :)
            ΔQ₃Q₁ᴴ = ΔQ₃ * Q₁'
            ΔQ̃ = mul!(ΔQ̃, ΔQ₃Q₁ᴴ', Q₃, -1, 1)
        end
    end
    if !iszerotangent(ΔL) && m > p
        L₂₁ = view(L, (p + 1):m, 1:p)
        ΔL₂₁ = view(ΔL, (p + 1):m, 1:p)
        ΔQ̃ = mul!(ΔQ̃, L₂₁' * ΔL₂₁, Q₁, -1, 1)
        # Adding ΔA₂ contribution
        ΔA₂ = mul!(ΔA₂, ΔL₂₁, Q₁, 1, 1)
    end

    # construct M
    M = zero!(similar(L, (p, p)))
    if !iszerotangent(ΔL)
        ΔL₁₁ = LowerTriangular(view(ΔL, 1:p, 1:p))
        M = mul!(M, L₁₁', ΔL₁₁, 1, 1)
    end
    M = mul!(M, ΔQ̃, Q₁', -1, 1)
    view(M, uppertriangularind(M)) .= conj.(view(M, lowertriangularind(M)))
    if eltype(M) <: Complex
        Md = diagview(M)
        Md .= real.(Md)
    end
    ldiv!(L₁₁', M)
    ldiv!(L₁₁', ΔQ̃)
    ΔA₁ = mul!(ΔA₁, M, Q₁, +1, 1)
    ΔA₁ .+= ΔQ̃
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
