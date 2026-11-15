"""
    left_polar_pullback!(
        ΔA, A, WP, ΔWP;
        rank_atol::Real = default_pullback_rank_atol(WP[2]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔWP...)
    )

Adds the pullback from the left polar decomposition of `A` to `ΔA` given the output `WP` and
cotangent `ΔWP` of `left_polar(A)`.

Eigenvalues of `P` smaller than `rank_atol` are treated as zero, in which case `A` is rank
deficient and `W` is not uniquely determined on the null space of `P`. Components of `ΔW`
along this null space are discarded.
"""
function left_polar_pullback!(
        ΔA::AbstractMatrix, A, WP, ΔWP;
        rank_atol::Real = default_pullback_rank_atol(WP[2]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔWP...)
    )
    # Extract the Polar components
    W, P = WP

    # Extract and check the cotangents
    ΔW, ΔP = ΔWP
    if !iszerotangent(ΔP)
        ΔP = project_hermitian(ΔP)
    end
    D, U = eigh_full(P)
    d = diagview(D)
    M = zero(P)
    !iszerotangent(ΔW) && mul!(M, W', ΔW, 1, 1)
    !iszerotangent(ΔP) && mul!(M, ΔP, P, -1, 1)
    C, Δgauge = _sylvester_psd(U, d, M - M', rank_atol)
    !iszerotangent(ΔP) && (C .+= ΔP)
    ΔA = mul!(ΔA, W, C, 1, 1)
    if !iszerotangent(ΔW)
        # (1 - W * W') * ΔW * pinv(P)
        ΔWU = ΔW * U
        WᴴΔWU = W' * ΔWU
        ΔWU = mul!(ΔWU, W, WᴴΔWU, -1, 1)
        Δgauge = max(Δgauge, maximum(abs, (abs.(transpose(d)) .< rank_atol) .* ΔWU; init = zero(real(eltype(ΔWU)))))
        ΔWU .*= transpose(inv_safe.(d, rank_atol))
        ΔA = mul!(ΔA, ΔWU, U', 1, 1)
    end
    Δgauge ≤ gauge_atol ||
        @warn "`left_polar` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
    return ΔA
end

"""
    right_polar_pullback!(
        ΔA, A, PWᴴ, ΔPWᴴ;
        rank_atol::Real = default_pullback_rank_atol(PWᴴ[1]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔPWᴴ...)
    )

Adds the pullback from the right polar decomposition of `A` to `ΔA` given the output `PWᴴ`
and cotangent `ΔPWᴴ` of `right_polar(A)`.

Eigenvalues of `P` smaller than `rank_atol` are treated as zero, in which case `A` is rank
deficient and `Wᴴ` is not uniquely determined on the null space of `P`. Components of `ΔWᴴ`
along this null space are discarded.
"""
function right_polar_pullback!(
        ΔA::AbstractMatrix, A, PWᴴ, ΔPWᴴ;
        rank_atol::Real = default_pullback_rank_atol(PWᴴ[1]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔPWᴴ...)
    )
    # Extract the Polar components
    P, Wᴴ = PWᴴ

    # Extract and check the cotangents
    ΔP, ΔWᴴ = ΔPWᴴ
    if !iszerotangent(ΔP)
        ΔP = project_hermitian(ΔP)
    end
    D, U = eigh_full(P)
    d = diagview(D)
    M = zero(P)
    !iszerotangent(ΔWᴴ) && mul!(M, ΔWᴴ, Wᴴ', 1, 1)
    !iszerotangent(ΔP) && mul!(M, P, ΔP, -1, 1)
    C, Δgauge = _sylvester_psd(U, d, M - M', rank_atol)
    !iszerotangent(ΔP) && (C .+= ΔP)
    ΔA = mul!(ΔA, C, Wᴴ, 1, 1)
    if !iszerotangent(ΔWᴴ)
        # pinv(P) * ΔWᴴ * (1 - Wᴴ' * Wᴴ)
        UᴴΔWᴴ = U' * ΔWᴴ
        UᴴΔWᴴW = UᴴΔWᴴ * Wᴴ'
        UᴴΔWᴴ = mul!(UᴴΔWᴴ, UᴴΔWᴴW, Wᴴ, -1, 1)
        Δgauge = max(Δgauge, maximum(abs, (abs.(d) .< rank_atol) .* UᴴΔWᴴ; init = zero(real(eltype(UᴴΔWᴴ)))))
        UᴴΔWᴴ .*= inv_safe.(d, rank_atol)
        ΔA = mul!(ΔA, U, UᴴΔWᴴ, 1, 1)
    end
    Δgauge ≤ gauge_atol ||
        @warn "`right_polar` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
    return ΔA
end
