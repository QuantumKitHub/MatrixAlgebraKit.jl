function check_and_prepare_eig_cotangents(
        D, V, ViG, ΔDmat, ΔV, ind = Colon();
        degeneracy_atol::Real = default_pullback_rank_atol(S),
        gauge_atol::Real = default_pullback_gauge_atol(ΔDmat, ΔV)
    )

    n, p = size(V)
    indD = select_indices(axes(D, 1), ind)
    indV = select_indices(axes(V, 2), ind)
    if !iszerotangent(ΔV)
        n == size(ΔV, 1) || throw(DimensionMismatch())
        length(indV) == size(ΔV, 2) || throw(DimensionMismatch())
        if is_leading_index(indV, p)
            ΔV₁ = copy(ΔV)
        else
            ΔV₁ = zero(V)
            ΔV₁[:, indV] = ΔV
        end
        VᴴΔV₁ = V' * ΔV₁
        if p == n
            ΔV₊ = zero!(ΔV₁)
        else
            ΔV₊ = mul!(ΔV₁, ViG, VᴴΔV₁, -1, 1)
        end
    else
        ΔV₊ = nothing
        VᴴΔV₁ = zero!(similar(V, (p, p)))
    end

    bc = Base.broadcasted(transpose(D), D, VᴴΔV₁) do d₁, d₂, v
        return abs(d₁ - d₂) < degeneracy_atol ? v : zero(v)
    end
    Δgauge = maximum(abs, Base.Broadcast.instantiate(bc); init = abs(zero(eltype(D))))

    Δgauge ≤ gauge_atol ||
        @warn "`eig` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"

    VᴴΔV₁ .*= conj.(inv_safe.(transpose(D) .- D, degeneracy_atol))
    VᴴAΔV = VᴴΔV₁

    if !iszerotangent(ΔDmat)
        ΔD = diagview(ΔDmat)
        length(indD) == length(ΔD) || throw(DimensionMismatch())
        VᴴAΔV[select_indices(diagind(VᴴAΔV), indD)] .+= ΔD
    else
        ΔD = nothing
    end

    return VᴴAΔV, ΔV₊
end

"""
    eig_pullback!(
        ΔA::AbstractMatrix, A, DV, ΔDV, [ind];
        degeneracy_atol::Real = default_pullback_rank_atol(DV[1]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔDV[2])
    )

Adds the pullback from the full eigenvalue decomposition of `A` to `ΔA`, given the output
`DV` of `eig_full` and the cotangent `ΔDV` of `eig_full` or `eig_trunc`.

In particular, it is assumed that `A ≈ V * D * inv(V)` with thus
`size(A) == size(V) == size(D)` and `D` diagonal. For the cotangents, an arbitrary number of
eigenvectors or eigenvalues can be missing, i.e. for a matrix `A` of size `(n, n)`, `ΔV` can
have size `(n, pV)` and `diagview(ΔD)` can have length `pD`. In those cases, additionally
`ind` is required to specify which eigenvectors or eigenvalues are present in `ΔV` or `ΔD`.
By default, it is assumed that all eigenvectors and eigenvalues are present.

A warning will be printed if the cotangents are not gauge-invariant, i.e. if the restriction
of `V' * ΔV` to rows `i` and columns `j` for which `abs(D[i] - D[j]) < degeneracy_atol`, is
not small compared to `gauge_atol`.
"""
function eig_pullback!(
        ΔA::AbstractMatrix, A, DV, ΔDV, ind = Colon();
        degeneracy_atol::Real = default_pullback_rank_atol(DV[1]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔDV[2])
    )

    # Basic size checks and determination
    Dmat, V = DV
    n = LinearAlgebra.checksquare(V)
    D = diagview(Dmat)
    n == length(D) || throw(DimensionMismatch())
    (n, n) == size(ΔA) || throw(DimensionMismatch())
    iszero(n) && return ΔA
    ViG = inv(V)'

    ΔDmat, ΔV = ΔDV
    VᴴΔAV, = check_and_prepare_eig_cotangents(
        D, V, ViG, ΔDmat, ΔV, ind; degeneracy_atol, gauge_atol
    )

    if eltype(ΔA) <: Real
        Z = ViG * VᴴΔAV
        ΔAc = mul!(VᴴΔAV, Z, V') # recycle VᴴΔAV
        ΔA .+= real.(ΔAc)
    else
        Z = ViG * VᴴΔAV
        ΔA = mul!(ΔA, Z, V', 1, 1)
    end
    return ΔA
end
# Diagonal: do not specialize on `A`, since we may insert `A = nothing` to assert independence of `A` in the implementation
function eig_pullback!(
        ΔA::Diagonal, A, DV, ΔDV, ind = Colon();
        degeneracy_atol::Real = default_pullback_rank_atol(DV[1]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔDV[2])
    )
    # TODO:
    # If A and ΔA are diagonal, then V is a permutation matrix and so is inv(V) = V'.
    # Furthermore, since V̇ is 0, the pullback ΔV cannot contribute and we only have to unpermute ΔD.
    ΔA_full = zero!(similar(ΔA, size(ΔA)))
    ΔA_full = eig_pullback!(ΔA_full, A, DV, ΔDV, ind; degeneracy_atol, gauge_atol)
    diagview(ΔA) .+= diagview(ΔA_full)
    return ΔA
end

"""
    eig_trunc_pullback!(
        ΔA::AbstractMatrix, ΔDV, A, DV;
        degeneracy_atol::Real = default_pullback_rank_atol(DV[1]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔDV[2])
    )

Adds the pullback from the truncated eigenvalue decomposition of `A` to `ΔA`, given the
output `DV` and the cotangent `ΔDV` of `eig_trunc`.

In particular, it is assumed that `A * V ≈ V * D` with `V` a rectangular matrix of
eigenvectors and `D` diagonal. For the cotangents, it is assumed that if `ΔV` is not zero,
then it has the same number of columns as `V`, and if `ΔD` is not zero, then it is a
diagonal matrix of the same size as `D`.

For this method to work correctly, it is also assumed that the remaining eigenvalues
(not included in `D`) are (sufficiently) separated from those in `D`.

A warning will be printed if the cotangents are not gauge-invariant, i.e. if the restriction
of `V' * ΔV` to rows `i` and columns `j` for which `abs(D[i] - D[j]) < degeneracy_atol`, is
not small compared to `gauge_atol`.
"""
function eig_trunc_pullback!(
        ΔA::AbstractMatrix, A, DV, ΔDV;
        degeneracy_atol::Real = default_pullback_rank_atol(DV[1]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔDV[2]),
        maxiter::Int = 100 # TODO: better default, depending on expected number of steps using quadratic convergence?
    )

    # Basic size checks and determination
    Dmat, V = DV
    (n, p) = size(V)
    (n, n) == size(ΔA) || throw(DimensionMismatch())
    D = diagview(Dmat)
    p == length(D) || throw(DimensionMismatch())
    iszero(p) && return ΔA
    G = V' * V
    ViG = V / LinearAlgebra.cholesky!(G)

    ΔDmat, ΔV = ΔDV
    VᴴΔAV, ΔV₊ = check_and_prepare_eig_cotangents(
        D, V, ViG, ΔDmat, ΔV; degeneracy_atol, gauge_atol
    )
    Z = ViG * VᴴΔAV

    # add contribution from orthogonal complement
    AP = mul!(complex.(A), V * Dmat, ViG', -1, 1)
    X₀ = iszerotangent(ΔV₊) ? AP' * Z : mul!(ΔV₊, AP', Z, 1, 1)
    X₀ ./= D'
    dabsmax = maximum(abs, D)
    AP ./= dabsmax
    D̄⁻¹ = dabsmax ./ conj.(D)
    X₁ = rmul!(AP' * X₀, Diagonal(D̄⁻¹))
    X₁ .+= X₀
    Xₖ, Xₖ₊₁ = X₁, X₀
    APₖ, APₖ₊₁ = AP * AP, AP
    D̄⁻¹ₖ, D̄⁻¹ₖ₊₁ = D̄⁻¹ .^ 2, D̄⁻¹
    for k in 1:maxiter
        Xₖ₊₁ = rmul!(mul!(Xₖ₊₁, APₖ', Xₖ), Diagonal(D̄⁻¹ₖ))
        if norm(Xₖ₊₁, Inf) < degeneracy_atol
            break
        end
        Xₖ₊₁ .+= Xₖ
        if k == maxiter
            @warn "Sylvester iteration did not converge after $k iterations, final norm of X: $(norm(Xₖ₊₁, Inf)))"
            break
        end
        D̄⁻¹ₖ₊₁ .= D̄⁻¹ₖ .^ 2
        APₖ₊₁ = mul!(APₖ₊₁, APₖ, APₖ)
        Xₖ, Xₖ₊₁ = Xₖ₊₁, Xₖ
        APₖ, APₖ₊₁ = APₖ₊₁, APₖ
        D̄⁻¹ₖ, D̄⁻¹ₖ₊₁ = D̄⁻¹ₖ₊₁, D̄⁻¹ₖ
    end
    Z .+= Xₖ
    if eltype(ΔA) <: Real
        ΔAc = mul!(AP, Z, V') # recycle AP
        ΔA .+= real.(ΔAc)
    else
        ΔA = mul!(ΔA, Z, V', 1, 1)
    end
    return ΔA
end
function eig_trunc_pullback!(
        ΔA::Diagonal, A, DV, ΔDV;
        degeneracy_atol::Real = default_pullback_rank_atol(DV[1]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔDV[2])
    )
    ΔA_full = zero!(similar(ΔA, size(ΔA)))
    ΔA_full = eig_trunc_pullback!(ΔA_full, A, DV, ΔDV; degeneracy_atol, gauge_atol)
    diagview(ΔA) .+= diagview(ΔA_full)
    return ΔA
end

"""
    eig_vals_pullback!(
        ΔA, A, DV, ΔD, [ind];
        degeneracy_atol::Real = default_pullback_rank_atol(DV[1]),
    )

Adds the pullback from the eigenvalues of `A` to `ΔA`, given the output
`DV` of `eig_full` and the cotangent `ΔD` of `eig_vals`.

In particular, it is assumed that `A ≈ V * D * inv(V)` with thus `size(A) == size(V) == size(D)`
and `D` diagonal. For the cotangents, an arbitrary number of eigenvalues can be missing, i.e.
for a matrix `A` of size `(n, n)`, `diagview(ΔD)` can have length `pD`. In those cases,
additionally `ind` is required to specify which eigenvalues are present in `ΔV` or `ΔD`.
By default, it is assumed that all eigenvectors and eigenvalues are present.
"""
function eig_vals_pullback!(
        ΔA, A, DV, ΔD, ind = Colon();
        degeneracy_atol::Real = default_pullback_rank_atol(DV[1]),
    )

    ΔDV = (diagonal(ΔD), nothing)
    return eig_pullback!(ΔA, A, DV, ΔDV, ind; degeneracy_atol)
end

"""
    remove_eig_gauge_dependence!(ΔV, D, V; degeneracy_atol = ...)

Remove the gauge-dependent part from the cotangent `ΔV` of the eigenvector matrix `V`. The
eigenvectors are only determined up to a scalar factor (or an abitrary linear transformation
across eigenvectors associated with degenerate eigenvalues), so the corresponding components of
`ΔV` are projected out.
"""
function remove_eig_gauge_dependence!(
        ΔV, D, V;
        degeneracy_atol = MatrixAlgebraKit.default_pullback_gauge_atol(D)
    )
    Ddiag = diagview(D)
    gaugepart = V' * ΔV
    gaugepart[abs.(transpose(Ddiag) .- Ddiag) .>= degeneracy_atol] .= 0
    ViG = V / LinearAlgebra.cholesky!(Hermitian(V' * V))
    mul!(ΔV, ViG, gaugepart, -1, 1)
    return ΔV
end
