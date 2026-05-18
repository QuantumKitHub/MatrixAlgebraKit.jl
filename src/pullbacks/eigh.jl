function check_and_prepare_eigh_cotangents(
        D, V, ΔDmat, ΔV, ind = Colon();
        degeneracy_atol::Real = default_pullback_rank_atol(S),
        gauge_atol::Real = default_pullback_gauge_atol(ΔDmat, ΔV)
    )

    n, p = size(V)
    indD = axes(D, 1)[ind]
    indV = axes(V, 2)[ind]
    if !iszerotangent(ΔV)
        n == size(ΔV, 1) || throw(DimensionMismatch())
        length(indV) == size(ΔV, 2) || throw(DimensionMismatch())
        if indV == 1:p
            ΔV₁ = copy(ΔV)
        else
            ΔV₁ = zero(V)
            ΔV₁[:, indV] = ΔV
        end
        VᴴΔV₁ = V' * ΔV₁
        if p == n
            ΔV₊ = zero!(ΔV₁)
        else
            ΔV₊ = mul!(ΔV₁, V, VᴴΔV₁, -1, 1)
        end
        aVᴴΔV₁ = project_antihermitian!(VᴴΔV₁)
    else
        ΔV₊ = nothing
        aVᴴΔV₁ = zero!(similar(V, (p, p)))
    end

    if !isempty(D) # norm(bc, Inf) calls eltype on empty inputs
        bc = Base.broadcasted(transpose(D), D, aVᴴΔV₁) do d₁, d₂, v
            return abs(d₁ - d₂) < degeneracy_atol ? v : zero(v)
        end
        Δgauge = norm(bc, Inf)
        Δgauge ≤ gauge_atol ||
            @warn "`eigh` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
    end

    aVᴴΔV₁ .*= inv_safe.(D' .- D, degeneracy_atol)
    VᴴAΔV = aVᴴΔV₁

    if !iszerotangent(ΔDmat)
        ΔD = diagview(ΔDmat)
        length(indD) == length(ΔD) || throw(DimensionMismatch())
        view(diagview(VᴴAΔV), indD) .+= real.(ΔD)
    else
        ΔD = nothing
    end

    return VᴴAΔV, ΔV₊
end

"""
    eigh_pullback!(
        ΔA::AbstractMatrix, A, DV, ΔDV, [ind];
        degeneracy_atol::Real = default_pullback_rank_atol(DV[1]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔDV[2])
    )

Adds the pullback from the Hermitian eigenvalue decomposition of `A` to `ΔA`, given the
output `DV` of `eigh_full` and the cotangent `ΔDV` of `eigh_full` or `eigh_trunc`.

In particular, it is assumed that `A ≈ V * D * V'` with thus `size(A) == size(V) == size(D)`
and `D` diagonal. For the cotangents, an arbitrary number of eigenvectors or eigenvalues can
be missing, i.e. for a matrix `A` of size `(n, n)`, `ΔV` can have size `(n, pV)` and
`diagview(ΔD)` can have length `pD`. In those cases, additionally `ind` is required to
specify which eigenvectors or eigenvalues are present in `ΔV` or `ΔD`. By default, it is
assumed that all eigenvectors and eigenvalues are present.

A warning will be printed if the cotangents are not gauge-invariant, i.e. if the
anti-hermitian part of `V' * ΔV`, restricted to rows `i` and columns `j` for which `abs(D[i]
- D[j]) < degeneracy_atol`, is not small compared to `gauge_atol`.
"""
function eigh_pullback!(
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

    ΔDmat, ΔV = ΔDV
    VᴴΔAV, = check_and_prepare_eigh_cotangents(
        D, V, ΔDmat, ΔV, ind; degeneracy_atol, gauge_atol
    )

    ΔA = mul!(ΔA, V * VᴴΔAV, V', 1, 1)
    return ΔA
end
function eigh_pullback!(
        ΔA::Diagonal, A, DV, ΔDV, ind = Colon();
        degeneracy_atol::Real = default_pullback_rank_atol(DV[1]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔDV[2])
    )
    ΔA_full = zero!(similar(ΔA, size(ΔA)))
    ΔA_full = eigh_pullback!(ΔA_full, A, DV, ΔDV, ind; degeneracy_atol, gauge_atol)
    diagview(ΔA) .+= diagview(ΔA_full)
    return ΔA
end

"""
    eigh_trunc_pullback!(
        ΔA::AbstractMatrix, A, DV, ΔDV;
        degeneracy_atol::Real = default_pullback_rank_atol(DV[1]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔDV[2])
    )

Adds the pullback from the truncated Hermitian eigenvalue decomposition of `A` to `ΔA`,
given the output `DV` and the cotangent `ΔDV` of `eig_trunc`.

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
function eigh_trunc_pullback!(
        ΔA::AbstractMatrix, A, DV, ΔDV;
        degeneracy_atol::Real = default_pullback_rank_atol(DV[1]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔDV[2]),
        maxiter::Int = 100 # TODO: better default, depending on expected number of steps using quadratic convergence?
    )

    # Basic size checks and determination
    Dmat, V = DV
    (n, p) = size(V)
    D = diagview(Dmat)
    p == length(D) || throw(DimensionMismatch())
    (n, n) == size(ΔA) || throw(DimensionMismatch())

    ΔDmat, ΔV = ΔDV
    VᴴΔAV, ΔV₊ = check_and_prepare_eigh_cotangents(
        D, V, ΔDmat, ΔV; degeneracy_atol, gauge_atol
    )
    Z = V * VᴴΔAV
    if !iszerotangent(ΔV₊)
        X₀ = rdiv!(ΔV₊, Diagonal(D))
        AP = mul!(copy(A), V * Dmat, V', -1, 1)
        dabsmax = maximum(abs, D)
        AP ./= dabsmax
        D⁻¹ = dabsmax ./ D
        X₁ = rmul!(AP * X₀, Diagonal(D⁻¹))
        X₁ .+= X₀
        Xₖ, Xₖ₊₁ = X₁, X₀
        APₖ, APₖ₊₁ = AP * AP, AP
        D⁻¹ₖ, D⁻¹ₖ₊₁ = D⁻¹ .^ 2, D⁻¹
        for k in 1:maxiter
            Xₖ₊₁ = rmul!(mul!(Xₖ₊₁, APₖ, Xₖ), Diagonal(D⁻¹ₖ))
            if norm(Xₖ₊₁, Inf) < degeneracy_atol
                break
            end
            Xₖ₊₁ .+= Xₖ
            if k == maxiter
                @warn "Sylvester iteration did not converge after $k iterations, final norm of X: $(norm(Xₖ₊₁, Inf)))"
                break
            end
            D⁻¹ₖ₊₁ .= D⁻¹ₖ .^ 2
            APₖ₊₁ = mul!(APₖ₊₁, APₖ, APₖ)
            Xₖ, Xₖ₊₁ = Xₖ₊₁, Xₖ
            APₖ, APₖ₊₁ = APₖ₊₁, APₖ
            D⁻¹ₖ, D⁻¹ₖ₊₁ = D⁻¹ₖ₊₁, D⁻¹ₖ
        end
        Z .+= Xₖ
        # we cannot directly multiply Z * V' into ΔA, because we have to
        # take the Hermitian part, and cannot apply project_hermitian! to
        # the current contents of ΔA
        # TODO: add an `add_project_hermitian!`
        ΔA′ = project_hermitian!(mul!(AP, Z, V', 1, 1)) # recycle AP
        ΔA .+= ΔA′
    else
        # in this case, Z * V' is automatically Hermitian, so we can directly add it to ΔA
        ΔA = mul!(ΔA, Z, V', 1, 1)
    end
    return ΔA
end
function eigh_trunc_pullback!(
        ΔA::Diagonal, A, DV, ΔDV;
        degeneracy_atol::Real = default_pullback_rank_atol(DV[1]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔDV[2])
    )
    ΔA_full = zero!(similar(ΔA, size(ΔA)))
    ΔA_full = eigh_trunc_pullback!(ΔA_full, A, DV, ΔDV; degeneracy_atol, gauge_atol)
    diagview(ΔA) .+= diagview(ΔA_full)
    return ΔA
end

"""
    eigh_vals_pullback!(
        ΔA, A, DV, ΔD, [ind];
        degeneracy_atol::Real = default_pullback_rank_atol(DV[1]),
    )

Adds the pullback from the eigenvalues of `A` to `ΔA`, given the output
`DV` of `eigh_full` and the cotangent `ΔD` of `eig_vals`.

In particular, it is assumed that `A ≈ V * D * inv(V)` with thus `size(A) == size(V) == size(D)`
and `D` diagonal. For the cotangents, an arbitrary number of eigenvalues can be missing, i.e.
for a matrix `A` of size `(n, n)`, `diagview(ΔD)` can have length `pD`. In those cases,
additionally `ind` is required to specify which eigenvalues are present in `ΔV` or `ΔD`.
By default, it is assumed that all eigenvectors and eigenvalues are present.
"""
function eigh_vals_pullback!(
        ΔA, A, DV, ΔD, ind = Colon();
        degeneracy_atol::Real = default_pullback_rank_atol(DV[1]),
    )

    ΔDV = (diagonal(ΔD), nothing)
    return eigh_pullback!(ΔA, A, DV, ΔDV, ind; degeneracy_atol)
end

"""
    remove_eigh_gauge_dependence!(ΔV, D, V; degeneracy_atol = ...)

Remove the gauge-dependent part from the cotangent `ΔV` of the Hermitian eigenvector matrix
`V`. The eigenvectors are only determined up to a complex phase (or a unitary transformation
across eigenvectors associated with degenerate eigenvalues), so the corresponding anti-Hermitian
components of `V' * ΔV` are projected out.
"""
function remove_eigh_gauge_dependence!(
        ΔV, D, V;
        degeneracy_atol = MatrixAlgebraKit.default_pullback_gauge_atol(D)
    )
    Ddiag = diagview(D)
    gaugepart = project_antihermitian!(V' * ΔV)
    gaugepart[abs.(transpose(Ddiag) .- Ddiag) .>= degeneracy_atol] .= 0
    mul!(ΔV, V, gaugepart, -1, 1)
    return ΔV
end
