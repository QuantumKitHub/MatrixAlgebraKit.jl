function check_and_prepare_eigh_cotangents(
        D, V, ΔDmat, ΔV, ind = Colon();
        degeneracy_atol::Real = default_pullback_rank_atol(S),
        gauge_atol::Real = default_pullback_gauge_atol(ΔDmat, ΔV)
    )

    # only the columns ind of VᴴΔV and VᴴAΔV are computed; their rows ind follow by antihermiticity
    n, p = size(V)
    ind′ = select_indices(axes(D, 1), ind)
    k = length(ind′)
    if !iszerotangent(ΔV)
        n == size(ΔV, 1) || throw(DimensionMismatch())
        k == size(ΔV, 2) || throw(DimensionMismatch())
        VᴴΔV₁ = V' * ΔV
        if p == n
            ΔV₊ = zero(ΔV)
        else
            ΔV₊ = mul!(copy(ΔV), V, VᴴΔV₁, -1, 1)
        end
        aVᴴΔV₁ = antihermitian_columns!(VᴴΔV₁, ind′)
    else
        ΔV₊ = nothing
        aVᴴΔV₁ = zero!(similar(V, (p, k)))
    end

    Dₖ = D[ind′]
    bc = Base.broadcasted(transpose(Dₖ), D, aVᴴΔV₁) do d₁, d₂, v
        return abs(d₁ - d₂) < degeneracy_atol ? v : zero(v)
    end
    Δgauge = maximum(abs, Base.Broadcast.instantiate(bc); init = abs(zero(eltype(D))))

    Δgauge ≤ gauge_atol ||
        @warn "`eigh` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"

    aVᴴΔV₁ .*= inv_safe.(transpose(Dₖ) .- D, degeneracy_atol)
    VᴴAΔV = aVᴴΔV₁

    if !iszerotangent(ΔDmat)
        ΔD = diagview(ΔDmat)
        k == length(ΔD) || throw(DimensionMismatch())
        VᴴAΔV[ind′ .+ p .* (0:(k - 1))] .+= real.(ΔD) # the entries (ind′[l], l)
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
    iszero(n) && return ΔA

    ΔDmat, ΔV = ΔDV
    VᴴΔAVₖ, = check_and_prepare_eigh_cotangents(
        D, V, ΔDmat, ΔV, ind; degeneracy_atol, gauge_atol
    )

    # VᴴΔAV is Hermitian and nonzero only in its rows and columns ind′, which are VᴴΔAVₖ' and VᴴΔAVₖ.
    # For k ≤ n / 2, applying these two blocks directly, in O(n² k), is faster than forming VᴴΔAV.
    ind′ = select_indices(axes(D, 1), ind)
    if 2 * length(ind′) <= n
        Xʳ = copy(VᴴΔAVₖ)
        Xʳ[ind′, :] .= zero(eltype(Xʳ)) # these entries are part of the columns ind′
        Vₖ = V[:, ind′]
        ΔA = mul!(ΔA, V * VᴴΔAVₖ, Vₖ', 1, 1)
        ΔA = mul!(ΔA, Vₖ, Xʳ' * V', 1, 1)
    else
        if is_leading_index(ind′, n) # NOTE: all columns in order (e.g. `ind = Colon()`): the original path
            VᴴΔAV = VᴴΔAVₖ
        else
            VᴴΔAV = zero!(similar(VᴴΔAVₖ, (n, n)))
            VᴴΔAV[ind′, :] .= VᴴΔAVₖ'
            VᴴΔAV[:, ind′] .= VᴴΔAVₖ
        end
        ΔA = mul!(ΔA, V * VᴴΔAV, V', 1, 1)
    end
    return ΔA
end
# Diagonal: do not specialize on `A`, since we may insert `A = nothing` to assert independence of `A` in the implementation
function eigh_pullback!(
        ΔA::Diagonal, A, DV, ΔDV, ind = Colon();
        degeneracy_atol::Real = default_pullback_rank_atol(DV[1]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔDV[2])
    )
    # TODO:
    # If A and ΔA are diagonal, then V is a permutation matrix and so is inv(V) = V'.
    # Furthermore, since V̇ is 0, the pullback ΔV cannot contribute and we only have to unpermute ΔD.
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
    iszero(p) && return ΔA

    ΔDmat, ΔV = ΔDV
    VᴴΔAV, ΔV₊ = check_and_prepare_eigh_cotangents(
        D, V, ΔDmat, ΔV; degeneracy_atol, gauge_atol
    )
    Z = V * VᴴΔAV
    if !iszerotangent(ΔV₊)
        X₀ = rdiv!(ΔV₊, Diagonal(D))
        AP = mul!(copy(A), V * Dmat, V', -1, 1)
        X = accelerative_smith_iteration!(X₀, similar(X₀), AP, inv.(D), degeneracy_atol, maxiter)
        Z .+= X
        # we cannot directly multiply Z * V' into ΔA, because we have to
        # take the Hermitian part, and cannot apply project_hermitian! to
        # the current contents of ΔA
        # TODO: add an `add_project_hermitian!`
        # recycle AP's storage, but overwrite it: `accelerative_smith_iteration!` may leave a power of AP in it
        ΔA′ = project_hermitian!(mul!(AP, Z, V'))
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
