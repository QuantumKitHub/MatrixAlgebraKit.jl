function check_eigh_cotangents(
        D, aVᴴΔV;
        degeneracy_atol::Real = default_pullback_rank_atol(D),
        gauge_atol::Real = default_pullback_gauge_atol(aVᴴΔV)
    )
    mask = abs.(D' .- D) .< degeneracy_atol
    Δgauge = norm(view(aVᴴΔV, mask))
    Δgauge ≤ gauge_atol ||
        @warn "`eigh` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
    return
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
    D = diagview(Dmat)
    ΔDmat, ΔV = ΔDV
    n = LinearAlgebra.checksquare(V)
    n == length(D) || throw(DimensionMismatch())
    (n, n) == size(ΔA) || throw(DimensionMismatch())

    if !iszerotangent(ΔV)
        n == size(ΔV, 1) || throw(DimensionMismatch())
        pV = size(ΔV, 2)
        VᴴΔV = fill!(similar(V), 0)
        indV = axes(V, 2)[ind]
        length(indV) == pV || throw(DimensionMismatch())
        mul!(view(VᴴΔV, :, indV), V', ΔV)
        aVᴴΔV = project_antihermitian(VᴴΔV) # can't use in-place or recycling doesn't work

        check_eigh_cotangents(D, aVᴴΔV; degeneracy_atol, gauge_atol)

        aVᴴΔV .*= inv_safe.(D' .- D, degeneracy_atol)

        if !iszerotangent(ΔDmat)
            ΔDvec = diagview(ΔDmat)
            pD = length(ΔDvec)
            indD = axes(D, 1)[ind]
            length(indD) == pD || throw(DimensionMismatch())
            view(diagview(aVᴴΔV), indD) .+= real.(ΔDvec)
        end
        # recycle VdΔV space
        ΔA = mul!(ΔA, mul!(VᴴΔV, V, aVᴴΔV), V', 1, 1)
    elseif !iszerotangent(ΔDmat)
        ΔDvec = diagview(ΔDmat)
        pD = length(ΔDvec)
        indD = axes(D, 1)[ind]
        length(indD) == pD || throw(DimensionMismatch())
        Vp = view(V, :, indD)
        ΔA = mul!(ΔA, Vp * Diagonal(real(ΔDvec)), Vp', 1, 1)
    end
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
        gauge_atol::Real = default_pullback_gauge_atol(ΔDV[2])
    )

    # Basic size checks and determination
    Dmat, V = DV
    D = diagview(Dmat)
    ΔDmat, ΔV = ΔDV
    (n, p) = size(V)
    p == length(D) || throw(DimensionMismatch())
    (n, n) == size(ΔA) || throw(DimensionMismatch())

    if !iszerotangent(ΔV)
        (n, p) == size(ΔV) || throw(DimensionMismatch())
        VᴴΔV = V' * ΔV
        aVᴴΔV = project_antihermitian!(VᴴΔV)

        check_eigh_cotangents(D, aVᴴΔV; degeneracy_atol, gauge_atol)

        aVᴴΔV .*= inv_safe.(D' .- D, degeneracy_atol)

        if !iszerotangent(ΔDmat)
            ΔDvec = diagview(ΔDmat)
            p == length(ΔDvec) || throw(DimensionMismatch())
            diagview(aVᴴΔV) .+= real.(ΔDvec)
        end

        Z = V * aVᴴΔV

        # add contribution from orthogonal complement
        W = qr_null(V)
        WᴴΔV = W' * ΔV
        X = _sylvester(W' * A * W, -Dmat, WᴴΔV)
        Z = mul!(Z, W, X, 1, 1)

        # put everything together: symmetrize for hermitian case
        ΔA = mul!(ΔA, Z, V', 1 // 2, 1)
        ΔA = mul!(ΔA, V, Z', 1 // 2, 1)
    elseif !iszerotangent(ΔDmat)
        ΔDvec = diagview(ΔDmat)
        p == length(ΔDvec) || throw(DimensionMismatch())
        ΔA = mul!(ΔA, V * Diagonal(real(ΔDvec)), V', 1, 1)
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
        ΔV, D, V, ind = axes(ΔV, 2);
        degeneracy_atol = MatrixAlgebraKit.default_pullback_gauge_atol(D)
    )
    length(ind) == size(ΔV, 2) || throw(DimensionMismatch())
    indV = axes(V, 2)[ind]
    Vp = view(V, :, indV)
    Ddiag = view(diagview(D), indV)
    gaugepart = project_antihermitian!(Vp' * ΔV)
    gaugepart[abs.(transpose(Ddiag) .- Ddiag) .>= degeneracy_atol] .= 0
    mul!(ΔV, Vp, gaugepart, -1, 1)
    return ΔV
end
