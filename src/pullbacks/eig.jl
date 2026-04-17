function check_eig_cotangents(
        D, VᴴΔV;
        degeneracy_atol::Real = default_pullback_rank_atol(D),
        gauge_atol::Real = default_pullback_gauge_atol(VᴴΔV)
    )
    mask = abs.(transpose(D) .- D) .< degeneracy_atol
    Δgauge = norm(view(VᴴΔV, mask))
    Δgauge ≤ gauge_atol ||
        @warn "`eig` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
    return
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

        check_eig_cotangents(D, VᴴΔV; degeneracy_atol, gauge_atol)

        VᴴΔV .*= conj.(inv_safe.(transpose(D) .- D, degeneracy_atol))

        if !iszerotangent(ΔDmat)
            ΔDvec = diagview(ΔDmat)
            pD = length(ΔDvec)
            indD = axes(D, 1)[ind]
            length(indD) == pD || throw(DimensionMismatch())
            view(diagview(VᴴΔV), indD) .+= ΔDvec
        end
        PΔV = V' \ VᴴΔV
        if eltype(ΔA) <: Real
            ΔAc = mul!(VᴴΔV, PΔV, V') # recycle VdΔV memory
            ΔA .+= real.(ΔAc)
        else
            ΔA = mul!(ΔA, PΔV, V', 1, 1)
        end
    elseif !iszerotangent(ΔDmat)
        ΔDvec = diagview(ΔDmat)
        pD = length(ΔDvec)
        indD = axes(D, 1)[ind]
        length(indD) == pD || throw(DimensionMismatch())
        Vp = view(V, :, indD)
        PΔV = Vp' \ Diagonal(ΔDvec)
        if eltype(ΔA) <: Real
            ΔAc = PΔV * Vp'
            ΔA .+= real.(ΔAc)
        else
            ΔA = mul!(ΔA, PΔV, V', 1, 1)
        end
    end
    return ΔA
end
function eig_pullback!(
        ΔA::Diagonal, A, DV, ΔDV, ind = Colon();
        degeneracy_atol::Real = default_pullback_rank_atol(DV[1]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔDV[2])
    )
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
        gauge_atol::Real = default_pullback_gauge_atol(ΔDV[2])
    )

    # Basic size checks and determination
    Dmat, V = DV
    D = diagview(Dmat)
    ΔDmat, ΔV = ΔDV
    (n, p) = size(V)
    p == length(D) || throw(DimensionMismatch())
    (n, n) == size(ΔA) || throw(DimensionMismatch())
    G = V' * V

    if !iszerotangent(ΔV)
        (n, p) == size(ΔV) || throw(DimensionMismatch())
        VᴴΔV = V' * ΔV
        check_eig_cotangents(D, VᴴΔV; degeneracy_atol, gauge_atol)

        ΔVperp = ΔV - V * inv(G) * VᴴΔV
        VᴴΔV .*= conj.(inv_safe.(transpose(D) .- D, degeneracy_atol))
    else
        VᴴΔV = zero(G)
    end

    if !iszerotangent(ΔDmat)
        ΔDvec = diagview(ΔDmat)
        p == length(ΔDvec) || throw(DimensionMismatch())
        diagview(VᴴΔV) .+= ΔDvec
    end
    Z = V' \ VᴴΔV

    # add contribution from orthogonal complement
    PA = A - (A * V) / V
    Y = mul!(ΔVperp, PA', Z, 1, 1)
    X = _sylvester(PA', -Dmat', Y)
    Z .+= X

    if eltype(ΔA) <: Real
        ΔAc = Z * V'
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
        ΔV, D, V, ind = axes(ΔV, 2);
        degeneracy_atol = MatrixAlgebraKit.default_pullback_gauge_atol(D)
    )
    length(ind) == size(ΔV, 2) || throw(DimensionMismatch())
    indV = axes(V, 2)[ind]
    Vp = view(V, :, indV)
    Ddiag = view(diagview(D), indV)
    gaugepart = Vp' * ΔV
    gaugepart[abs.(transpose(Ddiag) .- Ddiag) .>= degeneracy_atol] .= 0
    mul!(ΔV, Vp / (Vp' * Vp), gaugepart, -1, 1)
    return ΔV
end
