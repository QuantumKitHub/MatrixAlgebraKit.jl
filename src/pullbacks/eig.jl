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

        mask = abs.(transpose(D) .- D) .< degeneracy_atol
        Δgauge = norm(view(VᴴΔV, mask), Inf)
        Δgauge ≤ gauge_atol ||
            @warn "`eig` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"

        VᴴΔV ./= conj.(transpose(D) .- D)
        diagview(VᴴΔV) .= zero(eltype(VᴴΔV))

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
        mask = abs.(transpose(D) .- D) .< degeneracy_atol
        Δgauge = norm(view(VᴴΔV, mask), Inf)
        Δgauge ≤ gauge_atol ||
            @warn "`eig` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"

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
    X = sylvester(PA', -Dmat', Y)
    Z .+= X

    if eltype(ΔA) <: Real
        ΔAc = Z * V'
        ΔA .+= real.(ΔAc)
    else
        ΔA = mul!(ΔA, Z, V', 1, 1)
    end
    return ΔA
end
