"""
    eigh_pullback!(ΔA, DV, ΔDV, ind=nothing;
                    tol=default_pullback_gaugetol(DV[1]),
                    degeneracy_atol=tol,
                    gauge_atol=tol)

Adds the pullback from the Hermitian eigenvalue decomposition of `A` to `ΔA`,
given the output `DV` of `eigh_full` and the cotangent `ΔDV` of `eigh_full` or `eigh_trunc`.

In particular, it is assumed that `A ≈ V * D * V'` with thus `size(A) == size(V) == size(D)`
and `D` diagonal. For the cotangents, an arbitrary number of eigenvectors or eigenvalues can
be missing, i.e. for a matrix `A` of size `(n, n)`, `ΔV` can have size `(n, pV)` and
`diagview(ΔD)` can have length `pD`. In those cases, it is assumed that these
correspond to the first `pV` or `pD` eigenvectors or values, unless `ind` is provided,
in which case it is assumed that they correspond to the eigenvectors or values with
indices `ind`, and thus `length(ind) == pV == pD`.

A warning will be printed if the cotangents are not gauge-invariant, i.e. if the
anti-hermitian part of `V' * ΔV`, restricted to rows `i` and columns `j`
for which `abs(D[i] - D[j]) < degeneracy_atol`, is not small compared to `gauge_atol`.
"""
function eigh_pullback!(ΔA::AbstractMatrix, DV, ΔDV, ind=nothing;
                             tol::Real=default_pullback_gaugetol(DV[1]),
                             degeneracy_atol::Real=tol,
                             gauge_atol::Real=tol)

    # Basic size checks and determination
    Dmat, V = DV
    D = diagview(Dmat)
    ΔDmat, ΔV = ΔDV
    n = LinearAlgebra.checksquare(V)
    n == length(D) || throw(DimensionMismatch())

    if !iszerotangent(ΔV)
        n == size(ΔV, 1) || throw(DimensionMismatch())
        pV = size(ΔV, 2)
        VᴴΔV = fill!(similar(V), 0)
        if isnothing(ind)
            indV = 1:pV # default assumption?
        else
            length(ind) == pV || throw(DimensionMismatch())
            indV = ind
        end
        mul!(view(VᴴΔV, :, indV), V', ΔV)
        aVᴴΔV = rmul!(VᴴΔV - VᴴΔV', 1 / 2)

        mask = abs.(D' .- D) .< degeneracy_atol
        Δgauge = norm(view(aVᴴΔV, mask))
        Δgauge < gauge_atol ||
            @warn "`eigh` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"

        aVᴴΔV .*= inv_safe.(D' .- D, tol)

        if !iszerotangent(ΔDmat)
            ΔDvec = diagview(ΔDmat)
            pD = length(ΔDvec)
            if isnothing(ind)
                indD = 1:pD # default assumption?
            else
                length(ind) == pD || throw(DimensionMismatch())
                indD = ind
            end
            view(diagview(aVᴴΔV), indD) .+= real.(ΔDvec)
        end
        # recylce VdΔV space
        ΔA = mul!(ΔA, mul!(VᴴΔV, V, aVᴴΔV), V', 1, 1)
    elseif !iszerotangent(ΔDmat)
        ΔDvec = diagview(ΔDmat)
        pD = length(ΔDvec)
        if isnothing(ind)
            indD = 1:pD # default assumption?
        else
            length(ind) == pD || throw(DimensionMismatch())
            indD = ind
        end
        Vp = view(V, :, indD)
        ΔA = mul!(ΔA, Vp * Diagonal(real(ΔDvec)), Vp', 1, 1)
    end
    return ΔA
end
