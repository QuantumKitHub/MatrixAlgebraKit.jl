"""
    eig_pullback!(ΔA, DV, ΔDV, ind=nothing;
                    tol=default_pullback_gaugetol(DV[1]),
                    degeneracy_atol=tol,
                    gauge_atol=tol)

Adds the pullback from the Hermitian eigenvalue decomposition of `A` to `ΔA`,
given the output `DV` of `eigh_full` and the cotangent `ΔDV` of `eig_full` or `eig_trunc`.

In particular, it is assumed that `A ≈ V * D * inv(V)` with thus `size(A) == size(V) == size(D)`
and `D` diagonal. For the cotangents, an arbitrary number of eigenvectors or eigenvalues can
be missing, i.e. for a matrix `A` of size `(n, n)`, `ΔV` can have size `(n, pV)` and
`diagview(ΔD)` can have length `pD`. In those cases, it is assumed that these
correspond to the first `pV` or `pD` eigenvectors or values, unless `ind` is provided,
in which case it is assumed that they correspond to the eigenvectors or values with
indices `ind`, and thus `length(ind) == pV == pD`.

A warning will be printed if the cotangents are not gauge-invariant, i.e. if the
restriction of `V' * ΔV` to rows `i` and columns `j` for which
`abs(D[i] - D[j]) < degeneracy_atol`, is not small compared to `gauge_atol`.
"""
function eig_pullback!(ΔA::AbstractMatrix, DV, ΔDV, ind=nothing;
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

        mask = abs.(transpose(D) .- D) .< degeneracy_atol
        Δgauge = norm(view(VᴴΔV, mask), Inf)
        Δgauge < gauge_atol ||
            @warn "`eig` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"

        VᴴΔV .*= conj.(inv_safe.(transpose(D) .- D, degeneracy_atol))

        if !iszerotangent(ΔDmat)
            ΔDvec = diagview(ΔDmat)
            pD = length(ΔDvec)
            if isnothing(ind)
                indD = 1:pD # default assumption?
            else
                length(ind) == pD || throw(DimensionMismatch())
                indD = ind
            end
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
        if isnothing(ind)
            indD = 1:pD # default assumption?
        else
            length(ind) == pD || throw(DimensionMismatch())
            indD = ind
        end
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
