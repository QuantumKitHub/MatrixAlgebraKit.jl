"""
    svd_compact_pullback!(ΔA, USVᴴ, ΔUSVᴴ;
                            tol::Real=default_pullback_gaugetol(S),
                            rank_atol::Real = tol,
                            degeneracy_atol::Real = tol,
                            gauge_atol::Real = tol)

Adds the pullback from the SVD of `A` to `ΔA` given the output USVᴴ of `svd_compact`
or `svd_full` and the cotangent `ΔUSVᴴ` of `svd_compact`, `svd_full` or `svd_trunc`.

In particular, it is assumed that `A ≈ U * S * Vᴴ`, or thus, that no singular values
with magnitude less than `rank_atol` are missing from `S`.
For the cotangents, an arbitrary number of singular vectors or singular values can
be missing, i.e. `ΔU` and `ΔVᴴ` can have sizes `(m, pU)` and `(pV, n)` respectively,
whereas `diagview(ΔS)` can have length `pS`.

A warning will be printed if the cotangents are not gauge-invariant, i.e. if the
anti-hermitian part of `U' * ΔU + Vᴴ * ΔVᴴ'`, restricted to rows `i` and columns `j`
for which `abs(S[i] - S[j]) < degeneracy_atol`, is not small compared to `gauge_atol`.
"""
function svd_compact_pullback!(ΔA::AbstractMatrix, USVᴴ, ΔUSVᴴ;
                               tol::Real=default_pullback_gaugetol(USVᴴ[2]),
                               rank_atol::Real=tol,
                               degeneracy_atol::Real=tol,
                               gauge_atol::Real=tol)

    # Extract the SVD components
    U, Smat, Vᴴ = USVᴴ
    m, n = size(U, 1), size(Vᴴ, 2)
    minmn = min(m, n)
    S = diagview(Smat)
    length(S) == minmn || throw(DimensionMismatch())
    r = findlast(>=(rank_atol), S) # rank
    Ur = view(U, :, 1:r)
    Vᴴr = view(Vᴴ, 1:r, :)
    Sr = view(S, 1:r)

    # Extract and check the cotangents
    ΔU, ΔSmat, ΔVᴴ = ΔUSVᴴ
    UΔU = fill!(similar(U, (r, r)), 0)
    VΔV = fill!(similar(Vᴴ, (r, r)), 0)
    if !iszerotangent(ΔU)
        m == size(ΔU, 1) || throw(DimensionMismatch())
        pU = size(ΔU, 2)
        pU > r && throw(DimensionMismatch())
        UΔUp = view(UΔU, :, 1:pU)
        mul!(UΔUp, Ur', ΔU)
        ΔU -= Ur * UΔUp
    end
    if !iszerotangent(ΔVᴴ)
        n == size(ΔVᴴ, 2) || throw(DimensionMismatch())
        pV = size(ΔVᴴ, 1)
        pV > r && throw(DimensionMismatch())
        VΔVp = view(VΔV, :, 1:pV)
        mul!(VΔVp, Vᴴr, ΔVᴴ')
        ΔVᴴ = ΔVᴴ - VΔVp' * Vᴴr
    end

    # Project onto antihermitian part; hermitian part outside of Grassmann tangent space
    aUΔU = rmul!(UΔU - UΔU', 1 / 2)
    aVΔV = rmul!(VΔV - VΔV', 1 / 2)

    # check whether cotangents arise from gauge-invariance objective function
    mask = abs.(Sr' .- Sr) .< degeneracy_atol
    Δgauge = norm(view(aUΔU, mask) + view(aVΔV, mask), Inf)
    Δgauge < gauge_atol ||
        @warn "`svd` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"

    UdΔAV = (aUΔU .+ aVΔV) .* inv_safe.(Sr' .- Sr, degeneracy_atol) .+
            (aUΔU .- aVΔV) .* inv_safe.(Sr' .+ Sr, degeneracy_atol)
    if !iszerotangent(ΔSmat)
        ΔS = diagview(ΔSmat)
        pS = length(ΔS)
        view(diagview(UdΔAV), 1:pS) .+= real.(ΔS)
    end
    ΔA = mul!(ΔA, Ur, UdΔAV * Vᴴr, 1, 1) # add the contribution to ΔA

    # Add the remaining contributions
    if m > r && !iszerotangent(ΔU) # remaining ΔU is already orthogonal to Ur
        Sp = view(S, 1:pU)
        Vᴴp = view(Vᴴ, 1:pU, :)
        ΔA = mul!(ΔA, ΔU ./ Sp', Vᴴp, 1, 1)
    end
    if n > r && !iszerotangent(ΔVᴴ) # remaining ΔV is already orthogonal to Vᴴr
        Sp = view(S, 1:pV)
        Up = view(U, :, 1:pV)
        ΔA = mul!(ΔA, Up, Sp .\ ΔVᴴ, 1, 1)
    end
    return ΔA
end
