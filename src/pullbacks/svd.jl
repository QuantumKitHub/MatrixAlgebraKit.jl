"""
    svd_pullback!(
        ΔA, A, USVᴴ, ΔUSVᴴ, [ind];
        tol::Real=default_pullback_gaugetol(USVᴴ[2]),
        rank_atol::Real = tol,
        degeneracy_atol::Real = tol,
        gauge_atol::Real = tol
    )

Adds the pullback from the SVD of `A` to `ΔA` given the output USVᴴ of `svd_compact` or
`svd_full` and the cotangent `ΔUSVᴴ` of `svd_compact`, `svd_full` or `svd_trunc`.

In particular, it is assumed that `A ≈ U * S * Vᴴ`, or thus, that no singular values with
magnitude less than `rank_atol` are missing from `S`.  For the cotangents, an arbitrary
number of singular vectors or singular values can be missing, i.e. for a matrix `A` with
size `(m, n)`, `ΔU` and `ΔVᴴ` can have sizes `(m, pU)` and `(pV, n)` respectively, whereas
`diagview(ΔS)` can have length `pS`. In those cases, additionally `ind` is required to
specify which singular vectors and values are present in `ΔU`, `ΔS` and `ΔVᴴ`.

A warning will be printed if the cotangents are not gauge-invariant, i.e. if the
anti-hermitian part of `U' * ΔU + Vᴴ * ΔVᴴ'`, restricted to rows `i` and columns `j` for
which `abs(S[i] - S[j]) < degeneracy_atol`, is not small compared to `gauge_atol`.
"""
function svd_pullback!(
        ΔA::AbstractMatrix, A, USVᴴ, ΔUSVᴴ, ind = Colon();
        tol::Real = default_pullback_gaugetol(USVᴴ[2]),
        rank_atol::Real = tol,
        degeneracy_atol::Real = tol,
        gauge_atol::Real = tol
    )
    # Extract the SVD components
    U, Smat, Vᴴ = USVᴴ
    m, n = size(U, 1), size(Vᴴ, 2)
    (m, n) == size(ΔA) || throw(DimensionMismatch("size of ΔA ($(size(ΔA))) does not match size of U*S*Vᴴ ($m, $n)"))
    minmn = min(m, n)
    S = diagview(Smat)
    length(S) == minmn || throw(DimensionMismatch("length of S ($(length(S))) does not matrix minimum dimension of U, Vᴴ ($minmn)"))
    r = searchsortedlast(S, rank_atol; rev = true) # rank
    Ur = view(U, :, 1:r)
    Vᴴr = view(Vᴴ, 1:r, :)
    Sr = view(S, 1:r)

    # Extract and check the cotangents
    ΔU, ΔSmat, ΔVᴴ = ΔUSVᴴ
    UΔU = fill!(similar(U, (r, r)), 0)
    VΔV = fill!(similar(Vᴴ, (r, r)), 0)
    if !iszerotangent(ΔU)
        m == size(ΔU, 1) || throw(DimensionMismatch("first dimension of ΔU ($(size(ΔU, 1))) does not match first dimension of U ($m)"))
        pU = size(ΔU, 2)
        pU > r && throw(DimensionMismatch("second dimension of ΔU ($(size(ΔU, 2))) does not match rank of S ($r)"))
        indU = axes(U, 2)[ind]
        length(indU) == pU || throw(DimensionMismatch("length of selected U columns ($(length(indU))) does not match second dimension of ΔU ($(size(ΔU, 2)))"))
        UΔUp = view(UΔU, :, indU)
        mul!(UΔUp, Ur', ΔU)
        # ΔU -= Ur * UΔUp but one less allocation without overwriting ΔU
        ΔU = mul!(copy(ΔU), Ur, UΔUp, -1, 1)
    end
    if !iszerotangent(ΔVᴴ)
        n == size(ΔVᴴ, 2) || throw(DimensionMismatch("second dimension of ΔVᴴ ($(size(ΔVᴴ, 2))) does not match second dimension of Vᴴ ($n)"))
        pV = size(ΔVᴴ, 1)
        pV > r && throw(DimensionMismatch("first dimension of ΔVᴴ ($(size(ΔVᴴ, 1))) does not match rank of S ($r)"))
        indV = axes(Vᴴ, 1)[ind]
        length(indV) == pV || throw(DimensionMismatch("length of selected Vᴴ rows ($(length(indV))) does not match first dimension of ΔVᴴ ($(size(ΔVᴴ, 1)))"))
        VΔVp = view(VΔV, :, indV)
        mul!(VΔVp, Vᴴr, ΔVᴴ')
        # ΔVᴴ -= VΔVp' * Vᴴr but one less allocation without overwriting ΔVᴴ
        ΔVᴴ = mul!(copy(ΔVᴴ), VΔVp', Vᴴr, -1, 1)
    end

    # Project onto antihermitian part; hermitian part outside of Grassmann tangent space
    aUΔU = project_antihermitian!(UΔU)
    aVΔV = project_antihermitian!(VΔV)

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
        indS = axes(S, 1)[ind]
        #length(indS) == pS || throw(DimensionMismatch("length of selected S diagonals ($(length(indS))) does not match length of ΔS diagonal ($(length(ΔS)))"))
        view(diagview(UdΔAV), indS) .+= real.(ΔS)
    end
    ΔA = mul!(ΔA, Ur, UdΔAV * Vᴴr, 1, 1) # add the contribution to ΔA

    # Add the remaining contributions
    if m > r && !iszerotangent(ΔU) # remaining ΔU is already orthogonal to Ur
        Sp = view(S, indU)
        Vᴴp = view(Vᴴ, indU, :)
        ΔA = mul!(ΔA, ΔU ./ Sp', Vᴴp, 1, 1)
    end
    if n > r && !iszerotangent(ΔVᴴ) # remaining ΔV is already orthogonal to Vᴴr
        Sp = view(S, indV)
        Up = view(U, :, indV)
        ΔA = mul!(ΔA, Up, Sp .\ ΔVᴴ, 1, 1)
    end
    return ΔA
end

"""
    svd_trunc_pullback!(
        ΔA, A, USVᴴ, ΔUSVᴴ;
        tol::Real=default_pullback_gaugetol(S),
        rank_atol::Real = tol,
        degeneracy_atol::Real = tol,
        gauge_atol::Real = tol
    )

Adds the pullback from the truncated SVD of `A` to `ΔA`, given the output `USVᴴ` and the
cotangent `ΔUSVᴴ` of `svd_trunc`.

In particular, it is assumed that `A * Vᴴ' ≈ U * S` and `U' * A = S * Vᴴ`, with `U` and `Vᴴ`
rectangular matrices of left and right singular vectors, and `S` diagonal. For the
cotangents, it is assumed that if `ΔU` and `ΔVᴴ` are not zero, then they have the same size
as `U` and `Vᴴ` (respectively), and if `ΔS` is not zero, then it is a diagonal matrix of the
same size as `S`. For this method to work correctly, it is also assumed that the remaining
singular values (not included in `S`) are (sufficiently) separated from those in `S`.

A warning will be printed if the cotangents are not gauge-invariant, i.e. if the
anti-hermitian part of `U' * ΔU + Vᴴ * ΔVᴴ'`, restricted to rows `i` and columns `j` for
which `abs(S[i] - S[j]) < degeneracy_atol`, is not small compared to `gauge_atol`.
"""
function svd_trunc_pullback!(
        ΔA::AbstractMatrix, A, USVᴴ, ΔUSVᴴ;
        tol::Real = default_pullback_gaugetol(USVᴴ[2]),
        rank_atol::Real = tol,
        degeneracy_atol::Real = tol,
        gauge_atol::Real = tol
    )

    # Extract the SVD components
    U, Smat, Vᴴ = USVᴴ
    m, n = size(U, 1), size(Vᴴ, 2)
    (m, n) == size(ΔA) || throw(DimensionMismatch())
    p = size(U, 2)
    p == size(Vᴴ, 1) || throw(DimensionMismatch())
    S = diagview(Smat)
    p == length(S) || throw(DimensionMismatch())

    # Extract and check the cotangents
    ΔU, ΔSmat, ΔVᴴ = ΔUSVᴴ
    UΔU = fill!(similar(U, (p, p)), 0)
    VΔV = fill!(similar(Vᴴ, (p, p)), 0)
    if !iszerotangent(ΔU)
        (m, p) == size(ΔU) || throw(DimensionMismatch())
        mul!(UΔU, U', ΔU)
    end
    if !iszerotangent(ΔVᴴ)
        (p, n) == size(ΔVᴴ) || throw(DimensionMismatch())
        mul!(VΔV, Vᴴ, ΔVᴴ')
        # ΔVᴴ -= VΔVp' * Vᴴr but one less allocation without overwriting ΔVᴴ
        ΔVᴴ = mul!(copy(ΔVᴴ), VΔV', Vᴴ, -1, 1)
    end

    # Project onto antihermitian part; hermitian part outside of Grassmann tangent space
    aUΔU = project_antihermitian!(UΔU)
    aVΔV = project_antihermitian!(VΔV)

    # check whether cotangents arise from gauge-invariance objective function
    mask = abs.(S' .- S) .< degeneracy_atol
    Δgauge = norm(view(aUΔU, mask) + view(aVΔV, mask), Inf)
    Δgauge < gauge_atol ||
        @warn "`svd` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"

    UdΔAV = (aUΔU .+ aVΔV) .* inv_safe.(S' .- S, degeneracy_atol) .+
        (aUΔU .- aVΔV) .* inv_safe.(S' .+ S, degeneracy_atol)
    if !iszerotangent(ΔSmat)
        ΔS = diagview(ΔSmat)
        p == length(ΔS) || throw(DimensionMismatch())
        diagview(UdΔAV) .+= real.(ΔS)
    end
    ΔA = mul!(ΔA, U, UdΔAV * Vᴴ, 1, 1) # add the contribution to ΔA

    # add contribution from orthogonal complement
    Ũ = qr_null(U)
    Ṽᴴ = lq_null(Vᴴ)
    m̃ = m - p
    ñ = n - p
    Ã = Ũ' * A * Ṽᴴ'
    ÃÃ = similar(A, (m̃ + ñ, m̃ + ñ))
    fill!(ÃÃ, 0)
    view(ÃÃ, (1:m̃), m̃ .+ (1:ñ)) .= Ã
    view(ÃÃ, m̃ .+ (1:ñ), 1:m̃) .= Ã'

    rhs = similar(Ũ, (m̃ + ñ, p))
    if !iszerotangent(ΔU)
        mul!(view(rhs, 1:m̃, :), Ũ', ΔU)
    else
        fill!(view(rhs, 1:m̃, :), 0)
    end
    if !iszerotangent(ΔVᴴ)
        mul!(view(rhs, m̃ .+ (1:ñ), :), Ṽᴴ, ΔVᴴ')
    else
        fill!(view(rhs, m̃ .+ (1:ñ), :), 0)
    end
    XY = sylvester(ÃÃ, -Smat, rhs)
    X = view(XY, 1:m̃, :)
    Y = view(XY, m̃ .+ (1:ñ), :)
    ΔA = mul!(ΔA, Ũ, X * Vᴴ, 1, 1)
    ΔA = mul!(ΔA, U, Y' * Ṽᴴ, 1, 1)
    return ΔA
end
