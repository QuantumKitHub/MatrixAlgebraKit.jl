function svd_pushforward!(ΔA, A, USVᴴ, ΔUSVᴴ, ind = Colon(); rank_atol = default_pullback_rank_atol(A), kwargs...)
    U, Smat, Vᴴ = USVᴴ
    m, n = size(U, 1), size(Vᴴ, 2)
    (m, n) == size(ΔA) || throw(DimensionMismatch("size of ΔA ($(size(ΔA))) does not match size of U*S*Vᴴ ($m, $n)"))
    minmn = min(m, n)
    S = diagview(Smat)
    ΔU, ΔS, ΔVᴴ = ΔUSVᴴ
    r = svd_rank(S; rank_atol)

    vΔS = view(diagview(ΔS), 1:r)

    vU = view(U, :, 1:r)
    vS = view(S, 1:r)
    vSmat = view(Smat, 1:r, 1:r)
    vVᴴ = view(Vᴴ, 1:r, :)

    # compact region
    vV = adjoint(vVᴴ)
    UΔAV = vU' * ΔA * vV
    copyto!(vΔS, real.(diagview(UΔAV)))
    F = inv_safe.(transpose(vS) .- vS)
    G = inv_safe.(transpose(vS) .+ vS)
    hUΔAV = F .* (UΔAV + UΔAV') ./ 2
    aUΔAV = G .* (UΔAV - UΔAV') ./ 2
    K̇ = hUΔAV + aUΔAV
    Ṁ = hUΔAV - aUΔAV

    # check gauge condition
    @assert isantihermitian(K̇)
    @assert isantihermitian(Ṁ)
    K̇diag = diagview(K̇)

    ∂U = vU * K̇
    ∂V = vV * Ṁ
    # full component
    if size(U, 2) > minmn && size(Vᴴ, 1) > minmn
        Uperp = view(U, :, (minmn + 1):m)
        Vᴴperp = view(Vᴴ, (minmn + 1):n, :)

        aUAV = adjoint(Uperp) * A * adjoint(Vᴴperp)

        UÃÃV = similar(A, (size(aUAV, 1) + size(aUAV, 2), size(aUAV, 1) + size(aUAV, 2)))
        fill!(UÃÃV, 0)
        view(UÃÃV, (1:size(aUAV, 1)), size(aUAV, 1) .+ (1:size(aUAV, 2))) .= aUAV
        view(UÃÃV, size(aUAV, 1) .+ (1:size(aUAV, 2)), 1:size(aUAV, 1)) .= aUAV'
        rhs = vcat(adjoint(Uperp * ΔA * Vᴴ), Vᴴperp * ΔA' * U)
        superKM = -_sylvester(UÃÃV, Smat, rhs)
        K̇perp = view(superKM, 1:size(aUAV, 2))
        Ṁperp = view(superKM, (size(aUAV, 2) + 1):(size(aUAV, 1) + size(aUAV, 2)))
        ∂U .+= Uperp * K̇perp
        ∂V .+= Vᴴperp * Ṁperp
    else
        ImUU = (LinearAlgebra.diagm(one!(similar(U, m))) - vU * vU')
        ImVV = (LinearAlgebra.diagm(one!(similar(Vᴴ, n))) - vV * vVᴴ)
        upper = ImUU * ΔA * vV
        lower = ImVV * ΔA' * vU
        rhs = vcat(upper, lower)

        Ã = ImUU * A * ImVV
        ÃÃ = similar(A, (m + n, m + n))
        fill!(ÃÃ, 0)
        view(ÃÃ, (1:m), m .+ (1:n)) .= Ã
        view(ÃÃ, m .+ (1:n), 1:m) .= Ã'

        superLN = -_sylvester(ÃÃ, vSmat, rhs)
        ∂U += view(superLN, 1:size(upper, 1), :)
        ∂V += view(superLN, (size(upper, 1) + 1):(size(upper, 1) + size(lower, 1)), :)
    end
    if !iszerotangent(ΔU)
        vΔU = view(ΔU, :, 1:r)
        copyto!(vΔU, ∂U)
    end
    if !iszerotangent(ΔVᴴ)
        vΔVᴴ = view(ΔVᴴ, 1:r, :)
        adjoint!(vΔVᴴ, ∂V)
    end
    return (ΔU, ΔS, ΔVᴴ)
end

function svd_trunc_pushforward!(ΔA, A, USVᴴ, ΔUSVᴴ, ind; rank_atol = default_pullback_rank_atol(A), kwargs...)
    # TODO
end

function svd_vals_pushforward!(
        ΔA, A, USVᴴ, ΔS, ind = Colon();
        rank_atol::Real = default_pullback_rank_atol(USVᴴ[2]),
        degeneracy_atol::Real = default_pullback_rank_atol(USVᴴ[2])
    )
    ΔUSVᴴ = (nothing, diagonal(ΔS), nothing)
    return svd_pushforward!(ΔA, A, USVᴴ, ΔUSVᴴ, ind; rank_atol, degeneracy_atol)
end
