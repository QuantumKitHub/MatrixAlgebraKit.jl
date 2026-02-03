function svd_pushforward!(ΔA, A, USVᴴ, ΔUSVᴴ; rank_atol = default_pullback_rank_atol(A), kwargs...)
    U, Smat, Vᴴ = USVᴴ
    m, n = size(U, 1), size(Vᴴ, 2)
    (m, n) == size(ΔA) || throw(DimensionMismatch("size of ΔA ($(size(ΔA))) does not match size of U*S*Vᴴ ($m, $n)"))
    minmn = min(m, n)
    S = diagview(Smat)
    ΔU, ΔS, ΔVᴴ = ΔUSVᴴ
    r = searchsortedlast(S, rank_atol; rev = true) # rank

    vΔU = view(ΔU, :, 1:r)
    vΔS = view(ΔS, 1:r, 1:r)
    vΔVᴴ = view(ΔVᴴ, 1:r, :)

    vU = view(U, :, 1:r)
    vS = view(S, 1:r)
    vSmat = view(Smat, 1:r, 1:r)
    vVᴴ = view(Vᴴ, 1:r, :)

    # compact region
    vV = adjoint(vVᴴ)
    UΔAV = vU' * ΔA * vV
    copyto!(diagview(vΔS), diag(real.(UΔAV)))
    F = one(eltype(S)) ./ (transpose(vS) .- vS)
    G = one(eltype(S)) ./ (transpose(vS) .+ vS)
    diagview(F) .= zero(eltype(F))
    hUΔAV = F .* (UΔAV + UΔAV') ./ 2
    aUΔAV = G .* (UΔAV - UΔAV') ./ 2
    K̇ = hUΔAV + aUΔAV
    Ṁ = hUΔAV - aUΔAV

    # check gauge condition
    @assert isantihermitian(K̇)
    @assert isantihermitian(Ṁ)
    K̇diag = diagview(K̇)
    for i in 1:length(K̇diag)
        @assert K̇diag[i] ≈ (im / 2) * imag(diagview(UΔAV)[i]) / S[i]
    end

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
        superKM = -sylvester(UÃÃV, Smat, rhs)
        K̇perp = view(superKM, 1:size(aUAV, 2))
        Ṁperp = view(superKM, (size(aUAV, 2) + 1):(size(aUAV, 1) + size(aUAV, 2)))
        ∂U .+= Uperp * K̇perp
        ∂V .+= Vᴴperp * Ṁperp
    else
        ImUU = (LinearAlgebra.diagm(ones(eltype(U), m)) - vU * vU')
        ImVV = (LinearAlgebra.diagm(ones(eltype(Vᴴ), n)) - vV * vVᴴ)
        upper = ImUU * ΔA * vV
        lower = ImVV * ΔA' * vU
        rhs = vcat(upper, lower)

        Ã = ImUU * A * ImVV
        ÃÃ = similar(A, (m + n, m + n))
        fill!(ÃÃ, 0)
        view(ÃÃ, (1:m), m .+ (1:n)) .= Ã
        view(ÃÃ, m .+ (1:n), 1:m) .= Ã'

        superLN = -sylvester(ÃÃ, vSmat, rhs)
        ∂U += view(superLN, 1:size(upper, 1), :)
        ∂V += view(superLN, (size(upper, 1) + 1):(size(upper, 1) + size(lower, 1)), :)
    end
    copyto!(vΔU, ∂U)
    adjoint!(vΔVᴴ, ∂V)
    return (ΔU, ΔS, ΔVᴴ)
end

function svd_trunc_pushforward!(ΔA, A, USVᴴ, ΔUSVᴴ, ind; rank_atol = default_pullback_rank_atol(A), kwargs...)

end
