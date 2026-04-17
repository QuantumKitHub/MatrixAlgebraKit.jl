svd_rank(S; rank_atol = default_pullback_rank_atol(S)) = searchsortedlast(S, rank_atol; rev = true)

function check_and_prepare_svd_cotangents(
        U, S, Vᴴ, ΔU, ΔSmat, ΔVᴴ, r::Int, ind = Colon();
        degeneracy_atol::Real = default_pullback_rank_atol(S),
        gauge_atol::Real = default_pullback_gauge_atol(ΔU, ΔSmat, ΔVᴴ)
    )

    m, n = size(U, 1), size(Vᴴ, 2)
    minmn = min(m, n)

    U₁ = view(U, :, 1:r)
    V₁ᴴ = view(Vᴴ, 1:r, :)
    S₁ = view(S, 1:r)
    indU = axes(U, 2)[ind]
    indV = axes(Vᴴ, 1)[ind]
    indS = axes(S, 1)[ind]
    Δgauge = zero(eltype(S))

    if !iszerotangent(ΔU)
        ΔgaugeU = zero(eltype(S))
        m == size(ΔU, 1) || throw(DimensionMismatch("first dimension of ΔU ($(size(ΔU, 1))) does not match first dimension of U ($m)"))
        length(indU) == size(ΔU, 2) || throw(DimensionMismatch("length of selected U columns ($(length(indU))) does not match second dimension of ΔU ($(size(ΔU, 2)))"))
        if indU == 1:r
            ΔU₁ = copy(ΔU)
        else
            ΔU₁ = zero(U₁)
            wtmp = similar(U₁, (r,))
            utmp = similar(U₁, (m,))
            for (j, i) in enumerate(indU)
                if i <= r
                    ΔU₁[:, i] .= view(ΔU, :, j)
                elseif r == minmn # full rank case, ΔU₃ contains gauge-invariant information along U₁
                    mul!(wtmp, U₁', view(ΔU, :, j))
                    mul!(ΔU₁, view(U, :, i), wtmp', -1, 1)
                    utmp .= view(ΔU, :, j)
                    mul!(utmp, U₁, wtmp, -1, 1)
                    ΔgaugeU = max(ΔgaugeU, norm(utmp))
                else # remaining columns should be zero
                    ΔgaugeU = max(ΔgaugeU, norm(view(ΔU, :, j), Inf))
                end
            end
        end
        UᴴΔU₁ = U₁' * ΔU₁
        ΔU₁ = mul!(ΔU₁, U₁, UᴴΔU₁, -1, 1)
        aUᴴΔU₁ = project_antihermitian!(UᴴΔU₁)
        Δgauge = max(Δgauge, ΔgaugeU)
    else
        ΔU₁ = nothing
        aUᴴΔU₁ = zero!(similar(U₁, (r, r)))
    end
    if !iszerotangent(ΔVᴴ)
        ΔgaugeV = zero(eltype(S))
        n == size(ΔVᴴ, 2) || throw(DimensionMismatch("second dimension of ΔVᴴ ($(size(ΔVᴴ, 2))) does not match second dimension of Vᴴ ($n)"))
        length(indV) == size(ΔVᴴ, 1) || throw(DimensionMismatch("length of selected Vᴴ rows ($(length(indV))) does not match first dimension of ΔVᴴ ($(size(ΔVᴴ, 1)))"))
        if indV == 1:r
            ΔV₁ᴴ = copy(ΔVᴴ)
        else
            ΔV₁ᴴ = zero(V₁ᴴ)
            wtmp = similar(V₁ᴴ, (1, r))
            vtmp = similar(V₁ᴴ, (1, n))
            for (j, i) in enumerate(indV)
                if i <= r
                    ΔV₁ᴴ[i, :] .= view(ΔVᴴ, j, :)
                elseif r == minmn # full rank case, ΔV₃ contains gauge-invariant information along Vᴴ₁
                    mul!(wtmp, view(ΔVᴴ, j:j, :), V₁ᴴ')
                    mul!(ΔV₁ᴴ, wtmp', view(Vᴴ, i:i, :), -1, 1)
                    vtmp .= view(ΔVᴴ, j:j, :)
                    mul!(vtmp, wtmp, V₁ᴴ, -1, 1)
                    ΔgaugeV = max(ΔgaugeV, norm(vtmp))
                else # remaining rows should be zero
                    ΔgaugeV = max(ΔgaugeV, norm(view(ΔVᴴ, j, :), Inf))
                end
            end
        end
        VᴴΔV₁ = V₁ᴴ * ΔV₁ᴴ'
        ΔV₁ᴴ = mul!(ΔV₁ᴴ, VᴴΔV₁', V₁ᴴ, -1, 1)
        aVᴴΔV₁ = project_antihermitian!(VᴴΔV₁)
        Δgauge = max(Δgauge, ΔgaugeV)
    else
        ΔV₁ᴴ = nothing
        aVᴴΔV₁ = zero!(similar(V₁ᴴ, (r, r)))
    end
    mask = abs.(S₁' .- S₁) .< degeneracy_atol
    Δgauge = max(Δgauge, norm(view(aUᴴΔU₁, mask) + view(aVᴴΔV₁, mask), Inf))

    if !iszerotangent(ΔSmat)
        ΔS = diagview(ΔSmat)
        length(indS) == length(ΔS) || throw(DimensionMismatch("length of selected S values ($(length(indS))) does not match length of ΔS ($(length(ΔS)))"))
        ΔS₁ = zero(S₁)
        for (j, i) in enumerate(indS)
            if i <= r
                ΔS₁[i] = real(ΔS[j])
            else
                Δgauge = max(Δgauge, abs(ΔS[j]))
            end
        end
    else
        ΔS₁ = nothing
    end

    Δgauge ≤ gauge_atol ||
        @warn "`svd` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
    return ΔU₁, ΔS₁, ΔV₁ᴴ, aUᴴΔU₁, aVᴴΔV₁
end

"""
    svd_pullback!(
        ΔA, A, USVᴴ, ΔUSVᴴ, [ind];
        rank_atol::Real = default_pullback_rank_atol(USVᴴ[2]),
        degeneracy_atol::Real = default_pullback_rank_atol(USVᴴ[2]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔUSVᴴ...)
    )

Adds the pullback from the SVD of `A` to `ΔA` given the output `USVᴴ` of `svd_compact` or
`svd_full` and the cotangent `ΔUSVᴴ` of `svd_compact`, `svd_full` or `svd_trunc`.

In particular, it is assumed that `A ≈ U * S * Vᴴ`, or thus, that no singular values with
magnitude less than `rank_atol` are missing from `S`.  For the cotangents, an arbitrary
number of singular vectors or singular values can be missing, i.e. for a matrix `A` with
size `(m, n)`, `ΔU` and `ΔVᴴ` can have sizes `(m, p)` and `(p, n)` respectively, whereas
`diagview(ΔS)` can have length `p`. In those cases, an additional list `ind` of length `p`
is required to specify which singular vectors and values are present in `ΔU`, `ΔS` and `ΔVᴴ`.

A warning will be printed if the cotangents are not gauge-invariant, i.e. if the
anti-hermitian part of `U' * ΔU + Vᴴ * ΔVᴴ'`, restricted to rows `i` and columns `j` for
which `abs(S[i] - S[j]) < degeneracy_atol`, is not small compared to `gauge_atol`.
"""
function svd_pullback!(
        ΔA::AbstractMatrix, A, USVᴴ, ΔUSVᴴ, ind = Colon();
        rank_atol::Real = default_pullback_rank_atol(USVᴴ[2]),
        degeneracy_atol::Real = default_pullback_rank_atol(USVᴴ[2]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔUSVᴴ...)
    )
    # Extract the SVD components
    U, Smat, Vᴴ = USVᴴ
    m, n = size(U, 1), size(Vᴴ, 2)
    minmn = min(m, n)
    (m, n) == size(ΔA) || throw(DimensionMismatch("size of ΔA ($(size(ΔA))) does not match size of U*S*Vᴴ ($m, $n)"))
    S = diagview(Smat)
    r = svd_rank(S; rank_atol)

    U₁ = view(U, :, 1:r)
    V₁ᴴ = view(Vᴴ, 1:r, :)
    S₁ = view(S, 1:r)

    ΔU, ΔSmat, ΔVᴴ = ΔUSVᴴ
    ΔU₁, ΔS₁, ΔV₁ᴴ, aUᴴΔU₁, aVᴴΔV₁ = check_and_prepare_svd_cotangents(
        U, S, Vᴴ, ΔU, ΔSmat, ΔVᴴ, r, ind; degeneracy_atol, gauge_atol
    )

    UdΔAV = (aUᴴΔU₁ .+ aVᴴΔV₁) .* inv_safe.(S₁' .- S₁, degeneracy_atol) .+
        (aUᴴΔU₁ .- aVᴴΔV₁) .* inv_safe.(S₁' .+ S₁, degeneracy_atol)
    if !iszerotangent(ΔS₁)
        diagview(UdΔAV) .+= real.(ΔS₁)
    end
    ΔA = mul!(ΔA, U₁, UdΔAV * V₁ᴴ, 1, 1) # add the contribution to ΔA

    # Add the remaining contributions
    if m > r && !iszerotangent(ΔU₁) # ΔU₁ is already orthogonal to U₁
        ΔA = mul!(ΔA, ΔU₁ ./ S₁', V₁ᴴ, 1, 1)
    end
    if n > r && !iszerotangent(ΔV₁ᴴ) # ΔV₁ᴴ is already orthogonal to V₁ᴴ
        ΔA = mul!(ΔA, U₁, S₁ .\ ΔV₁ᴴ, 1, 1)
    end
    return ΔA
end
function svd_pullback!(
        ΔA::Diagonal, A, USVᴴ, ΔUSVᴴ, ind = Colon();
        rank_atol::Real = default_pullback_rank_atol(USVᴴ[2]),
        degeneracy_atol::Real = default_pullback_rank_atol(USVᴴ[2]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔUSVᴴ...)
    )
    ΔA_full = zero!(similar(ΔA, size(ΔA)))
    ΔA_full = svd_pullback!(ΔA_full, A, USVᴴ, ΔUSVᴴ, ind; rank_atol, degeneracy_atol, gauge_atol)
    diagview(ΔA) .+= diagview(ΔA_full)
    return ΔA
end

"""
    svd_trunc_pullback!(
        ΔA, A, USVᴴ, ΔUSVᴴ;
        rank_atol::Real = default_pullback_rank_atol(USVᴴ[2]),
        degeneracy_atol::Real = default_pullback_rank_atol(USVᴴ[2]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔUSVᴴ...)
    )

Adds the pullback from the truncated SVD of `A` to `ΔA`, given the output `USVᴴ` and the
cotangent `ΔUSVᴴ` of `svd_trunc`.

In particular, it is assumed that `A * Vᴴ' ≈ U * S` and `U' * A = S * Vᴴ`, with `U` and `Vᴴ`
rectangular matrices of left and right singular vectors, and `S` diagonal. For the
cotangents, it is assumed that if `ΔU` and `ΔVᴴ` are not zero, then they have the same size
as `U` and `Vᴴ` (respectively), and if `ΔS` is not zero, then it is a diagonal matrix of the
same size as `S`. For this method to work correctly, it is also assumed that the remaining
singular values (not included in `S`) are (sufficiently) smaller than those in `S`.

A warning will be printed if the cotangents are not gauge-invariant, i.e. if the
anti-hermitian part of `U' * ΔU + Vᴴ * ΔVᴴ'`, restricted to rows `i` and columns `j` for
which `abs(S[i] - S[j]) < degeneracy_atol`, is not small compared to `gauge_atol`.
"""
function svd_trunc_pullback!(
        ΔA::AbstractMatrix, A, USVᴴ, ΔUSVᴴ;
        rank_atol::Real = 0,
        degeneracy_atol::Real = default_pullback_rank_atol(USVᴴ[2]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔUSVᴴ...),
        maxiter::Int = 1000,
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
    ΔU, ΔS, ΔVᴴ, aUᴴΔU, aVᴴΔV = check_and_prepare_svd_cotangents(
        U, S, Vᴴ, ΔU, ΔSmat, ΔVᴴ, p; degeneracy_atol, gauge_atol
    )

    # This part is the same as in `svd_pullback!`
    UdΔAV = (aUᴴΔU .+ aVᴴΔV) .* inv_safe.(S' .- S, degeneracy_atol) .+
        (aUᴴΔU .- aVᴴΔV) .* inv_safe.(S' .+ S, degeneracy_atol)
    if !iszerotangent(ΔS)
        diagview(UdΔAV) .+= real.(ΔS)
    end
    ΔA = mul!(ΔA, U, UdΔAV * Vᴴ, 1, 1) # add the contribution to ΔA

    # The contribtutions from the orthogonal complement need to be treated differently
    # ΔU and ΔVᴴ are already orthogonal to U and Vᴴ
    if !(iszerotangent(ΔU) && iszerotangent(ΔVᴴ))
        US = U * Smat
        APAᴴ = mul!(A * A', US, US', -1, 1)
        SVᴴ = Smat * Vᴴ
        AᴴPA = mul!(A' * A, SVᴴ', SVᴴ, -1, 1)

        rhs = [iszerotangent(ΔU) ? zero(U) : ΔU; iszerotangent(ΔVᴴ) ? zero(Vᴴ') : ΔVᴴ']
        AA = [zero(APAᴴ) (A - U * (U' * A)); (A' - Vᴴ' * (Vᴴ * A')) zero(AᴴPA)]
        XY = _sylvester(AA, -Smat, rhs)

        Aperp = A - U * Smat * Vᴴ
        x₀ = iszerotangent(ΔU) ? zero(U) : rdiv!(ΔU, Diagonal(S))
        y₀ᴴ = iszerotangent(ΔVᴴ) ? zero(Vᴴ) : ldiv!(Diagonal(S), ΔVᴴ)
        X = copy(x₀)
        Yᴴ = copy(y₀ᴴ)
        xₖ, xₖ₊₁ = x₀, zero(x₀)
        yₖᴴ, yₖ₊₁ᴴ = y₀ᴴ, zero(y₀ᴴ)
        for k in 1:maxiter
            xₖ₊₁ = rdiv!(mul!(xₖ₊₁, Aperp, yₖᴴ'), Diagonal(S))
            yₖ₊₁ᴴ = ldiv!(Diagonal(S), mul!(yₖ₊₁ᴴ, xₖ', Aperp))
            X .+= xₖ₊₁
            Yᴴ .+= yₖ₊₁ᴴ
            if norm(xₖ₊₁, Inf) < degeneracy_atol && norm(yₖ₊₁ᴴ, Inf) < degeneracy_atol
                break
            end
            xₖ, xₖ₊₁ = xₖ₊₁, xₖ
            yₖᴴ, yₖ₊₁ᴴ = yₖ₊₁ᴴ, yₖᴴ
            if k == maxiter
                @warn "Sylvester iteration did not converge after $k iterations, final norms: (x: $(norm(xₖ₊₁, Inf)), y: $(norm(yₖ₊₁ᴴ, Inf)))"
            end
        end
        ΔA = mul!(ΔA, X, Vᴴ, 1, 1)
        ΔA = mul!(ΔA, U, Yᴴ, 1, 1)
    end
    return ΔA
end
function svd_trunc_pullback!(
        ΔA::Diagonal, A, USVᴴ, ΔUSVᴴ;
        rank_atol::Real = 0,
        degeneracy_atol::Real = default_pullback_rank_atol(USVᴴ[2]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔUSVᴴ[1], ΔUSVᴴ[3])
    )
    ΔA_full = zero!(similar(ΔA, size(ΔA)))
    ΔA_full = svd_trunc_pullback!(ΔA_full, A, USVᴴ, ΔUSVᴴ; rank_atol, degeneracy_atol, gauge_atol)
    diagview(ΔA) .+= diagview(ΔA_full)
    return ΔA
end

"""
    svd_vals_pullback!(
        ΔA, A, USVᴴ, ΔS, [ind];
        rank_atol::Real = default_pullback_rank_atol(USVᴴ[2]),
        degeneracy_atol::Real = default_pullback_rank_atol(USVᴴ[2])
    )


Adds the pullback from the singular values of `A` to `ΔA`, given the output
`USVᴴ` of `svd_compact`, and the cotangent `ΔS` of `svd_vals`.

In particular, it is assumed that `A ≈ U * S * Vᴴ`, or thus, that no singular values with
magnitude less than `rank_atol` are missing from `S`. For the cotangents, an arbitrary
number of singular vectors or singular values can be missing, i.e. for a matrix `A` with
size `(m, n)`, `diagview(ΔS)` can have length `pS`. In those cases, additionally `ind` is required to
specify which singular vectors and values are present in `ΔS`.
"""
function svd_vals_pullback!(
        ΔA, A, USVᴴ, ΔS, ind = Colon();
        rank_atol::Real = default_pullback_rank_atol(USVᴴ[2]),
        degeneracy_atol::Real = default_pullback_rank_atol(USVᴴ[2])
    )
    ΔUSVᴴ = (nothing, diagonal(ΔS), nothing)
    return svd_pullback!(ΔA, A, USVᴴ, ΔUSVᴴ, ind; rank_atol, degeneracy_atol)
end

"""
    remove_svd_gauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ; degeneracy_atol = ..., rank_atol = ...)

Remove the gauge-dependent part from the cotangents `ΔU` and `ΔVᴴ` of the SVD factors. The
singular vectors are only determined up to a common complex phase per singular value (or a
unitary transformation across singular vectors associated with degenerate singular values),
so the corresponding anti-Hermitian components of `U₁' * ΔU₁ + Vᴴ₁ * ΔVᴴ₁'` are projected out.
For the full SVD, the extra columns of `U` and rows of `Vᴴ` beyond the rank `r` are
additionally zeroed out, where `r = count(diagview(S) .> rank_atol)`.
"""
function remove_svd_gauge_dependence!(
        ΔU, ΔVᴴ, U, S, Vᴴ;
        degeneracy_atol = MatrixAlgebraKit.default_pullback_gauge_atol(S),
        rank_atol = MatrixAlgebraKit.default_pullback_rank_atol(S)
    )
    Sdiag = diagview(S)
    r = MatrixAlgebraKit.svd_rank(Sdiag; rank_atol)
    U₁ = view(U, :, 1:r)
    Vᴴ₁ = view(Vᴴ, 1:r, :)
    ΔU₁ = view(ΔU, :, 1:r)
    ΔVᴴ₁ = view(ΔVᴴ, 1:r, :)
    Sdiag = diagview(S)
    gaugepart = mul!(U₁' * ΔU₁, Vᴴ₁, ΔVᴴ₁', true, true)
    gaugepart = project_antihermitian!(gaugepart)
    gaugepart[abs.(transpose(Sdiag) .- Sdiag) .>= degeneracy_atol] .= 0
    mul!(ΔU₁, U₁, gaugepart, -1, 1)
    if size(ΔU, 2) > r
        if r < length(Sdiag) # rank-deficient case, no stable information can be extracted from extra columns of U
            ΔU[:, (r + 1):end] .= 0
        else # the component of ΔU₂ along U₁ contains gauge-invariant information
            p = size(ΔU, 2)
            ΔU₂ = view(ΔU, :, (r + 1):p)
            U₁ᴴΔU₂ = U₁' * ΔU₂
            mul!(ΔU₂, U₁, U₁ᴴΔU₂)
        end
    end
    if size(ΔVᴴ, 1) > r
        if r < length(Sdiag) # rank-deficient case, no stable information can be extracted from extra rows of Vᴴ
            ΔVᴴ[(r + 1):end, :] .= 0
        else # the component of ΔVᴴ₂ along Vᴴ₁ contains gauge-invariant information
            p = size(ΔVᴴ, 1)
            ΔVᴴ₂ = view(ΔVᴴ, (r + 1):p, :)
            ΔVᴴ₂V₁ = ΔVᴴ₂ * Vᴴ₁'
            mul!(ΔVᴴ₂, ΔVᴴ₂V₁, Vᴴ₁)
        end
    end
    return ΔU, ΔVᴴ
end
