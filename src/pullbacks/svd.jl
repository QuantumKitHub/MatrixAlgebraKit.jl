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

    # Only the columns ind′ ⊆ 1:r of UᴴΔAV are computed, its rows ind′ follow by antihermiticity. These
    # are the columns of the cotangents within the rank, or all of 1:r in the full rank case
    # if there are cotangents beyond it, since those have components along all of U₁ or V₁ᴴ.
    j₁ = all(<=(r), indS) ? eachindex(indS) : findall(<=(r), indS)
    fold = r == minmn && max(length(indU), length(indV)) > length(j₁)
    ind′ = fold ? (1:r) : indS[j₁]
    l₁ = fold ? indS[j₁] : eachindex(j₁) # columns of the cotangents j₁ among ind′
    k = length(ind′)

    if !iszerotangent(ΔU)
        ΔgaugeU = zero(eltype(S))
        m == size(ΔU, 1) || throw(DimensionMismatch(lazy"first dimension of ΔU ($(size(ΔU, 1))) does not match first dimension of U ($m)"))
        length(indU) == size(ΔU, 2) || throw(DimensionMismatch(lazy"length of selected U columns ($(length(indU))) does not match second dimension of ΔU ($(size(ΔU, 2)))"))
        if indU == ind′
            ΔU₁ = copy(ΔU)
        else
            ΔU₁ = zero!(similar(U, (m, k)))
            ΔU₁[:, l₁] .= view(ΔU, :, j₁)
            wtmp = similar(U₁, (r,))
            utmp = similar(U₁, (m,))
            zeroj = Int[]
            for (j, i) in enumerate(indU)
                if i <= r
                    continue
                elseif fold # full rank case, ΔU₃ contains gauge-invariant information along U₁
                    mul!(wtmp, U₁', view(ΔU, :, j))
                    mul!(ΔU₁, view(U, :, i), wtmp', -1, 1)
                    utmp .= view(ΔU, :, j)
                    mul!(utmp, U₁, wtmp, -1, 1)
                    ΔgaugeU = max(ΔgaugeU, norm(utmp))
                else # remaining columns should be zero
                    push!(zeroj, j)
                end
            end
            # index with a vector rather than looping over views, so wrapped GPU arrays
            # (e.g. `Adjoint{<:CuArray}`) don't fall back to scalar iteration
            ΔgaugeU = max(ΔgaugeU, maximum(abs, ΔU[:, zeroj]; init = abs(zero(eltype(ΔU)))))
        end
        UᴴΔU₁ = U₁' * ΔU₁
        ΔU₊ = mul!(ΔU₁, U₁, UᴴΔU₁, -1, 1)
        aUᴴΔU₁ = antihermitian_columns!(UᴴΔU₁, ind′)
        Δgauge = max(Δgauge, ΔgaugeU)
    else
        ΔU₊ = nothing
        aUᴴΔU₁ = zero!(similar(U₁, (r, k)))
    end
    if !iszerotangent(ΔVᴴ)
        ΔgaugeV = zero(eltype(S))
        n == size(ΔVᴴ, 2) || throw(DimensionMismatch(lazy"second dimension of ΔVᴴ ($(size(ΔVᴴ, 2))) does not match second dimension of Vᴴ ($n)"))
        length(indV) == size(ΔVᴴ, 1) || throw(DimensionMismatch(lazy"length of selected Vᴴ rows ($(length(indV))) does not match first dimension of ΔVᴴ ($(size(ΔVᴴ, 1)))"))
        if indV == ind′
            ΔV₁ᴴ = copy(ΔVᴴ)
        else
            ΔV₁ᴴ = zero!(similar(Vᴴ, (k, n)))
            ΔV₁ᴴ[l₁, :] .= view(ΔVᴴ, j₁, :)
            wtmp = similar(V₁ᴴ, (1, r))
            vtmp = similar(V₁ᴴ, (1, n))
            zeroj = Int[]
            for (j, i) in enumerate(indV)
                if i <= r
                    continue
                elseif fold # full rank case, ΔV₃ contains gauge-invariant information along Vᴴ₁
                    mul!(wtmp, view(ΔVᴴ, j:j, :), V₁ᴴ')
                    mul!(ΔV₁ᴴ, wtmp', view(Vᴴ, i:i, :), -1, 1)
                    vtmp .= view(ΔVᴴ, j:j, :)
                    mul!(vtmp, wtmp, V₁ᴴ, -1, 1)
                    ΔgaugeV = max(ΔgaugeV, norm(vtmp))
                else # remaining rows should be zero
                    push!(zeroj, j)
                end
            end
            ΔgaugeV = max(ΔgaugeV, maximum(abs, ΔVᴴ[zeroj, :]; init = abs(zero(eltype(ΔVᴴ)))))
        end
        VᴴΔV₁ = V₁ᴴ * ΔV₁ᴴ'
        ΔV₊ᴴ = mul!(ΔV₁ᴴ, VᴴΔV₁', V₁ᴴ, -1, 1)
        aVᴴΔV₁ = antihermitian_columns!(VᴴΔV₁, ind′)
        Δgauge = max(Δgauge, ΔgaugeV)
    else
        ΔV₊ᴴ = nothing
        aVᴴΔV₁ = zero!(similar(V₁ᴴ, (r, k)))
    end

    Sₖ = view(S, ind′)
    gaugepart = (abs.(transpose(Sₖ) .- S₁) .< degeneracy_atol) .* (aUᴴΔU₁ .+ aVᴴΔV₁)
    Δgauge = max(Δgauge, maximum(abs, gaugepart; init = abs(zero(eltype(S)))))

    if !iszerotangent(ΔSmat)
        ΔS = diagview(ΔSmat)
        length(indS) == length(ΔS) || throw(DimensionMismatch(lazy"length of selected S values ($(length(indS))) does not match length of ΔS ($(length(ΔS)))"))
        badΔS = view(ΔS, findall(>(r), indS))
        Δgauge = max(Δgauge, maximum(abs, badΔS; init = abs(zero(eltype(ΔS)))))
    end

    Δgauge ≤ gauge_atol ||
        @warn "`svd` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"

    # columns ind′ of UᴴΔAV, and the adjoint of its rows ind′ (not needed if ind′ contains all rows)
    # NOTE: for all columns (k == r) only UᴴΔAV is computed, as in the original full-matrix path
    UᴴΔAV = (aUᴴΔU₁ .+ aVᴴΔV₁) .* inv_safe.(transpose(Sₖ) .- S₁, degeneracy_atol) .+
        (aUᴴΔU₁ .- aVᴴΔV₁) .* inv_safe.(transpose(Sₖ) .+ S₁, degeneracy_atol)
    UᴴΔAVʳ = k == r ? nothing :
        (aUᴴΔU₁ .+ aVᴴΔV₁) .* inv_safe.(transpose(Sₖ) .- S₁, degeneracy_atol) .-
        (aUᴴΔU₁ .- aVᴴΔV₁) .* inv_safe.(transpose(Sₖ) .+ S₁, degeneracy_atol)
    if !iszerotangent(ΔSmat)
        UᴴΔAV[indS[j₁] .+ r .* (l₁ .- 1)] .+= real.(view(ΔS, j₁)) # the entries (ind′[l₁], l₁)
    end

    return UᴴΔAV, ΔU₊, ΔV₊ᴴ, UᴴΔAVʳ, ind′
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
size `(m, n)`, `ΔU`, `ΔS` and `ΔVᴴ` can have sizes `(m, p)`, `(p, p)` and `(p, n)` respectively
and the argument `ind` is a list of length `p` indicating that these are cotangents corresponding to `U[:, ind]`, `S[ind, ind]` and `Vᴴ[ind, :]`,
whereas cotangents with respect to the other rows and columns are zero.
If `ind` is not present, `ΔU`, `ΔS` and `ΔVᴴ` are assumed to have the same size as `U`, `S` and `Vᴴ` respectively.

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
    (m, n) == size(ΔA) || throw(DimensionMismatch(lazy"size of ΔA ($(size(ΔA))) does not match size of USVᴴ ($m, $n)"))
    S = diagview(Smat)
    r = svd_rank(S; rank_atol)
    iszero(r) && return ΔA

    U₁ = view(U, :, 1:r)
    V₁ᴴ = view(Vᴴ, 1:r, :)
    S₁ = view(S, 1:r)

    ΔU, ΔSmat, ΔVᴴ = ΔUSVᴴ
    UᴴΔAVₖ, ΔU₊, ΔV₊ᴴ, UᴴΔAVʳ, ind′ = check_and_prepare_svd_cotangents(
        U, S, Vᴴ, ΔU, ΔSmat, ΔVᴴ, r, ind; degeneracy_atol, gauge_atol
    )

    # UᴴΔAV is nonzero only in its rows and columns ind′, which are UᴴΔAVₖ and UᴴΔAVʳ'. For k ≤ r / 2,
    # applying these two blocks directly, in O(m n k), is faster than forming UᴴΔAV.
    Sₖ = view(S, ind′)
    Uₖ = U[:, ind′]
    Vᴴₖ = Vᴴ[ind′, :]
    if 2 * length(ind′) <= r
        UᴴΔAVʳ[ind′, :] .= zero(eltype(UᴴΔAVʳ)) # these entries are part of the columns ind′
        ΔA = mul!(ΔA, U₁ * UᴴΔAVₖ, Vᴴₖ, 1, 1)
        ΔA = mul!(ΔA, Uₖ, UᴴΔAVʳ' * V₁ᴴ, 1, 1)
    else
        if is_leading_index(ind′, r) # NOTE: all columns in order (e.g. `ind = Colon()`): the original path
            UᴴΔAV = UᴴΔAVₖ
        else
            UᴴΔAV = zero!(similar(UᴴΔAVₖ, (r, r)))
            isnothing(UᴴΔAVʳ) || (UᴴΔAV[ind′, :] .= UᴴΔAVʳ')
            UᴴΔAV[:, ind′] .= UᴴΔAVₖ # after the rows: only the columns hold ΔS
        end
        ΔA = mul!(ΔA, U₁, UᴴΔAV * V₁ᴴ, 1, 1) # add the contribution to ΔA
    end

    # Add the remaining contributions
    if m > r && !iszerotangent(ΔU₊) # ΔU₁ is already orthogonal to U₁
        ΔU₊ ./= transpose(Sₖ)
        ΔA = mul!(ΔA, ΔU₊, Vᴴₖ, 1, 1)
    end
    if n > r && !iszerotangent(ΔV₊ᴴ) # ΔV₁ᴴ is already orthogonal to V₁ᴴ
        ΔV₊ᴴ .= Sₖ .\ ΔV₊ᴴ
        ΔA = mul!(ΔA, Uₖ, ΔV₊ᴴ, 1, 1)
    end
    return ΔA
end
# Diagonal: do not specialize on `A`, since we may insert `A = nothing` to assert independence of `A` in the implementation
function svd_pullback!(
        ΔA::Diagonal, A, USVᴴ, ΔUSVᴴ, ind = Colon();
        rank_atol::Real = default_pullback_rank_atol(USVᴴ[2]),
        degeneracy_atol::Real = default_pullback_rank_atol(USVᴴ[2]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔUSVᴴ...)
    )
    # TODO:
    # If A and ΔA are diagonal, then U and V are permutation matrices (up to signs/phases).
    # Furthermore, since U̇ and V̇ are 0, the pullbacks ΔU and ΔV cannot contribute and we only have to unpermute ΔS.
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
        maxiter::Int = 100 # TODO: better default, depending on expected number of steps using quadratic convergence?
    )
    # Extract the SVD components
    U, Smat, Vᴴ = USVᴴ
    m, n = size(U, 1), size(Vᴴ, 2)
    (m, n) == size(ΔA) || throw(DimensionMismatch(lazy"size of ΔA ($(size(ΔA))) does not match size of USVᴴ ($m, $n)"))
    S = diagview(Smat)
    p = length(S)
    p == size(U, 2) || throw(DimensionMismatch(lazy"U has $p columns but S has $(length(S)) singular values"))
    p == size(Vᴴ, 1) || throw(DimensionMismatch(lazy"Vᴴ has $p rows but  S has $(length(S)) singular values"))
    iszero(p) && return ΔA

    # Extract and check the cotangents
    ΔU, ΔSmat, ΔVᴴ = ΔUSVᴴ
    UᴴΔAV, ΔU₊, ΔV₊ᴴ = check_and_prepare_svd_cotangents(
        U, S, Vᴴ, ΔU, ΔSmat, ΔVᴴ, p; degeneracy_atol, gauge_atol
    )
    ΔAV = U * UᴴΔAV
    ΔA = mul!(ΔA, ΔAV, Vᴴ, 1, 1) # add the contribution to ΔA

    # The contribtutions from the orthogonal complement need to be treated differently
    # ΔU and ΔVᴴ are already orthogonal to U and Vᴴ
    if !(iszerotangent(ΔU₊) && iszerotangent(ΔV₊ᴴ))
        X₀ = iszerotangent(ΔU₊) ? zero(U) : rdiv!(ΔU₊, Diagonal(S))
        Y₀ᴴ = iszerotangent(ΔV₊ᴴ) ? zero(Vᴴ) : ldiv!(Diagonal(S), ΔV₊ᴴ)
        US = mul!(ΔAV, U, Smat) # recycle ΔAV
        AP = mul!(copy(A), US, Vᴴ, -1, 1)
        S⁻¹ = inv.(S)
        # sum the series on the smaller side only, the other side follows from
        # Yᴴ = Y₀ᴴ + S⁻¹ X' AP (m ≤ n) or X = X₀ + AP Y S⁻¹ (m > n)
        if m ≤ n
            X = rmul!(AP * Y₀ᴴ', Diagonal(S⁻¹))
            X .+= X₀
            X = accelerative_smith_iteration!(X, X₀, AP * AP', S⁻¹ .^ 2, degeneracy_atol, maxiter) # recycle X₀
            Yᴴ = lmul!(Diagonal(S⁻¹), X' * AP)
            Yᴴ .+= Y₀ᴴ
            ΔA = mul!(ΔA, X, Vᴴ, 1, 1)
            ΔA = mul!(ΔA, U, Yᴴ, 1, 1)
        else
            Y = rmul!(AP' * X₀, Diagonal(S⁻¹))
            Y .+= Y₀ᴴ'
            Y = accelerative_smith_iteration!(Y, similar(Y), AP' * AP, S⁻¹ .^ 2, degeneracy_atol, maxiter)
            X = rmul!(AP * Y, Diagonal(S⁻¹))
            X .+= X₀
            ΔA = mul!(ΔA, X, Vᴴ, 1, 1)
            ΔA = mul!(ΔA, U, Y', 1, 1)
        end
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
    gaugepart[abs.(transpose(view(Sdiag, 1:r)) .- view(Sdiag, 1:r)) .>= degeneracy_atol] .= 0
    mul!(ΔU₁, U₁, gaugepart, -1, 1)
    if size(ΔU, 2) > r
        if r < length(Sdiag) # rank-deficient case, no stable information can be extracted from extra columns of U
            zero!(view(ΔU, :, (r + 1):size(ΔU, 2)))
        else # the component of ΔU₂ along U₁ contains gauge-invariant information
            p = size(ΔU, 2)
            ΔU₂ = view(ΔU, :, (r + 1):p)
            U₁ᴴΔU₂ = U₁' * ΔU₂
            mul!(ΔU₂, U₁, U₁ᴴΔU₂)
        end
    end
    if size(ΔVᴴ, 1) > r
        if r < length(Sdiag) # rank-deficient case, no stable information can be extracted from extra rows of Vᴴ
            zero!(view(ΔVᴴ, (r + 1):size(ΔVᴴ, 1), :))
        else # the component of ΔVᴴ₂ along Vᴴ₁ contains gauge-invariant information
            p = size(ΔVᴴ, 1)
            ΔVᴴ₂ = view(ΔVᴴ, (r + 1):p, :)
            ΔVᴴ₂V₁ = ΔVᴴ₂ * Vᴴ₁'
            mul!(ΔVᴴ₂, ΔVᴴ₂V₁, Vᴴ₁)
        end
    end
    return ΔU, ΔVᴴ
end
