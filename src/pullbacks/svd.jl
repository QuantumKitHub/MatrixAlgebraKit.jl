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

    indU = select_indices(axes(U, 2), ind)
    indV = select_indices(axes(Vᴴ, 1), ind)
    indS = select_indices(axes(S, 1), ind)
    Δgauge = zero(eltype(S))

    # Only the columns ind₀ ⊆ 1:r of UᴴΔAV are computed.
    # By keeping its hermitian and antihermitian parts separate, we can reconstruct the full UᴴΔAV
    J₁ = findall(<=(r), indS)
    J₂ = findall(>(r), indS)
    ind₀ = indS[J₁]
    full_rank = (ind₀ == 1:minmn)

    if !iszerotangent(ΔU)
        ΔgaugeU = zero(eltype(S))
        m == size(ΔU, 1) || throw(DimensionMismatch(lazy"first dimension of ΔU ($(size(ΔU, 1))) does not match first dimension of U ($m)"))
        length(indU) == size(ΔU, 2) || throw(DimensionMismatch(lazy"length of selected U columns ($(length(indU))) does not match second dimension of ΔU ($(size(ΔU, 2)))"))
        if indU == indS
            ΔU₀ = ΔU[:, J₁]
            ΔgaugeU = max(ΔgaugeU, maximum(abs, view(ΔU, :, J₂); init = zero(ΔgaugeU)))
        elseif full_rank && indU == 1:m
            ΔU₀ = ΔU[:, ind₀]
            J₃ = (r + 1):m
            U₃ = view(U, :, J₃)
            ΔU₃ = ΔU[:, J₃]
            U₁ᴴΔU₃ = U₁' * ΔU₃ # gauge-invariant part
            mul!(ΔU₀, U₃, U₁ᴴΔU₃', -1, 1)
            mul!(ΔU₃, U₁, U₁ᴴΔU₃, -1, 1)
            ΔgaugeU = max(ΔgaugeU, maximum(abs, ΔU₃; init = zero(ΔgaugeU)))
        else
            throw(ArgumentError(lazy"Unexpected selection of U columns: indU = $indU, expected indS = $indS or 1:$m"))
        end
        UᴴΔU₁₀ = U₁' * ΔU₀
        ΔU₊ = mul!(ΔU₀, U₁, UᴴΔU₁₀, -1, 1)
        aUᴴΔU₁₀ = antihermitian_columns!(UᴴΔU₁₀, ind₀)
        Δgauge = max(Δgauge, ΔgaugeU)
    else
        ΔU₊ = nothing
        aUᴴΔU₁₀ = zero!(similar(U₁, (r, length(ind₀))))
    end
    if !iszerotangent(ΔVᴴ)
        ΔgaugeV = zero(eltype(S))
        n == size(ΔVᴴ, 2) || throw(DimensionMismatch(lazy"second dimension of ΔVᴴ ($(size(ΔVᴴ, 2))) does not match second dimension of Vᴴ ($n)"))
        length(indV) == size(ΔVᴴ, 1) || throw(DimensionMismatch(lazy"length of selected Vᴴ rows ($(length(indV))) does not match first dimension of ΔVᴴ ($(size(ΔVᴴ, 1)))"))
        if indV == indS
            ΔV₀ᴴ = ΔVᴴ[J₁, :]
            ΔgaugeV = max(ΔgaugeV, maximum(abs, view(ΔVᴴ, J₂, :); init = zero(ΔgaugeV)))
        elseif full_rank && indV == 1:n
            ΔV₀ᴴ = ΔVᴴ[ind₀, :]
            J₃ = (r + 1):n
            V₃ᴴ = view(Vᴴ, J₃, :)
            ΔV₃ᴴ = ΔVᴴ[J₃, :]
            V₁ᴴΔV₃ = V₁ᴴ * (ΔV₃ᴴ)' # gauge-invariant part
            mul!(ΔV₀ᴴ, V₁ᴴΔV₃, V₃ᴴ, -1, 1)
            mul!(ΔV₃ᴴ, V₁ᴴΔV₃', V₁ᴴ, -1, 1)
            ΔgaugeV = max(ΔgaugeV, maximum(abs, ΔV₃ᴴ; init = zero(ΔgaugeV)))
        else
            throw(ArgumentError(lazy"Unexpected selection of Vᴴ rows: indV = $indV, expected indS = $indS or 1:$n"))
        end
        VᴴΔV₁₀ = V₁ᴴ * ΔV₀ᴴ'
        ΔV₊ᴴ = mul!(ΔV₀ᴴ, VᴴΔV₁₀', V₁ᴴ, -1, 1)
        aVᴴΔV₁₀ = antihermitian_columns!(VᴴΔV₁₀, ind₀)
        Δgauge = max(Δgauge, ΔgaugeV)
    else
        ΔV₊ᴴ = nothing
        aVᴴΔV₁₀ = zero!(similar(V₁ᴴ, (r, length(ind₀))))
    end

    S₀ = S[ind₀] # view fails broadcasting below on GPU
    hUᴴΔAV₁₀ = (aUᴴΔU₁₀ .+ aVᴴΔV₁₀) .* inv_safe.(transpose(S₀) .- S₁, degeneracy_atol) # hermitian part of UᴴΔAV, restricted to rows 1:r and columns ind₀
    aUᴴΔAV₁₀ = (aUᴴΔU₁₀ .- aVᴴΔV₁₀) .* inv_safe.(transpose(S₀) .+ S₁, degeneracy_atol) # antihermitian part of UᴴΔAV, restricted to rows 1:r and columns ind₀

    gaugepart = (abs.(transpose(S₀) .- S₁) .< degeneracy_atol) .* (aUᴴΔU₁₀ .+ aVᴴΔV₁₀)
    Δgauge = max(Δgauge, maximum(abs, gaugepart; init = zero(Δgauge)))

    if !iszerotangent(ΔSmat)
        ΔS = diagview(ΔSmat)
        length(indS) == length(ΔS) || throw(DimensionMismatch(lazy"length of selected S values ($(length(indS))) does not match length of ΔS ($(length(ΔS)))"))
        hUᴴΔAV₁₀[ind₀ .+ r .* (0:(length(ind₀) - 1))] .+= real.(ΔS[J₁]) # diagonal entries
        Δgauge = max(Δgauge, maximum(abs, ΔS[J₂]; init = zero(Δgauge)))
    end

    Δgauge ≤ gauge_atol ||
        @warn "`svd` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"

    return hUᴴΔAV₁₀, aUᴴΔAV₁₀, ΔU₊, ΔV₊ᴴ, ind₀
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
    (m, n) == size(ΔA) || throw(DimensionMismatch(lazy"size of ΔA ($(size(ΔA))) does not match size of USVᴴ ($m, $n)"))
    S = diagview(Smat)
    r = svd_rank(S; rank_atol)
    iszero(r) && return ΔA

    # Extract and check the cotangents
    ΔU, ΔSmat, ΔVᴴ = ΔUSVᴴ
    hUᴴΔAV₁₀, aUᴴΔAV₁₀, ΔU₊, ΔV₊ᴴ, ind₀ = check_and_prepare_svd_cotangents(
        U, S, Vᴴ, ΔU, ΔSmat, ΔVᴴ, r, ind; degeneracy_atol, gauge_atol
    )

    U₀ = U[:, ind₀] # ind₀ is not necessarily a range
    U₁ = view(U, :, 1:r)
    V₀ᴴ = Vᴴ[ind₀, :]
    V₁ᴴ = view(Vᴴ, 1:r, :)
    S₀ = S[ind₀]

    # UᴴΔAV is nonzero only in its columns ind₀, which are hUᴴΔAV₁₀ + aUᴴΔAV₁₀, and its rows ind₀,
    # which are hUᴴΔAV₁₀' - aUᴴΔAV₁₀'. For k ≤ r / 2, applying these two blocks directly, in O(m n k),
    # is faster than forming UᴴΔAV.
    if 2 * length(ind₀) <= r
        ΔA = mul!(ΔA, U₁ * (hUᴴΔAV₁₀ + aUᴴΔAV₁₀), V₀ᴴ, 1, 1)
        hUᴴΔAV₁₀[ind₀, :] .= zero(eltype(hUᴴΔAV₁₀))
        aUᴴΔAV₁₀[ind₀, :] .= zero(eltype(aUᴴΔAV₁₀))
        ΔA = mul!(ΔA, U₀, (hUᴴΔAV₁₀' - aUᴴΔAV₁₀') * V₁ᴴ, 1, 1)
    else
        if is_leading_index(ind₀, r) # NOTE: all columns in order (e.g. `ind = Colon()`): the original path
            UᴴΔAV = hUᴴΔAV₁₀ + aUᴴΔAV₁₀
        else
            UᴴΔAV = zero!(similar(hUᴴΔAV₁₀, (r, r)))
            UᴴΔAV[ind₀, :] .= hUᴴΔAV₁₀' .- aUᴴΔAV₁₀'
            UᴴΔAV[:, ind₀] .= hUᴴΔAV₁₀ .+ aUᴴΔAV₁₀
        end
        ΔA = mul!(ΔA, U₁, UᴴΔAV * V₁ᴴ, 1, 1) # add the contribution to ΔA
    end

    # Add the remaining contributions
    if m > r && !iszerotangent(ΔU₊) # ΔU₁ is already orthogonal to U₁
        ΔU₊ ./= transpose(S₀)
        ΔA = mul!(ΔA, ΔU₊, V₀ᴴ, 1, 1)
    end
    if n > r && !iszerotangent(ΔV₊ᴴ) # ΔV₁ᴴ is already orthogonal to V₁ᴴ
        ΔV₊ᴴ .= S₀ .\ ΔV₊ᴴ
        ΔA = mul!(ΔA, U₀, ΔV₊ᴴ, 1, 1)
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
    hUᴴΔAV, aUᴴΔAV, ΔU₊, ΔV₊ᴴ = check_and_prepare_svd_cotangents(
        U, S, Vᴴ, ΔU, ΔSmat, ΔVᴴ, p; degeneracy_atol, gauge_atol
    )
    UᴴΔAV = hUᴴΔAV .+ aUᴴΔAV
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
