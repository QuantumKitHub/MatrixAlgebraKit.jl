"""
    remove_eig_gauge_dependence!(ΔV, D, V)

Remove the gauge-dependent part from the cotangent `ΔV` of the eigenvector matrix `V`. The
eigenvectors are only determined up to complex phase (and unitary mixing for degenerate
eigenvalues), so the corresponding components of `ΔV` are projected out.
"""
function remove_eig_gauge_dependence!(
        ΔV, D, V;
        degeneracy_atol = MatrixAlgebraKit.default_pullback_gauge_atol(D)
    )
    gaugepart = V' * ΔV
    gaugepart[abs.(transpose(diagview(D)) .- diagview(D)) .>= degeneracy_atol] .= 0
    mul!(ΔV, V / (V' * V), gaugepart, -1, 1)
    return ΔV
end

"""
    remove_eigh_gauge_dependence!(ΔV, D, V)

Remove the gauge-dependent part from the cotangent `ΔV` of the Hermitian eigenvector matrix
`V`. The eigenvectors are only determined up to complex phase (and unitary mixing for
degenerate eigenvalues), so the corresponding anti-Hermitian components of `V' * ΔV` are
projected out.
"""
function remove_eigh_gauge_dependence!(
        ΔV, D, V;
        degeneracy_atol = MatrixAlgebraKit.default_pullback_gauge_atol(D)
    )
    gaugepart = V' * ΔV
    gaugepart = project_antihermitian!(gaugepart)
    gaugepart[abs.(transpose(diagview(D)) .- diagview(D)) .>= degeneracy_atol] .= 0
    mul!(ΔV, V, gaugepart, -1, 1)
    return ΔV
end

"""
    remove_svd_gauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ)

Remove the gauge-dependent part from the cotangents `ΔU` and `ΔVᴴ` of the SVD factors. The
singular vectors are only determined up to a common complex phase per singular value (and
unitary mixing for degenerate singular values), so the corresponding anti-Hermitian components
of `U₁' * ΔU₁ + Vᴴ₁ * ΔVᴴ₁'` are projected out. For the full SVD, the extra columns of `U`
and rows of `Vᴴ` beyond `min(m, n)` are additionally zeroed out.
"""
function remove_svd_gauge_dependence!(
        ΔU, ΔVᴴ, U, S, Vᴴ;
        degeneracy_atol = MatrixAlgebraKit.default_pullback_gauge_atol(S)
    )
    minmn = length(diagview(S))
    U₁ = view(U, :, 1:minmn)
    Vᴴ₁ = view(Vᴴ, 1:minmn, :)
    ΔU₁ = view(ΔU, :, 1:minmn)
    ΔVᴴ₁ = view(ΔVᴴ, 1:minmn, :)
    Sdiag = diagview(S)
    gaugepart = mul!(U₁' * ΔU₁, Vᴴ₁, ΔVᴴ₁', true, true)
    gaugepart = project_antihermitian!(gaugepart)
    gaugepart[abs.(transpose(Sdiag) .- Sdiag) .>= degeneracy_atol] .= 0
    mul!(ΔU₁, U₁, gaugepart, -1, 1)
    ΔU[:, (minmn + 1):end] .= 0
    ΔVᴴ[(minmn + 1):end, :] .= 0
    return ΔU, ΔVᴴ
end

"""
    remove_qr_gauge_dependence!(ΔQ, A, Q, R)

Remove the gauge-dependent part from the cotangent `ΔQ` of the full-QR orthogonal factor `Q`.
For the full QR decomposition, the extra columns of `Q` beyond `min(m, n)` are not uniquely
determined by `A`, so the corresponding part of `ΔQ` is projected to remove this ambiguity.
"""
function remove_qr_gauge_dependence!(ΔQ, A, Q, R)
    m, n = size(A)
    minmn = min(m, n)
    Q₁ = @view Q[:, 1:minmn]
    ΔQ₂ = @view ΔQ[:, (minmn + 1):end]
    Q₁ᴴΔQ₂ = Q₁' * ΔQ₂
    mul!(ΔQ₂, Q₁, Q₁ᴴΔQ₂)
    MatrixAlgebraKit.check_qr_full_cotangents(Q₁, ΔQ₂, Q₁ᴴΔQ₂)
    return ΔQ
end

"""
    remove_qr_null_gauge_dependence!(ΔN, A, N)

Remove the gauge-dependent part from the cotangent `ΔN` of the QR null space `N`. The null
space is only determined up to a unitary rotation, so `ΔN` is projected onto the column span
of the compact QR factor `Q₁`.
"""
function remove_qr_null_gauge_dependence!(ΔN, A, N)
    Q, _ = qr_compact(A)
    return mul!(ΔN, Q, Q' * ΔN)
end

"""
    remove_lq_gauge_dependence!(ΔQ, A, L, Q)

Remove the gauge-dependent part from the cotangent `ΔQ` of the full-LQ orthogonal factor `Q`.
For the full LQ decomposition, the extra rows of `Q` beyond `min(m, n)` are not uniquely
determined by `A`, so the corresponding part of `ΔQ` is projected to remove this ambiguity.
"""
function remove_lq_gauge_dependence!(ΔQ, A, L, Q)
    m, n = size(A)
    minmn = min(m, n)
    Q₁ = @view Q[1:minmn, :]
    ΔQ₂ = @view ΔQ[(minmn + 1):end, :]
    ΔQ₂Q₁ᴴ = ΔQ₂ * Q₁'
    mul!(ΔQ₂, ΔQ₂Q₁ᴴ, Q₁)
    MatrixAlgebraKit.check_lq_full_cotangents(Q₁, ΔQ₂, ΔQ₂Q₁ᴴ)
    return ΔQ
end

"""
    remove_lq_null_gauge_dependence!(ΔNᴴ, A, Nᴴ)

Remove the gauge-dependent part from the cotangent `ΔNᴴ` of the LQ null space `Nᴴ`. The null
space is only determined up to a unitary rotation, so `ΔNᴴ` is projected onto the row span of
the compact LQ factor `Q₁`.
"""
function remove_lq_null_gauge_dependence!(ΔNᴴ, A, Nᴴ)
    _, Q = lq_compact(A)
    ΔNᴴQᴴ = ΔNᴴ * Q'
    return mul!(ΔNᴴ, ΔNᴴQᴴ, Q)
end

"""
    remove_left_null_gauge_dependence!(ΔN, A, N)

Remove the gauge-dependent part from the cotangent `ΔN` of the left null space `N`. The null
space basis is only determined up to a unitary rotation, so `ΔN` is projected onto the column
span of the compact QR factor `Q₁` of `A`.
"""
function remove_left_null_gauge_dependence!(ΔN, A, N)
    Q, _ = qr_compact(A)
    mul!(ΔN, Q, Q' * ΔN)
    return ΔN
end

"""
    remove_right_null_gauge_dependence!(ΔNᴴ, A, Nᴴ)

Remove the gauge-dependent part from the cotangent `ΔNᴴ` of the right null space `Nᴴ`. The
null space basis is only determined up to a unitary rotation, so `ΔNᴴ` is projected onto the
row span of the compact LQ factor `Q₁` of `A`.
"""
function remove_right_null_gauge_dependence!(ΔNᴴ, A, Nᴴ)
    _, Q = lq_compact(A)
    mul!(ΔNᴴ, ΔNᴴ * Q', Q)
    return ΔNᴴ
end

function stabilize_eigvals!(D::AbstractVector)
    absD = collect(abs.(D))
    p = invperm(sortperm(collect(absD))) # rank of abs(D)
    # account for exact degeneracies in absolute value when having complex conjugate pairs
    for i in 1:(length(D) - 1)
        if absD[i] == absD[i + 1] # conjugate pairs will appear sequentially
            p[p .>= p[i + 1]] .-= 1 # lower the rank of all higher ones
        end
    end
    n = maximum(p)
    # rescale eigenvalues so that they lie on distinct radii in the complex plane
    # that are chosen randomly in non-overlapping intervals [10 * k/n, 10 * (k+0.5)/n)] for k=1,...,n
    radii = 10 .* ((1:n) .+ rand(real(eltype(D)), n) ./ 2) ./ n
    hD = sign.(collect(D)) .* radii[p]
    copyto!(D, hD)
    return D
end
function make_eig_matrix(T, sz)
    A = instantiate_matrix(T, sz)
    D, V = eig_full(A)
    stabilize_eigvals!(diagview(D))
    Ac = V * D * inv(V)
    Af = (eltype(T) <: Real) ? real(Ac) : Ac
    if T <: Diagonal
        copyto!(diagview(A), diagview(Af))
    else
        copyto!(A, Af)
    end
    return A
end
function make_eigh_matrix(T, sz)
    A = project_hermitian!(instantiate_matrix(T, sz))
    D, V = eigh_full(A)
    stabilize_eigvals!(diagview(D))
    return project_hermitian!(V * D * V')
end

function ad_qr_compact_setup(A)
    m, n = size(A)
    minmn = min(m, n)
    QR = qr_compact(A)
    T = eltype(A)
    ΔQ = randn!(similar(A, T, m, minmn))
    ΔR = randn!(similar(A, T, minmn, n))
    return QR, (ΔQ, ΔR)
end

function ad_qr_compact_setup(A::Diagonal)
    m, n = size(A)
    minmn = min(m, n)
    QR = qr_compact(A)
    T = eltype(A)
    ΔQ = Diagonal(randn!(similar(A.diag, T, m)))
    ΔR = Diagonal(randn!(similar(A.diag, T, m)))
    return QR, (ΔQ, ΔR)
end

function ad_qr_null_setup(A)
    m, n = size(A)
    minmn = min(m, n)
    Q, R = qr_compact(A)
    T = eltype(A)
    ΔN = Q * randn!(similar(A, T, minmn, max(0, m - minmn)))
    N = qr_null(A)
    return N, ΔN
end

function ad_qr_full_setup(A)
    m, n = size(A)
    minmn = min(m, n)
    T = eltype(A)
    Q, R = qr_full(A)
    Q1 = view(Q, 1:m, 1:minmn)
    ΔQ = randn!(similar(A, T, m, m))
    ΔQ2 = view(ΔQ, :, (minmn + 1):m)
    mul!(ΔQ2, Q1, Q1' * ΔQ2)
    ΔR = randn!(similar(A, T, m, n))
    return (Q, R), (ΔQ, ΔR)
end

ad_qr_full_setup(A::Diagonal) = ad_qr_compact_setup(A)

function ad_qr_rank_deficient_compact_setup(A)
    m, n = size(A)
    minmn = min(m, n)
    T = eltype(A)
    r = minmn - 5
    Ard = randn!(similar(A, T, m, r)) * randn!(similar(A, T, r, n))
    Q, R = qr_compact(Ard)
    QR = (Q, R)
    ΔQ = randn!(similar(A, T, m, minmn))
    Q1 = view(Q, 1:m, 1:r)
    Q2 = view(Q, 1:m, (r + 1):minmn)
    ΔQ2 = view(ΔQ, 1:m, (r + 1):minmn)
    MatrixAlgebraKit.zero!(ΔQ2)
    ΔR = randn!(similar(A, T, minmn, n))
    view(ΔR, (r + 1):minmn, :) .= 0
    return (Q, R), (ΔQ, ΔR)
end

function ad_qr_rank_deficient_compact_setup(A::Diagonal)
    m, n = size(A)
    minmn = min(m, n)
    T = eltype(A)
    r = minmn - 5
    Ard_ = randn!(similar(A, T, m))
    MatrixAlgebraKit.zero!(view(Ard_, (r + 1):m))
    Ard = Diagonal(Ard_)
    Q, R = qr_compact(Ard)
    ΔQ = Diagonal(randn!(similar(diagview(A), T, m)))
    ΔR = Diagonal(randn!(similar(diagview(A), T, m)))
    MatrixAlgebraKit.zero!(view(diagview(ΔQ), (r + 1):m))
    MatrixAlgebraKit.zero!(view(diagview(ΔR), (r + 1):m))
    return (Q, R), (ΔQ, ΔR)
end

function ad_lq_compact_setup(A)
    m, n = size(A)
    minmn = min(m, n)
    LQ = lq_compact(A)
    T = eltype(A)
    ΔL = randn!(similar(A, T, m, minmn))
    ΔQ = randn!(similar(A, T, minmn, n))
    return LQ, (ΔL, ΔQ)
end
ad_lq_compact_setup(A::Diagonal) = ad_qr_compact_setup(A)

function ad_lq_null_setup(A)
    m, n = size(A)
    minmn = min(m, n)
    T = eltype(A)
    L, Q = lq_compact(A)
    ΔNᴴ = randn!(similar(A, T, max(0, n - minmn), minmn)) * Q
    Nᴴ = randn!(similar(A, T, max(0, n - minmn), n))
    return Nᴴ, ΔNᴴ
end

function ad_lq_full_setup(A)
    m, n = size(A)
    minmn = min(m, n)
    T = eltype(A)
    L, Q = lq_full(A)
    Q1 = view(Q, 1:minmn, 1:n)
    ΔQ = randn!(similar(A, T, n, n))
    ΔQ2 = view(ΔQ, (minmn + 1):n, 1:n)
    ΔQ2 .= (ΔQ2 * Q1') * Q1
    ΔL = randn!(similar(A, T, m, n))
    return (L, Q), (ΔL, ΔQ)
end
ad_lq_full_setup(A::Diagonal) = ad_qr_full_setup(A)

function ad_lq_rank_deficient_compact_setup(A)
    m, n = size(A)
    minmn = min(m, n)
    T = eltype(A)
    r = minmn - 5
    Ard = randn!(similar(A, T, m, r)) * randn!(similar(A, T, r, n))
    L, Q = lq_compact(Ard)
    ΔL = randn!(similar(A, T, m, minmn))
    ΔQ = randn!(similar(A, T, minmn, n))
    Q1 = view(Q, 1:r, 1:n)
    Q2 = view(Q, (r + 1):minmn, 1:n)
    ΔQ2 = view(ΔQ, (r + 1):minmn, 1:n)
    ΔQ2 .= 0
    view(ΔL, :, (r + 1):minmn) .= 0
    return (L, Q), (ΔL, ΔQ)
end
ad_lq_rank_deficient_compact_setup(A::Diagonal) = ad_qr_rank_deficient_compact_setup(A)

function ad_eig_full_setup(A)
    m, n = size(A)
    T = eltype(A)
    DV = eig_full(A)
    D, V = DV
    Ddiag = diagview(D)
    ΔV = randn!(similar(A, complex(T), m, m))
    ΔV = remove_eig_gauge_dependence!(ΔV, D, V)
    ΔD = randn!(similar(A, complex(T), m, m))
    ΔD2 = Diagonal(randn!(similar(A, complex(T), m)))
    return DV, (ΔD, ΔV), (ΔD2, ΔV)
end

function ad_eig_full_setup(A::Diagonal)
    m, n = size(A)
    T = complex(eltype(A))
    DV = eig_full(A)
    D, V = DV
    ΔV = randn!(similar(A.diag, T, m, m))
    ΔV = remove_eig_gauge_dependence!(ΔV, D, V)
    ΔD = Diagonal(randn!(similar(A.diag, T, m)))
    ΔD2 = Diagonal(randn!(similar(A.diag, T, m)))
    return DV, (ΔD, ΔV), (ΔD2, ΔV)
end

function ad_eig_vals_setup(A)
    m, n = size(A)
    T = complex(eltype(A))
    D = eig_vals(A)
    ΔD = randn!(similar(A, complex(T), m))
    return D, ΔD
end

function ad_eig_vals_setup(A::Diagonal)
    m, n = size(A)
    T = complex(eltype(A))
    D = eig_vals(A)
    ΔD = randn!(similar(A.diag, T, m))
    return D, ΔD
end

function ad_eig_trunc_setup(A, truncalg)
    DV, ΔDV, ΔD2V = ad_eig_full_setup(A)
    ind = MatrixAlgebraKit.findtruncated(diagview(DV[1]), truncalg.trunc)
    Dtrunc = Diagonal(diagview(DV[1])[ind])
    Vtrunc = DV[2][:, ind]
    ΔDtrunc = Diagonal(diagview(ΔD2V[1])[ind])
    ΔVtrunc = ΔDV[2][:, ind]
    return DV, (Dtrunc, Vtrunc), ΔD2V, (ΔDtrunc, ΔVtrunc)
end

function ad_eigh_full_setup(A)
    m, n = size(A)
    T = eltype(A)
    DV = eigh_full(A)
    D, V = DV
    Ddiag = diagview(D)
    ΔV = randn!(similar(A, T, m, m))
    ΔV = remove_eigh_gauge_dependence!(ΔV, D, V)
    ΔD = randn!(similar(A, real(T), m, m))
    ΔD2 = Diagonal(randn!(similar(A, real(T), m)))
    return DV, (ΔD, ΔV), (ΔD2, ΔV)
end

function ad_eigh_vals_setup(A)
    m, n = size(A)
    T = eltype(A)
    D = eigh_vals(A)
    ΔD = randn!(similar(A, real(T), m))
    return D, ΔD
end

function ad_eigh_trunc_setup(A, truncalg)
    DV, ΔDV, ΔD2V = ad_eigh_full_setup(A)
    ind = MatrixAlgebraKit.findtruncated(diagview(DV[1]), truncalg.trunc)
    Dtrunc = Diagonal(diagview(DV[1])[ind])
    Vtrunc = DV[2][:, ind]
    ΔDtrunc = Diagonal(diagview(ΔD2V[1])[ind])
    ΔVtrunc = ΔDV[2][:, ind]
    return DV, (Dtrunc, Vtrunc), ΔD2V, (ΔDtrunc, ΔVtrunc)
end

function ad_svd_compact_setup(A)
    m, n = size(A)
    T = eltype(A)
    minmn = min(m, n)
    ΔU = randn!(similar(A, T, m, minmn))
    ΔS = randn!(similar(A, real(T), minmn, minmn))
    ΔS2 = Diagonal(randn!(similar(A, real(T), minmn)))
    ΔVᴴ = randn!(similar(A, T, minmn, n))
    U, S, Vᴴ = svd_compact(A)
    ΔU, ΔVᴴ = remove_svd_gauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ)
    return (U, S, Vᴴ), (ΔU, ΔS, ΔVᴴ), (ΔU, ΔS2, ΔVᴴ)
end

function ad_svd_compact_setup(A::Diagonal)
    m, n = size(A)
    T = eltype(A)
    minmn = min(m, n)
    ΔU = randn!(similar(A.diag, T, m, n))
    ΔS = Diagonal(randn!(similar(A.diag, real(T), minmn)))
    ΔS2 = Diagonal(randn!(similar(A.diag, real(T), minmn)))
    ΔVᴴ = randn!(similar(A.diag, T, m, n))
    U, S, Vᴴ = svd_compact(A)
    ΔU, ΔVᴴ = remove_svd_gauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ)
    return (U, S, Vᴴ), (ΔU, ΔS, ΔVᴴ), (ΔU, ΔS2, ΔVᴴ)
end

function ad_svd_full_setup(A)
    m, n = size(A)
    T = eltype(A)
    minmn = min(m, n)
    ΔU = randn!(similar(A, T, m, minmn))
    ΔS = randn!(similar(A, real(T), minmn, minmn))
    ΔS2 = Diagonal(randn!(similar(A, real(T), minmn)))
    ΔVᴴ = randn!(similar(A, T, minmn, n))
    U, S, Vᴴ = svd_compact(A)
    ΔU, ΔVᴴ = remove_svd_gauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ)
    ΔUfull = similar(A, T, m, m)
    ΔUfull .= zero(T)
    ΔSfull = similar(A, real(T), m, n)
    ΔSfull .= zero(real(T))
    ΔVᴴfull = similar(A, T, n, n)
    ΔVᴴfull .= zero(T)
    U, S, Vᴴ = svd_full(A)
    view(ΔUfull, :, 1:minmn) .= ΔU
    view(ΔVᴴfull, 1:minmn, :) .= ΔVᴴ
    diagview(ΔSfull)[1:minmn] .= diagview(ΔS2)
    return (U, S, Vᴴ), (ΔUfull, ΔSfull, ΔVᴴfull)
end

ad_svd_full_setup(A::Diagonal) = ad_svd_compact_setup(A)

function ad_svd_vals_setup(A)
    m, n = size(A)
    minmn = min(m, n)
    T = eltype(A)
    S = svd_vals(A)
    ΔS = randn!(similar(A, real(T), minmn))
    return S, ΔS
end

function ad_svd_trunc_setup(A, truncalg)
    USVᴴ, ΔUSVᴴ, ΔUS2Vᴴ = ad_svd_compact_setup(A)
    ind = MatrixAlgebraKit.findtruncated(diagview(USVᴴ[2]), truncalg.trunc)
    Strunc = Diagonal(diagview(USVᴴ[2])[ind])
    Utrunc = USVᴴ[1][:, ind]
    Vᴴtrunc = USVᴴ[3][ind, :]
    ΔStrunc = Diagonal(diagview(ΔUS2Vᴴ[2])[ind])
    ΔUtrunc = ΔUSVᴴ[1][:, ind]
    ΔVᴴtrunc = ΔUSVᴴ[3][ind, :]
    return USVᴴ, ΔUS2Vᴴ, (ΔUtrunc, ΔStrunc, ΔVᴴtrunc)
end

function ad_left_polar_setup(A)
    m, n = size(A)
    T = eltype(A)
    WP = left_polar(A)
    ΔWP = (randn!(similar(A, T, m, n)), randn!(similar(A, T, n, n)))
    return WP, ΔWP
end

function ad_left_polar_setup(A::Diagonal)
    m, n = size(A)
    T = eltype(A)
    WP = left_polar(A)
    ΔWP = (Diagonal(randn!(similar(A.diag))), randn!(similar(WP[2])))
    return WP, ΔWP
end

function ad_right_polar_setup(A)
    m, n = size(A)
    T = eltype(A)
    PWᴴ = right_polar(A)
    ΔPWᴴ = (randn!(similar(A, T, m, m)), randn!(similar(A, T, m, n)))
    return PWᴴ, ΔPWᴴ
end
function ad_right_polar_setup(A::Diagonal)
    m, n = size(A)
    T = eltype(A)
    PWᴴ = right_polar(A)
    ΔPWᴴ = (randn!(similar(PWᴴ[1])), Diagonal(randn!(similar(A.diag))))
    return PWᴴ, ΔPWᴴ
end

function ad_left_orth_setup(A)
    m, n = size(A)
    T = eltype(A)
    VC = left_orth(A)
    ΔVC = (randn!(similar(A, T, size(VC[1])...)), randn!(similar(A, T, size(VC[2])...)))
    return VC, ΔVC
end
function ad_left_orth_setup(A::Diagonal)
    m, n = size(A)
    T = eltype(A)
    VC = left_orth(A)
    ΔVC = (Diagonal(randn!(similar(A.diag, T, m))), Diagonal(randn!(similar(A.diag, T, m))))
    return VC, ΔVC
end

function ad_left_null_setup(A)
    m, n = size(A)
    T = eltype(A)
    N = left_orth(A; alg = :qr)[1] * randn!(similar(A, T, min(m, n), m - min(m, n)))
    ΔN = left_orth(A; alg = :qr)[1] * randn!(similar(A, T, min(m, n), m - min(m, n)))
    return N, ΔN
end

function ad_right_orth_setup(A)
    m, n = size(A)
    T = eltype(A)
    CVᴴ = right_orth(A)
    ΔCVᴴ = (randn!(similar(A, T, size(CVᴴ[1])...)), randn!(similar(A, T, size(CVᴴ[2])...)))
    return CVᴴ, ΔCVᴴ
end
ad_right_orth_setup(A::Diagonal) = ad_left_orth_setup(A)

function ad_right_null_setup(A)
    m, n = size(A)
    T = eltype(A)
    Nᴴ = randn!(similar(A, T, n - min(m, n), min(m, n))) * right_orth(A; alg = :lq)[2]
    ΔNᴴ = randn!(similar(A, T, n - min(m, n), min(m, n))) * right_orth(A; alg = :lq)[2]
    return Nᴴ, ΔNᴴ
end
