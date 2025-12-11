function remove_svdgauge_dependence!(
        ΔU, ΔVᴴ, U, S, Vᴴ;
        degeneracy_atol = MatrixAlgebraKit.default_pullback_gauge_atol(S)
    )
    gaugepart = mul!(U' * ΔU, Vᴴ, ΔVᴴ', true, true)
    gaugepart = project_antihermitian!(gaugepart)
    gaugepart[abs.(transpose(diagview(S)) .- diagview(S)) .>= degeneracy_atol] .= 0
    mul!(ΔU, U, gaugepart, -1, 1)
    return ΔU, ΔVᴴ
end
function remove_eiggauge_dependence!(
        ΔV, D, V;
        degeneracy_atol = MatrixAlgebraKit.default_pullback_gauge_atol(D)
    )
    gaugepart = V' * ΔV
    gaugepart[abs.(transpose(diagview(D)) .- diagview(D)) .>= degeneracy_atol] .= 0
    mul!(ΔV, V / (V' * V), gaugepart, -1, 1)
    return ΔV
end
function remove_eighgauge_dependence!(
        ΔV, D, V;
        degeneracy_atol = MatrixAlgebraKit.default_pullback_gauge_atol(D)
    )
    gaugepart = V' * ΔV
    gaugepart = project_antihermitian!(gaugepart)
    gaugepart[abs.(transpose(diagview(D)) .- diagview(D)) .>= degeneracy_atol] .= 0
    mul!(ΔV, V, gaugepart, -1, 1)
    return ΔV
end

function ad_qr_compact_setup(rng, A)
    m, n = size(A)
    minmn = min(m, n)
    QR = qr_compact(A)
    T = eltype(A)
    ΔQ = randn(rng, T, m, minmn)
    ΔR = randn(rng, T, minmn, n)
    return QR, (ΔQ, ΔR)
end

function ad_qr_null_setup(rng, A)
    m, n = size(A)
    minmn = min(m, n)
    Q, R = qr_compact(A)
    T = eltype(A)
    ΔN = Q * randn(rng, T, minmn, max(0, m - minmn))
    N = qr_null(A)
    return N, ΔN
end

function ad_qr_full_setup(rng, A)
    m, n = size(A)
    minmn = min(m, n)
    T = eltype(A)
    Q, R = qr_full(A)
    Q1 = view(Q, 1:m, 1:minmn)
    ΔQ = randn(rng, T, m, m)
    ΔQ2 = view(ΔQ, :, (minmn + 1):m)
    mul!(ΔQ2, Q1, Q1' * ΔQ2)
    ΔR = randn(rng, T, m, n)
    return (Q, R), (ΔQ, ΔR)
end

function ad_qr_rd_compact_setup(rng, A)
    m, n = size(A)
    minmn = min(m, n)
    T = eltype(A)
    r = minmn - 5
    Ard = randn(rng, T, m, r) * randn(rng, T, r, n)
    Q, R = qr_compact(Ard)
    QR = (Q, R)
    ΔQ = randn(rng, T, m, minmn)
    Q1 = view(Q, 1:m, 1:r)
    Q2 = view(Q, 1:m, (r + 1):minmn)
    ΔQ2 = view(ΔQ, 1:m, (r + 1):minmn)
    ΔQ2 .= 0
    ΔR = randn(rng, T, minmn, n)
    view(ΔR, (r + 1):minmn, :) .= 0
    return (Q, R), (ΔQ, ΔR)
end

function ad_lq_compact_setup(rng, A)
    m, n = size(A)
    minmn = min(m, n)
    LQ = lq_compact(A)
    T = eltype(A)
    ΔL = randn(rng, T, m, minmn)
    ΔQ = randn(rng, T, minmn, n)
    return LQ, (ΔL, ΔQ)
end

function ad_lq_null_setup(rng, A)
    m, n = size(A)
    minmn = min(m, n)
    T = eltype(A)
    L, Q = lq_compact(A)
    ΔNᴴ = randn(rng, T, max(0, n - minmn), minmn) * Q
    Nᴴ = randn(rng, T, max(0, n - minmn), n)
    return Nᴴ, ΔNᴴ
end

function ad_lq_full_setup(rng, A)
    m, n = size(A)
    minmn = min(m, n)
    T = eltype(A)
    L, Q = lq_full(A)
    Q1 = view(Q, 1:minmn, 1:n)
    ΔQ = randn(rng, T, n, n)
    ΔQ2 = view(ΔQ, (minmn + 1):n, 1:n)
    mul!(ΔQ2, ΔQ2 * Q1', Q1)
    ΔL = randn(rng, T, m, n)
    return (L, Q), (ΔL, ΔQ)
end

function ad_lq_rd_compact_setup(rng, A)
    m, n = size(A)
    minmn = min(m, n)
    T = eltype(A)
    r = minmn - 5
    Ard = randn(rng, T, m, r) * randn(rng, T, r, n)
    L, Q = lq_compact(Ard)
    ΔL = randn(rng, T, m, minmn)
    ΔQ = randn(rng, T, minmn, n)
    Q1 = view(Q, 1:r, 1:n)
    Q2 = view(Q, (r + 1):minmn, 1:n)
    ΔQ2 = view(ΔQ, (r + 1):minmn, 1:n)
    ΔQ2 .= 0
    view(ΔL, :, (r + 1):minmn) .= 0
    return (L, Q), (ΔL, ΔQ)
end

function ad_eig_full_setup(rng, A)
    m, n = size(A)
    T = eltype(A)
    DV = eig_full(A)
    D, V = DV
    Ddiag = diagview(D)
    ΔV = randn(rng, complex(T), m, m)
    ΔV = remove_eiggauge_dependence!(ΔV, D, V)
    ΔD = randn(rng, complex(T), m, m)
    ΔD2 = Diagonal(randn(rng, complex(T), m))
    return DV, (ΔD, ΔV), (ΔD2, ΔV)
end

function ad_eig_vals_setup(rng, A)
    m, n = size(A)
    T = eltype(A)
    D = eig_vals(A)
    ΔD = randn(rng, complex(T), m)
    return D, ΔD
end

function ad_eig_trunc_setup(rng, A, truncalg)
    m, n = size(A)
    T = eltype(A)
    DV = eig_full(A)
    D, V = DV
    Ddiag = diagview(D)
    ΔV = randn(rng, complex(T), m, m)
    ΔV = remove_eiggauge_dependence!(ΔV, D, V)
    ΔD = randn(rng, complex(T), m, m)
    ΔD2 = Diagonal(randn(rng, complex(T), m))
    ind = MatrixAlgebraKit.findtruncated(Ddiag, truncalg.trunc)
    Dtrunc = Diagonal(diagview(D)[ind])
    Vtrunc = V[:, ind]
    ΔDtrunc = Diagonal(diagview(ΔD2)[ind])
    ΔVtrunc = ΔV[:, ind]
    return DV, (ΔD2, ΔV), (ΔDtrunc, ΔVtrunc)
end

function copy_eigh_full(A; kwargs...)
    A = (A + A') / 2
    return eigh_full(A; kwargs...)
end

function copy_eigh_full!(A, DV; kwargs...)
    A = (A + A') / 2
    return eigh_full!(A, DV; kwargs...)
end

function copy_eigh_vals(A; kwargs...)
    A = (A + A') / 2
    return eigh_vals(A; kwargs...)
end

function copy_eigh_vals!(A, D; kwargs...)
    A = (A + A') / 2
    return eigh_vals!(A, D; kwargs...)
end

function copy_eigh_trunc(A, alg; kwargs...)
    A = (A + A') / 2
    return eigh_trunc(A, alg; kwargs...)
end

function copy_eigh_trunc!(A, DV, alg; kwargs...)
    A = (A + A') / 2
    return eigh_trunc!(A, DV, alg; kwargs...)
end

MatrixAlgebraKit.copy_input(::typeof(copy_eigh_full), A) = MatrixAlgebraKit.copy_input(eigh_full, A)
MatrixAlgebraKit.copy_input(::typeof(copy_eigh_vals), A) = MatrixAlgebraKit.copy_input(eigh_vals, A)
MatrixAlgebraKit.copy_input(::typeof(copy_eigh_trunc), A) = MatrixAlgebraKit.copy_input(eigh_trunc, A)

function ad_eigh_full_setup(rng, A)
    m, n = size(A)
    T = eltype(A)
    DV = eigh_full(A)
    D, V = DV
    Ddiag = diagview(D)
    ΔV = randn(rng, T, m, m)
    ΔV = remove_eighgauge_dependence!(ΔV, D, V)
    ΔD = randn(rng, real(T), m, m)
    ΔD2 = Diagonal(randn(rng, real(T), m))
    return DV, (ΔD, ΔV), (ΔD2, ΔV)
end

function ad_eigh_vals_setup(rng, A)
    m, n = size(A)
    T = eltype(A)
    D = eigh_vals(A)
    ΔD = randn(rng, real(T), m)
    return D, ΔD
end

function ad_eigh_trunc_setup(rng, A, truncalg)
    m, n = size(A)
    T = eltype(A)
    DV = eigh_full(A)
    D, V = DV
    Ddiag = diagview(D)
    ΔV = randn(rng, T, m, m)
    ΔV = remove_eighgauge_dependence!(ΔV, D, V)
    ΔD = randn(rng, real(T), m, m)
    ΔD2 = Diagonal(randn(rng, real(T), m))
    ind = MatrixAlgebraKit.findtruncated(Ddiag, truncalg.trunc)
    Dtrunc = Diagonal(diagview(D)[ind])
    Vtrunc = V[:, ind]
    ΔDtrunc = Diagonal(diagview(ΔD2)[ind])
    ΔVtrunc = ΔV[:, ind]
    return DV, (ΔD2, ΔV), (ΔDtrunc, ΔVtrunc)
end

function ad_svd_compact_setup(rng, A)
    m, n = size(A)
    T = eltype(A)
    minmn = min(m, n)
    ΔU = randn(rng, T, m, minmn)
    ΔS = randn(rng, real(T), minmn, minmn)
    ΔS2 = Diagonal(randn(rng, real(T), minmn))
    ΔVᴴ = randn(rng, T, minmn, n)
    U, S, Vᴴ = svd_compact(A)
    ΔU, ΔVᴴ = remove_svdgauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ)
    return (U, S, Vᴴ), (ΔU, ΔS, ΔVᴴ), (ΔU, ΔS2, ΔVᴴ)
end

function ad_svd_full_setup(rng, A)
    m, n = size(A)
    T = eltype(A)
    minmn = min(m, n)
    ΔU = randn(rng, T, m, minmn)
    ΔS = randn(rng, real(T), minmn, minmn)
    ΔS2 = Diagonal(randn(rng, real(T), minmn))
    ΔVᴴ = randn(rng, T, minmn, n)
    U, S, Vᴴ = svd_compact(A)
    ΔU, ΔVᴴ = remove_svdgauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ)
    ΔUfull = zeros(T, m, m)
    ΔSfull = zeros(real(T), m, n)
    ΔVᴴfull = zeros(T, n, n)
    U, S, Vᴴ = svd_full(A)
    view(ΔUfull, :, 1:minmn) .= ΔU
    view(ΔVᴴfull, 1:minmn, :) .= ΔVᴴ
    diagview(ΔSfull)[1:minmn] .= diagview(ΔS2)
    return (U, S, Vᴴ), (ΔUfull, ΔSfull, ΔVᴴfull)
end

function ad_svd_vals_setup(rng, A)
    m, n = size(A)
    minmn = min(m, n)
    T = eltype(A)
    S = svd_vals(A)
    ΔS = randn(rng, real(T), minmn)
    return S, ΔS
end

function ad_svd_trunc_setup(rng, A, truncalg)
    m, n = size(A)
    minmn = min(m, n)
    T = eltype(A)
    ΔU = randn(rng, T, m, minmn)
    ΔS = randn(rng, real(T), minmn, minmn)
    ΔS2 = Diagonal(randn(rng, real(T), minmn))
    ΔVᴴ = randn(rng, T, minmn, n)
    U, S, Vᴴ = svd_compact(A)
    ΔU, ΔVᴴ = remove_svdgauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ)
    ind = MatrixAlgebraKit.findtruncated(diagview(S), truncalg.trunc)
    Strunc = Diagonal(diagview(S)[ind])
    Utrunc = U[:, ind]
    Vᴴtrunc = Vᴴ[ind, :]
    ΔStrunc = Diagonal(diagview(ΔS2)[ind])
    ΔUtrunc = ΔU[:, ind]
    ΔVᴴtrunc = ΔVᴴ[ind, :]
    return (U, S, Vᴴ), (ΔU, ΔS2, ΔVᴴ), (ΔUtrunc, ΔStrunc, ΔVᴴtrunc)
end

function ad_left_polar_setup(rng, A)
    m, n = size(A)
    T = eltype(A)
    WP = left_polar(A)
    ΔWP = (randn(rng, T, m, n), randn(rng, T, n, n))
    return WP, ΔWP
end

function ad_right_polar_setup(rng, A)
    m, n = size(A)
    T = eltype(A)
    PWᴴ = right_polar(A)
    ΔPWᴴ = (randn(rng, T, m, m), randn(rng, T, m, n))
    return PWᴴ, ΔPWᴴ
end

function ad_left_orth_setup(rng, A)
    m, n = size(A)
    T = eltype(A)
    VC = left_orth(A)
    ΔVC = (randn(rng, T, size(VC[1])...), randn(rng, T, size(VC[2])...))
    return VC, ΔVC
end

function ad_left_null_setup(rng, A)
    m, n = size(A)
    T = eltype(A)
    N = left_orth(A; alg = :qr)[1] * randn(rng, T, min(m, n), m - min(m, n))
    ΔN = left_orth(A; alg = :qr)[1] * randn(rng, T, min(m, n), m - min(m, n))
    return N, ΔN
end

function ad_right_orth_setup(rng, A)
    m, n = size(A)
    T = eltype(A)
    CVᴴ = right_orth(A)
    ΔCVᴴ = (randn(rng, T, size(CVᴴ[1])...), randn(rng, T, size(CVᴴ[2])...))
    return CVᴴ, ΔCVᴴ
end

function ad_right_null_setup(rng, A)
    m, n = size(A)
    T = eltype(A)
    Nᴴ = randn(rng, T, n - min(m, n), min(m, n)) * right_orth(A; alg = :lq)[2]
    ΔNᴴ = randn(rng, T, n - min(m, n), min(m, n)) * right_orth(A; alg = :lq)[2]
    return Nᴴ, ΔNᴴ
end
