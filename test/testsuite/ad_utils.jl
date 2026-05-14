using MatrixAlgebraKit: remove_svd_gauge_dependence!,
    remove_eig_gauge_dependence!, remove_eigh_gauge_dependence!,
    remove_qr_gauge_dependence!, remove_qr_null_gauge_dependence!,
    remove_lq_gauge_dependence!, remove_lq_null_gauge_dependence!,
    remove_left_null_gauge_dependence!, remove_right_null_gauge_dependence!

structured_randn!(A::AbstractMatrix) = randn!(A)
structured_randn!(A::Diagonal) = (randn!(diagview(A)); return A)

"""
    call_and_zero!(f!, A, alg)

Helper for testing in-place Mooncake rules.
Calls `f!(A, alg)`, followed by zeroing out `A` and returns the output of `f!`.
This allows `Mooncake.TestUtils.test_rule` to verify the reverse rule of `f!` through finite differences,
without counting the contributions of `A`, as this is used solely as scratch space.
"""
function call_and_zero!(f!, A, alg)
    F′ = f!(A, alg)
    MatrixAlgebraKit.zero!(A)
    return F′
end

is_cpu(A) = typeof(parent(A)) <: Array

"""
    project_hermitian_inplace!(A, alg)

Wrapper for `project_hermitian!(A, A, alg)`, invoked this way
to avoid Enzyme's finite differences comparison getting confused.
"""
project_hermitian_inplace!(A, alg) = project_hermitian!(A, A, alg)

"""
    project_antihermitian_inplace!(A, alg)

Wrapper for `project_hermitian!(A, A, alg)`, invoked this way
to avoid Enzyme's finite differences comparison getting confused.
"""
project_antihermitian_inplace!(A, alg) = project_antihermitian!(A, A, alg)


enzyme_fdm(T) = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)


"""
    eigh_wrapper(f, A, alg)

Wrapper that symmetrizes `A` before calling `f(A, alg)`. Used to test Hermitian
eigendecomposition rules on a general matrix by first projecting onto the Hermitian subspace.
"""
eigh_wrapper(f, A, alg) = f(project_hermitian(A), alg)

"""
    eigh!_wrapper(f!, A, alg)

Wrapper that symmetrizes `A` in-place before calling `f!(A, alg)`, then zeros `A`. Used to
test in-place Hermitian eigendecomposition rules via Mooncake's non-primitive AD path.
"""
eigh!_wrapper(f!, A, alg) = (F = f!(project_hermitian!(A), alg); MatrixAlgebraKit.zero!(A); F)

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
    QR = qr_compact(A)
    ΔQR = structured_randn!.(similar.(QR))
    remove_qr_gauge_dependence!(ΔQR..., A, QR...)
    return QR, ΔQR
end

function ad_qr_null_setup(A)
    N = qr_null(A)
    ΔN = structured_randn!(similar(N))
    remove_qr_null_gauge_dependence!(ΔN, A, N)
    return N, ΔN
end

function ad_qr_full_setup(A)
    QR = qr_full(A)
    ΔQR = structured_randn!.(similar.(QR))
    remove_qr_gauge_dependence!(ΔQR..., A, QR...)
    return QR, ΔQR
end

function ad_lq_compact_setup(A)
    LQ = lq_compact(A)
    ΔLQ = structured_randn!.(similar.(LQ))
    remove_lq_gauge_dependence!(ΔLQ..., A, LQ...)
    return LQ, ΔLQ
end

function ad_lq_null_setup(A)
    Nᴴ = lq_null(A)
    ΔNᴴ = structured_randn!(similar(Nᴴ))
    remove_lq_null_gauge_dependence!(ΔNᴴ, A, Nᴴ)
    return Nᴴ, ΔNᴴ
end

function ad_lq_full_setup(A)
    LQ = lq_full(A)
    ΔLQ = structured_randn!.(similar.(LQ))
    remove_lq_gauge_dependence!(ΔLQ..., A, LQ...)
    return LQ, ΔLQ
end

function ad_eig_full_setup(A)
    D, V = eig_full(A)
    ΔD, ΔV = structured_randn!.(similar.((D, V)))
    ΔV = remove_eig_gauge_dependence!(ΔV, D, V)
    return (D, V), (ΔD, ΔV)
end

function ad_eig_vals_setup(A)
    D = eig_vals(A)
    ΔD = randn!(similar(D))
    return D, ΔD
end

function ad_eig_trunc_setup(A, truncalg)
    DV, ΔDV = ad_eig_full_setup(A)
    ind = MatrixAlgebraKit.findtruncated(diagview(DV[1]), truncalg.trunc)
    Dtrunc = Diagonal(diagview(DV[1])[ind])
    Vtrunc = DV[2][:, ind]
    ΔDtrunc = Diagonal(diagview(ΔDV[1])[ind])
    ΔVtrunc = ΔDV[2][:, ind]
    return DV, (Dtrunc, Vtrunc), ΔDV, (ΔDtrunc, ΔVtrunc)
end

function ad_eigh_full_setup(A)
    D, V = eigh_full(A)
    ΔD, ΔV = structured_randn!.(similar.((D, V)))
    ΔV = remove_eigh_gauge_dependence!(ΔV, D, V)
    return (D, V), (ΔD, ΔV)
end

function ad_eigh_vals_setup(A)
    D = eigh_vals(A)
    ΔD = randn!(similar(D))
    return D, ΔD
end

function ad_eigh_trunc_setup(A, truncalg)
    DV, ΔDV = ad_eigh_full_setup(A)
    ind = MatrixAlgebraKit.findtruncated(diagview(DV[1]), truncalg.trunc)
    Dtrunc = Diagonal(diagview(DV[1])[ind])
    Vtrunc = DV[2][:, ind]
    ΔDtrunc = Diagonal(diagview(ΔDV[1])[ind])
    ΔVtrunc = ΔDV[2][:, ind]
    return DV, (Dtrunc, Vtrunc), ΔDV, (ΔDtrunc, ΔVtrunc)
end

function ad_svd_compact_setup(A)
    U, S, Vᴴ = svd_compact(A)
    ΔU, ΔS, ΔVᴴ = structured_randn!.(similar.((U, S, Vᴴ)))
    ΔU, ΔVᴴ = remove_svd_gauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ)
    return (U, S, Vᴴ), (ΔU, ΔS, ΔVᴴ)
end

function ad_svd_full_setup(A)
    U, S, Vᴴ = svd_full(A)
    ΔU = structured_randn!(similar(U))
    ΔVᴴ = structured_randn!(similar(Vᴴ))
    ΔU, ΔVᴴ = remove_svd_gauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ)
    ΔS = zero(S)
    randn!(diagview(ΔS))
    return (U, S, Vᴴ), (ΔU, ΔS, ΔVᴴ)
end

function ad_svd_vals_setup(A)
    S = svd_vals(A)
    ΔS = randn!(similar(S))
    return S, ΔS
end

function ad_svd_trunc_setup(A, truncalg)
    USVᴴ, ΔUSVᴴ = ad_svd_compact_setup(A)
    ind = MatrixAlgebraKit.findtruncated(diagview(USVᴴ[2]), truncalg.trunc)
    Strunc = Diagonal(diagview(USVᴴ[2])[ind])
    Utrunc = USVᴴ[1][:, ind]
    Vᴴtrunc = USVᴴ[3][ind, :]
    ΔStrunc = Diagonal(diagview(ΔUSVᴴ[2])[ind])
    ΔUtrunc = ΔUSVᴴ[1][:, ind]
    ΔVᴴtrunc = ΔUSVᴴ[3][ind, :]
    return USVᴴ, (Utrunc, Strunc, Vᴴtrunc), ΔUSVᴴ, (ΔUtrunc, ΔStrunc, ΔVᴴtrunc)
end

function ad_left_polar_setup(A)
    WP = left_polar(A)
    ΔWP = structured_randn!.(similar.(WP))
    return WP, ΔWP
end

function ad_right_polar_setup(A)
    PWᴴ = right_polar(A)
    ΔPWᴴ = structured_randn!.(similar.(PWᴴ))
    return PWᴴ, ΔPWᴴ
end

function ad_left_orth_setup(A)
    VC = left_orth(A)
    ΔVC = structured_randn!.(similar.(VC))
    return VC, ΔVC
end
function ad_left_orth_setup(A::Diagonal)
    VC = left_orth(A)
    ΔVC = structured_randn!.(similar.(VC))
    return VC, ΔVC
end

function ad_left_null_setup(A)
    m, n = size(A)
    T = eltype(A)
    N = left_orth(A)[1] * randn!(similar(A, T, min(m, n), m - min(m, n)))
    ΔN = left_orth(A)[1] * randn!(similar(A, T, min(m, n), m - min(m, n)))
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
    Nᴴ = randn!(similar(A, T, n - min(m, n), min(m, n))) * right_orth(A)[2]
    ΔNᴴ = randn!(similar(A, T, n - min(m, n), min(m, n))) * right_orth(A)[2]
    return Nᴴ, ΔNᴴ
end
