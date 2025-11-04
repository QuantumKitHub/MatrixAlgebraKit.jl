module MatrixAlgebraKitGenericLinearAlgebraExt

using MatrixAlgebraKit
using MatrixAlgebraKit: sign_safe, check_input
using GenericLinearAlgebra: svd, svdvals!, eigen, eigvals, Hermitian, qr
using LinearAlgebra: I, Diagonal

function MatrixAlgebraKit.default_svd_algorithm(::Type{T}; kwargs...) where {T <: StridedMatrix{<:Union{BigFloat, Complex{BigFloat}}}}
    return GLA_svd_QRIteration()
end

function MatrixAlgebraKit.svd_compact!(A::AbstractMatrix{T}, USVᴴ, alg::GLA_svd_QRIteration) where {T}
    check_input(svd_compact!, A, USVᴴ, alg)
    U, S, Vᴴ = USVᴴ
    Ũ, S̃, Ṽ = svd!(A)
    copyto!(U, Ũ)
    copyto!(S, Diagonal(S̃))
    copyto!(Vᴴ, Ṽ') # conjugation to account for difference in convention
    return U, S, Vᴴ
end

function MatrixAlgebraKit.svd_full!(A::AbstractMatrix{T}, USVᴴ, alg::GLA_svd_QRIteration) where {T}
    check_input(svd_full!, A, USVᴴ, alg)
    U, S, Vᴴ = USVᴴ
    m, n = size(A)
    minmn = min(m, n)
    if minmn == 0
        MatrixAlgebraKit.one!(U)
        MatrixAlgebraKit.zero!(S)
        MatrixAlgebraKit.one!(Vᴴ)
        return USVᴴ
    end
    S̃ = fill!(S, zero(T))
    U_compact, S_compact, V_compact = svd(A)
    S̃[1:minmn, 1:minmn] .= Diagonal(S_compact)
    copyto!(S, S̃)

    U = _gram_schmidt!(U, U_compact)
    Vᴴ = _gram_schmidt!(Vᴴ, V_compact; adjoint = true)

    return MatrixAlgebraKit.gaugefix!(svd_full!, U, S, Vᴴ, m, n)
end

function MatrixAlgebraKit.svd_vals!(A::AbstractMatrix{T}, S, alg::GLA_svd_QRIteration) where {T}
    check_input(svd_vals!, A, S, alg)
    S̃ = svdvals!(A)
    copyto!(S, S̃)
    return S
end

function MatrixAlgebraKit.default_eigh_algorithm(::Type{T}; kwargs...) where {T <: StridedMatrix{<:Union{BigFloat, Complex{BigFloat}}}}
    return GLA_eigh_Francis(; kwargs...)
end

function MatrixAlgebraKit.eigh_full!(A::AbstractMatrix{T}, DV, alg::GLA_eigh_Francis) where {T}
    check_input(eigh_full!, A, DV, alg)
    D, V = DV
    eigval, eigvec = eigen!(Hermitian(A); sortby = real)
    copyto!(D, Diagonal(eigval))
    copyto!(V, eigvec)
    return D, V
end

function MatrixAlgebraKit.eigh_vals!(A::AbstractMatrix{T}, D, alg::GLA_eigh_Francis) where {T}
    check_input(eigh_vals!, A, D, alg)
    D = eigvals(A; sortby = real)
    return real.(D)
end

function MatrixAlgebraKit.default_qr_algorithm(::Type{T}; kwargs...) where {T <: StridedMatrix{<:Union{BigFloat, Complex{BigFloat}}}}
    return GLA_QR_Householder(; kwargs...)
end

function MatrixAlgebraKit.qr_full!(A::AbstractMatrix, QR, alg::GLA_QR_Householder)
    Q, R = QR
    m, n = size(A)
    minmn = min(m, n)
    computeR = length(R) > 0

    Q_zero = zeros(eltype(Q), (m, minmn))
    R_zero = zeros(eltype(R), (minmn, n))
    Q_compact, R_compact = _gla_householder_qr!(A, Q_zero, R_zero; alg.kwargs...)
    Q = _gram_schmidt!(Q, Q_compact[:, 1:min(m, n)])
    if computeR
        R = fill!(R, zero(eltype(R)))
        R[1:minmn, 1:n] .= R_compact
    end
    return Q, R
end

function MatrixAlgebraKit.qr_compact!(A::AbstractMatrix, QR, alg::GLA_QR_Householder)
    check_input(qr_compact!, A, QR, alg)
    Q, R = QR
    Q, R = _gla_householder_qr!(A, Q, R; alg.kwargs...)
    return Q, R
end

function _gla_householder_qr!(A::AbstractMatrix{T}, Q, R; positive = false, blocksize = 1, pivoted = false) where {T}
    pivoted && throw(ArgumentError("Only pivoted = false implemented for GLA_QR_Householder."))
    (blocksize == 1) || throw(ArgumentError("Only blocksize = 1 implemented for GLA_QR_Householder."))

    m, n = size(A)
    k = min(m, n)
    computeR = length(R) > 0
    Q̃, R̃ = qr!(A)
    Q̃ = convert(Array, Q̃)
    if positive
        @inbounds for j in 1:k
            s = sign_safe(R̃[j, j])
            @simd for i in 1:m
                Q̃[i, j] *= s
            end
        end
    end
    copyto!(Q, Q̃)
    if computeR
        if positive
            @inbounds for j in n:-1:1
                @simd for i in 1:min(k, j)
                    R̃[i, j] = R̃[i, j] * conj(sign_safe(R̃[i, i]))
                end
            end
        end
        copyto!(R, R̃)
    end
    return Q, R
end

function _gram_schmidt(Q_compact)
    m, minmn = size(Q_compact)
    if minmn >= m
        return Q_compact
    end
    Q = zeros(eltype(Q_compact), (m, m))
    Q[:, 1:minmn] .= Q_compact
    for j in (minmn + 1):m
        v = rand(eltype(Q), m)
        for i in 1:(j - 1)
            r = sum([v[k] * conj(Q[k, i])] for k in 1:size(v)[1])[1]
            v .= v .- r * Q[:, i]
        end
        Q[:, j] = v ./ MatrixAlgebraKit.norm(v)
    end
    return Q
end

function _gram_schmidt!(Q, Q_compact; adjoint = false)
    Q̃ = _gram_schmidt(Q_compact)
    if adjoint
        copyto!(Q, Q̃')
    else
        copyto!(Q, Q̃)
    end
    return Q
end

function MatrixAlgebraKit.default_lq_algorithm(::Type{T}; kwargs...) where {T <: StridedMatrix{<:Union{BigFloat, Complex{BigFloat}}}}
    return MatrixAlgebraKit.LQViaTransposedQR(GLA_QR_Householder(; kwargs...))
end

end
