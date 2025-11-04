module MatrixAlgebraKitGenericLinearAlgebraExt

using MatrixAlgebraKit
using MatrixAlgebraKit: sign_safe, check_input
using GenericLinearAlgebra: svd!, svdvals!, eigen!, eigvals!, Hermitian, qr!
using LinearAlgebra: I, Diagonal, rmul!, lmul!, transpose!, dot

function MatrixAlgebraKit.default_svd_algorithm(::Type{T}; kwargs...) where {T <: StridedMatrix{<:Union{BigFloat, Complex{BigFloat}}}}
    return GLA_QRIteration()
end

function MatrixAlgebraKit.svd_compact!(A::AbstractMatrix, USVᴴ, alg::GLA_QRIteration)
    check_input(svd_compact!, A, USVᴴ, alg)
    U, S, Vᴴ = USVᴴ
    F = svd!(A)
    copyto!(U, F.U)
    copyto!(S, Diagonal(F.S))
    copyto!(Vᴴ, F.Vt) # conjugation to account for difference in convention
    return U, S, Vᴴ
end

function MatrixAlgebraKit.svd_full!(A::AbstractMatrix, USVᴴ, alg::GLA_QRIteration)
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
    S = MatrixAlgebraKit.zero!(S)
    U_compact, S_compact, Vᴴ_compact = svd!(A)
    S[1:minmn, 1:minmn] .= Diagonal(S_compact)

    U = _gram_schmidt!(U, U_compact)
    Vᴴ = _gram_schmidt!(Vᴴ, Vᴴ_compact; adjoint = true)

    return MatrixAlgebraKit.gaugefix!(svd_full!, U, S, Vᴴ, m, n)
end

function MatrixAlgebraKit.svd_vals!(A::AbstractMatrix{T}, S, alg::GLA_QRIteration) where {T}
    check_input(svd_vals!, A, S, alg)
    S̃ = svdvals!(A)
    copyto!(S, S̃)
    return S
end

function MatrixAlgebraKit.default_eigh_algorithm(::Type{T}; kwargs...) where {T <: StridedMatrix{<:Union{BigFloat, Complex{BigFloat}}}}
    return GLA_QRIteration(; kwargs...)
end

function MatrixAlgebraKit.eigh_full!(A::AbstractMatrix{T}, DV, alg::GLA_QRIteration) where {T}
    check_input(eigh_full!, A, DV, alg)
    D, V = DV
    eigval, eigvec = eigen!(Hermitian(A); sortby = real)
    copyto!(D, Diagonal(eigval))
    copyto!(V, eigvec)
    return D, V
end

function MatrixAlgebraKit.eigh_vals!(A::AbstractMatrix{T}, D, alg::GLA_QRIteration) where {T}
    check_input(eigh_vals!, A, D, alg)
    return eigvals!(Hermitian(A); sortby = real)
end

function MatrixAlgebraKit.default_qr_algorithm(::Type{T}; kwargs...) where {T <: StridedMatrix{<:Union{BigFloat, Complex{BigFloat}}}}
    return GLA_HouseholderQR(; kwargs...)
end

function MatrixAlgebraKit.qr_full!(A::AbstractMatrix, QR, alg::GLA_HouseholderQR)
    Q, R = QR
    m, n = size(A)
    minmn = min(m, n)
    computeR = length(R) > 0

    Q_zero = zeros(eltype(Q), (m, minmn))
    R_zero = zeros(eltype(R), (minmn, n))
    Q_compact, R_compact = _gla_householder_qr!(A, Q_zero, R_zero; alg.kwargs...)
    Q = _gram_schmidt!(Q, view(Q_compact, :, 1:minmn))
    if computeR
        R[1:minmn, :] .= R_compact
        R[(minmn + 1):end, :] .= zero(eltype(R))
    end
    return Q, R
end

function MatrixAlgebraKit.qr_compact!(A::AbstractMatrix, QR, alg::GLA_HouseholderQR)
    check_input(qr_compact!, A, QR, alg)
    Q, R = QR
    Q, R = _gla_householder_qr!(A, Q, R; alg.kwargs...)
    return Q, R
end

function _gla_householder_qr!(A::AbstractMatrix{T}, Q, R; positive = false, blocksize = 1, pivoted = false) where {T}
    pivoted && throw(ArgumentError("Only pivoted = false implemented for GLA_HouseholderQR."))
    (blocksize == 1) || throw(ArgumentError("Only blocksize = 1 implemented for GLA_HouseholderQR."))

    m, n = size(A)
    k = min(m, n)
    computeR = length(R) > 0
    Q̃, R̃ = qr!(A)

    if k < m
        copyto!(Q, Q̃)
    else
        rmul!(MatrixAlgebraKit.one!(Q), Q̃)
    end
    if positive
        @inbounds for j in 1:k
            s = sign_safe(R̃[j, j])
            @simd for i in 1:m
                Q[i, j] *= s
            end
        end
    end
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

function _gram_schmidt!(Q, Q_compact; adjoint = false)
    m, minmn = size(Q_compact)
    Q[:, 1:minmn] .= Q_compact
    for j in (minmn + 1):m
        v = rand(eltype(Q), m)
        for i in 1:(j - 1)
            r = dot(view(Q, :, i), v)
            v .-= r * view(Q, :, i)
        end
        Q[:, j] = v ./ MatrixAlgebraKit.norm(v)
    end
    if adjoint
        copyto!(Q, Q')
    end
    return Q
end

function MatrixAlgebraKit.default_lq_algorithm(::Type{T}; kwargs...) where {T <: StridedMatrix{<:Union{BigFloat, Complex{BigFloat}}}}
    return MatrixAlgebraKit.LQViaTransposedQR(GLA_HouseholderQR(; kwargs...))
end

end
