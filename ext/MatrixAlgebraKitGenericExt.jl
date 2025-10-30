module MatrixAlgebraKitGenericExt

using MatrixAlgebraKit
using MatrixAlgebraKit: LAPACK_SVDAlgorithm, LAPACK_EigAlgorithm, LAPACK_EighAlgorithm, LAPACK_QRIteration
using MatrixAlgebraKit: uppertriangular!
using MatrixAlgebraKit: @algdef, Algorithm, check_input
using MatrixAlgebraKit: sign_safe
using GenericLinearAlgebra
using GenericSchur
using LinearAlgebra: I, Diagonal

function MatrixAlgebraKit.default_svd_algorithm(::Type{T}; kwargs...) where {T <: StridedMatrix{<:Union{BigFloat, Complex{BigFloat}}}}
    return BigFloat_svd_QRIteration()
end

function MatrixAlgebraKit.svd_compact!(A::AbstractMatrix{T}, USVᴴ, alg::BigFloat_svd_QRIteration) where {T <: Union{BigFloat, Complex{BigFloat}}}
    check_input(svd_compact!, A, USVᴴ, alg)
    U, S, V = GenericLinearAlgebra.svd(A)
    return U, Diagonal(S), V' # conjugation to account for difference in convention
end

function MatrixAlgebraKit.svd_full!(A::AbstractMatrix{T}, USVᴴ, alg::BigFloat_svd_QRIteration)::Tuple{Matrix{T}, Matrix{BigFloat}, Matrix{T}} where {T <: Union{BigFloat, Complex{BigFloat}}}
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
    S̃ = zeros(eltype(S), size(S))
    U_compact, S_compact, V_compact = GenericLinearAlgebra.svd(A)
    S̃[1:minmn, 1:minmn] .= Diagonal(S_compact)
    Ũ = _gram_schmidt(U_compact)
    Ṽ = _gram_schmidt(V_compact)

    copyto!(U, Ũ)
    copyto!(S, S̃)
    copyto!(Vᴴ, Ṽ')

    return MatrixAlgebraKit.gaugefix!(svd_full!, U, S, Vᴴ, m, n)
end

function MatrixAlgebraKit.svd_vals!(A::AbstractMatrix{T}, S, alg::BigFloat_svd_QRIteration) where {T <: Union{BigFloat, Complex{BigFloat}}}
    check_input(svd_vals!, A, S, alg)
    S̃ = GenericLinearAlgebra.svdvals!(A)
    copyto!(S, S̃)
    return S
end

function MatrixAlgebraKit.default_eig_algorithm(::Type{T}; kwargs...) where {T <: StridedMatrix{<:Union{BigFloat, Complex{BigFloat}}}}
    return BigFloat_eig_Francis(; kwargs...)
end

function MatrixAlgebraKit.eig_full!(A::AbstractMatrix{T}, DV, alg::BigFloat_eig_Francis)::Tuple{Diagonal{Complex{BigFloat}}, Matrix{Complex{BigFloat}}} where {T <: Union{BigFloat, Complex{BigFloat}}}
    D, V = DV
    D̃, Ṽ = GenericSchur.eigen!(A)
    copyto!(D, Diagonal(D̃))
    copyto!(V, Ṽ)
    return D, V
end

function MatrixAlgebraKit.eig_vals!(A::AbstractMatrix{T}, D, alg::BigFloat_eig_Francis)::Vector{Complex{BigFloat}} where {T <: Union{BigFloat, Complex{BigFloat}}}
    check_input(eig_vals!, A, D, alg)
    eigval = GenericSchur.eigvals!(A)
    return eigval
end


function MatrixAlgebraKit.default_eigh_algorithm(::Type{T}; kwargs...) where {T <: StridedMatrix{<:Union{BigFloat, Complex{BigFloat}}}}
    return BigFloat_eigh_Francis(; kwargs...)
end

function MatrixAlgebraKit.eigh_full!(A::AbstractMatrix{T}, DV, alg::BigFloat_eigh_Francis)::Tuple{Diagonal{BigFloat}, Matrix{T}} where {T <: Union{BigFloat, Complex{BigFloat}}}
    check_input(eigh_full!, A, DV, alg)
    eigval, eigvec = GenericLinearAlgebra.eigen(A; sortby = real)
    return Diagonal(eigval), eigvec
end

function MatrixAlgebraKit.eigh_vals!(A::AbstractMatrix{T}, D, alg::BigFloat_eigh_Francis)::Vector{BigFloat} where {T <: Union{BigFloat, Complex{BigFloat}}}
    check_input(eigh_vals!, A, D, alg)
    D = GenericLinearAlgebra.eigvals(A; sortby = real)
    return real.(D)
end

function MatrixAlgebraKit.default_qr_algorithm(::Type{T}; kwargs...) where {T <: StridedMatrix{<:Union{BigFloat, Complex{BigFloat}}}}
    return BigFloat_QR_Householder(; kwargs...)
end

function MatrixAlgebraKit.qr_full!(A::AbstractMatrix, QR, alg::BigFloat_QR_Householder)
    Q, R = QR
    m, n = size(A)
    minmn = min(m, n)
    computeR = length(R) > 0

    Q_zero = zeros(eltype(Q), (m, minmn))
    R_zero = zeros(eltype(R), (minmn, n))
    Q_compact, R_compact = _bigfloat_householder_qr!(A, Q_zero, R_zero; alg.kwargs...)
    copyto!(Q, _gram_schmidt(Q_compact[:, 1:min(m, n)]))
    if computeR
        R̃ = zeros(eltype(R), m, n)
        R̃[1:minmn, 1:n] .= R_compact
        copyto!(R, R̃)
    end
    return Q, R
end

function MatrixAlgebraKit.lq_full!(A::AbstractMatrix, LQ, alg::BigFloat_LQ_Householder)
    L, Q = LQ
    m, n = size(A)
    minmn = min(m, n)
    computeL = length(L) > 0

    L_zero = zeros(eltype(L), (m, minmn))
    Q_zero = zeros(eltype(Q), (minmn, n))
    L_compact, Q_compact = _bigfloat_householder_lq!(A, L_zero, Q_zero; alg.kwargs...)
    copyto!(Q, _gram_schmidt(Q_compact'[:, 1:min(m, n)])')
    if computeL
        L̃ = zeros(eltype(L), m, n)
        L̃[1:m, 1:minmn] .= L_compact
        copyto!(L, L̃)
    end
    return L, Q
end

function MatrixAlgebraKit.qr_compact!(A::AbstractMatrix, QR, alg::BigFloat_QR_Householder)
    check_input(qr_compact!, A, QR, alg)
    Q, R = QR
    Q, R = _bigfloat_householder_qr!(A, Q, R; alg.kwargs...)
    return Q, R
end

function _bigfloat_householder_qr!(A::AbstractMatrix{T}, Q, R; positive = false, blocksize = 1, pivoted = false) where {T <: Union{BigFloat, Complex{BigFloat}}}
    pivoted && throw(ArgumentError("Only pivoted = false implemented for BigFloats."))
    (blocksize == 1) || throw(ArgumentError("Only blocksize = 1 implemented for BigFloats."))

    m, n = size(A)
    k = min(m, n)
    computeR = length(R) > 0
    Q̃, R̃ = GenericLinearAlgebra.qr(A)
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

function MatrixAlgebraKit.default_lq_algorithm(::Type{T}; kwargs...) where {T <: StridedMatrix{<:Union{BigFloat, Complex{BigFloat}}}}
    return BigFloat_LQ_Householder(; kwargs...)
end

function MatrixAlgebraKit.lq_compact!(A::AbstractMatrix, LQ, alg::BigFloat_LQ_Householder)
    check_input(lq_compact!, A, LQ, alg)
    L, Q = LQ
    L, Q = _bigfloat_householder_lq!(A, L, Q; alg.kwargs...)
    return L, Q
end


function _bigfloat_householder_lq!(A::AbstractMatrix{T}, L, Q; positive = false, blocksize = 1, pivoted = false) where {T <: Union{BigFloat, Complex{BigFloat}}}
    pivoted && throw(ArgumentError("Only pivoted = false implemented for BigFloats."))
    (blocksize == 1) || throw(ArgumentError("Only blocksize = 1 implemented for BigFloats."))

    m, n = size(A)
    k = min(m, n)
    computeL = length(L) > 0

    Q̃, R̃ = GenericLinearAlgebra.qr(A')
    Q̃ = convert(Array, Q̃)

    if positive
        @inbounds for j in 1:k
            s = sign_safe(R̃[j, j])
            @simd for i in 1:n
                Q̃[i, j] *= s
            end
        end
    end
    copyto!(Q, Q̃')
    if computeL
        if positive
            @inbounds for j in m:-1:1
                for i in 1:min(k, j)
                    R̃[i, j] = R̃[i, j] * conj(sign_safe(R̃[i, i]))
                end
            end
        end
        copyto!(L, R̃')

    end
    return L, Q
end

end
