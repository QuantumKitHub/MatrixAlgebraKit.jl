module MatrixAlgebraKitGenericLinearAlgebraExt

using MatrixAlgebraKit
using MatrixAlgebraKit: sign_safe, check_input, diagview
using GenericLinearAlgebra: svd!, svdvals!, eigen!, eigvals!, Hermitian, qr!
using LinearAlgebra: I, Diagonal, lmul!

function MatrixAlgebraKit.default_svd_algorithm(::Type{T}; kwargs...) where {T <: StridedMatrix{<:Union{BigFloat, Complex{BigFloat}}}}
    return GLA_QRIteration()
end

for f! in (:svd_compact!, :svd_full!, :svd_vals!)
    @eval MatrixAlgebraKit.initialize_output(::typeof($f!), A::AbstractMatrix, ::GLA_QRIteration) = nothing
end

function MatrixAlgebraKit.svd_compact!(A::AbstractMatrix, USVᴴ, ::GLA_QRIteration)
    F = svd!(A)
    U, S, Vᴴ = F.U, Diagonal(F.S), F.Vt
    return MatrixAlgebraKit.gaugefix!(svd_compact!, U, S, Vᴴ, size(A)...)
end

function MatrixAlgebraKit.svd_full!(A::AbstractMatrix, USVᴴ, ::GLA_QRIteration)
    F = svd!(A; full = true)
    U, Vᴴ = F.U, F.Vt
    S = MatrixAlgebraKit.zero!(similar(F.S, (size(U, 2), size(Vᴴ, 1))))
    diagview(S) .= F.S
    return MatrixAlgebraKit.gaugefix!(svd_full!, U, S, Vᴴ, size(A)...)
end

function MatrixAlgebraKit.svd_vals!(A::AbstractMatrix, S, ::GLA_QRIteration)
    return svdvals!(A)
end

function MatrixAlgebraKit.default_eigh_algorithm(::Type{T}; kwargs...) where {T <: StridedMatrix{<:Union{BigFloat, Complex{BigFloat}}}}
    return GLA_QRIteration(; kwargs...)
end

for f! in (:eigh_full!, :eigh_vals!)
    @eval MatrixAlgebraKit.initialize_output(::typeof($f!), A::AbstractMatrix, ::GLA_QRIteration) = nothing
end

function MatrixAlgebraKit.eigh_full!(A::AbstractMatrix, DV, ::GLA_QRIteration)
    eigval, eigvec = eigen!(Hermitian(A); sortby = real)
    return Diagonal(eigval::AbstractVector{real(eltype(A))}), eigvec::AbstractMatrix{eltype(A)}
end

function MatrixAlgebraKit.eigh_vals!(A::AbstractMatrix, D, ::GLA_QRIteration)
    return eigvals!(Hermitian(A); sortby = real)
end

function MatrixAlgebraKit.default_qr_algorithm(::Type{T}; kwargs...) where {T <: StridedMatrix{<:Union{BigFloat, Complex{BigFloat}}}}
    return GLA_HouseholderQR(; kwargs...)
end

function MatrixAlgebraKit.qr_full!(A::AbstractMatrix, QR, alg::GLA_HouseholderQR)
    check_input(qr_full!, A, QR, alg)
    Q, R = QR
    return _gla_householder_qr!(A, Q, R; alg.kwargs...)
end

function MatrixAlgebraKit.qr_compact!(A::AbstractMatrix, QR, alg::GLA_HouseholderQR)
    check_input(qr_compact!, A, QR, alg)
    Q, R = QR
    return _gla_householder_qr!(A, Q, R; alg.kwargs...)
end

function _gla_householder_qr!(A::AbstractMatrix, Q, R; positive = false, blocksize = 1, pivoted = false)
    pivoted && throw(ArgumentError("Only pivoted = false implemented for GLA_HouseholderQR."))
    (blocksize == 1) || throw(ArgumentError("Only blocksize = 1 implemented for GLA_HouseholderQR."))

    m, n = size(A)
    k = min(m, n)
    Q̃, R̃ = qr!(A)
    lmul!(Q̃, MatrixAlgebraKit.one!(Q))

    if positive
        @inbounds for j in 1:k
            s = sign_safe(R̃[j, j])
            @simd for i in 1:m
                Q[i, j] *= s
            end
        end
    end

    computeR = length(R) > 0
    if computeR
        if positive
            @inbounds for j in n:-1:1
                @simd for i in 1:min(k, j)
                    R[i, j] = R̃[i, j] * conj(sign_safe(R̃[i, i]))
                end
                @simd for i in (min(k, j) + 1):size(R, 1)
                    R[i, j] = zero(eltype(R))
                end
            end
        else
            R[1:k, :] .= R̃
            MatrixAlgebraKit.zero!(@view(R[(k + 1):end, :]))
        end
    end
    return Q, R
end

function MatrixAlgebraKit.default_lq_algorithm(::Type{T}; kwargs...) where {T <: StridedMatrix{<:Union{BigFloat, Complex{BigFloat}}}}
    return MatrixAlgebraKit.LQViaTransposedQR(GLA_HouseholderQR(; kwargs...))
end

end
