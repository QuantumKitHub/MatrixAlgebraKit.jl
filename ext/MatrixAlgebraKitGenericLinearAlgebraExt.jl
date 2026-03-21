module MatrixAlgebraKitGenericLinearAlgebraExt

using MatrixAlgebraKit
using MatrixAlgebraKit: sign_safe, check_input, diagview, gaugefix!, one!, zero!, default_fixgauge
using MatrixAlgebraKit: GLA, Driver
import MatrixAlgebraKit: gesvd!, heev!
using GenericLinearAlgebra: svd!, svdvals!, eigen!, eigvals!, Hermitian, qr!
using LinearAlgebra: I, Diagonal, lmul!

const GlaFloat = Union{BigFloat, Complex{BigFloat}}
const GlaStridedVecOrMatrix{T <: GlaFloat} = Union{StridedVector{T}, StridedMatrix{T}}
MatrixAlgebraKit.default_driver(::Type{<:QRIteration}, ::Type{TA}) where {TA <: GlaStridedVecOrMatrix} = GLA()

MatrixAlgebraKit.supports_svd_full(::GLA, f::Symbol) = f === :qr_iteration

function MatrixAlgebraKit.default_svd_algorithm(
        ::Type{T};
        driver::Driver = GLA(), kwargs...
    ) where {T <: GlaStridedVecOrMatrix}
    return QRIteration(; driver, kwargs...)
end

function gesvd!(::GLA, A::AbstractMatrix, S::AbstractVector, U::AbstractMatrix, Vᴴ::AbstractMatrix; kwargs...)
    m, n = size(A)
    if length(U) == 0 && length(Vᴴ) == 0
        Sv = svdvals!(A)
        copyto!(S, Sv)
    else
        minmn = min(m, n)
        # full SVD if U has m columns or Vᴴ has n rows (beyond the compact min(m,n))
        full = (length(U) > 0 && size(U, 2) > minmn) || (length(Vᴴ) > 0 && size(Vᴴ, 1) > minmn)
        F = svd!(A; full = full)
        length(S) > 0 && copyto!(S, F.S)
        length(U) > 0 && copyto!(U, F.U)
        length(Vᴴ) > 0 && copyto!(Vᴴ, F.Vt)
    end
    return S, U, Vᴴ
end

function MatrixAlgebraKit.default_eigh_algorithm(::Type{T}; driver::Driver = GLA(), kwargs...) where {T <: GlaStridedVecOrMatrix}
    return QRIteration(; driver, kwargs...)
end

function heev!(::GLA, A::AbstractMatrix, Dd::AbstractVector, V::AbstractMatrix; kwargs...)
    if length(V) > 0
        eigval, eigvec = eigen!(Hermitian(A); sortby = real)
        copyto!(Dd, eigval)
        copyto!(V, eigvec)
    else
        eigval = eigvals!(Hermitian(A); sortby = real)
        copyto!(Dd, eigval)
    end
    return Dd, V
end

function MatrixAlgebraKit.householder_qr!(
        driver::MatrixAlgebraKit.GLA, A::AbstractMatrix, Q::AbstractMatrix, R::AbstractMatrix;
        positive::Bool = true, pivoted::Bool = false, blocksize::Int = 0
    )
    blocksize <= 1 ||
        throw(ArgumentError(lazy"$driver does not provide a blocked QR decomposition"))
    pivoted &&
        throw(ArgumentError(lazy"$driver does not provide a pivoted QR decomposition"))

    m, n = size(A)
    minmn = min(m, n)
    computeR = length(R) > 0

    # compute QR
    Q̃, R̃ = qr!(A)
    lmul!(Q̃, MatrixAlgebraKit.one!(Q))

    if positive
        @inbounds for j in 1:minmn
            s = sign_safe(R̃[j, j])
            @simd for i in 1:m
                Q[i, j] *= s
            end
        end
    end

    if computeR
        if positive
            @inbounds for j in n:-1:1
                @simd for i in 1:min(minmn, j)
                    R[i, j] = R̃[i, j] * conj(sign_safe(R̃[i, i]))
                end
                @simd for i in (min(minmn, j) + 1):size(R, 1)
                    R[i, j] = zero(eltype(R))
                end
            end
        else
            R[1:minmn, :] .= R̃
            MatrixAlgebraKit.zero!(@view(R[(minmn + 1):end, :]))
        end
    end
    return Q, R
end

function MatrixAlgebraKit.householder_qr_null!(
        driver::MatrixAlgebraKit.GLA, A::AbstractMatrix, N::AbstractMatrix;
        positive::Bool = true, pivoted::Bool = false, blocksize::Int = 0
    )
    blocksize <= 1 ||
        throw(ArgumentError(lazy"$driver does not provide a blocked QR decomposition"))
    pivoted &&
        throw(ArgumentError(lazy"$driver does not provide a pivoted QR decomposition"))

    m, n = size(A)
    minmn = min(m, n)
    zero!(N)
    one!(view(N, (minmn + 1):m, 1:(m - minmn)))
    Q̃, = qr!(A)
    return lmul!(Q̃, N)
end

MatrixAlgebraKit.left_orth_alg(alg::GLA_HouseholderQR) = MatrixAlgebraKit.LeftOrthViaQR(alg)

end
