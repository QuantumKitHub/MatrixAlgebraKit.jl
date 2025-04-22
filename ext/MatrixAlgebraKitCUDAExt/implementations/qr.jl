"""
    CUSOLVER_HouseholderQR(; positive = false)

Algorithm type to denote the standard CUSOLVER algorithm for computing the QR decomposition of
a matrix using Householder reflectors. The keyword `positive=true` can be used to ensure that
the diagonal elements of `R` are non-negative.
"""
@algdef CUSOLVER_HouseholderQR

function MatrixAlgebraKit.default_qr_algorithm(A::CuMatrix{<:BlasFloat}; kwargs...)
    return CUSOLVER_HouseholderQR(; kwargs...)
end

# Implementation
# --------------
# actual implementation
function MatrixAlgebraKit.qr_full!(A::AbstractMatrix, QR, alg::CUSOLVER_HouseholderQR)
    check_input(qr_full!, A, QR)
    Q, R = QR
    _cusolver_qr!(A, Q, R; alg.kwargs...)
    return Q, R
end
function MatrixAlgebraKit.qr_compact!(A::AbstractMatrix, QR, alg::CUSOLVER_HouseholderQR)
    check_input(qr_compact!, A, QR)
    Q, R = QR
    _cusolver_qr!(A, Q, R; alg.kwargs...)
    return Q, R
end
function MatrixAlgebraKit.qr_null!(A::AbstractMatrix, N, alg::CUSOLVER_HouseholderQR)
    check_input(qr_null!, A, N)
    _cusolver_qr_null!(A, N; alg.kwargs...)
    return N
end

function _cusolver_qr!(A::AbstractMatrix, Q::AbstractMatrix, R::AbstractMatrix;
                       positive=false, blocksize=1)
    blocksize > 1 &&
        throw(ArgumentError("CUSOLVER does not provide a blocked implementation for a QR decomposition"))
    m, n = size(A)
    minmn = min(m, n)
    computeR = length(R) > 0
    inplaceQ = Q === A
    if inplaceQ && (computeR || positive || m < n)
        throw(ArgumentError("inplace Q only supported if matrix is tall (`m >= n`), R is not required and using `positive=false`"))
    end

    A, τ = YACUSOLVER.geqrf!(A)
    if inplaceQ
        Q = YACUSOLVER.ungqr!(A, τ)
    else
        Q = YACUSOLVER.unmqr!('L', 'N', A, τ, one!(Q))
    end
    # henceforth, τ is no longer needed and can be reused

    if positive # already fix Q even if we do not need R
        # TODO: report that `lmul!` and `rmul!` with `Diagonal` don't work with CUDA
        τ .= sign_safe.(diagview(A))
        Qf = view(Q, 1:m, 1:minmn) # first minmn columns of Q
        Qf .= Qf .* transpose(τ)
    end

    if computeR
        R̃ = uppertriangular!(view(A, axes(R)...))
        if positive
            R̃f = view(R̃, 1:minmn, 1:n) # first minmn rows of R
            R̃f .= conj.(τ) .* R̃f
        end
        copyto!(R, R̃)
    end
    return Q, R
end

function _cusolver_qr_null!(A::AbstractMatrix, N::AbstractMatrix;
                            positive=false, blocksize=1)
    blocksize > 1 &&
        throw(ArgumentError("CUSOLVER does not provide a blocked implementation for a QR decomposition"))
    m, n = size(A)
    minmn = min(m, n)
    fill!(N, zero(eltype(N)))
    one!(view(N, (minmn + 1):m, 1:(m - minmn)))
    A, τ = YACUSOLVER.geqrf!(A)
    N = YACUSOLVER.unmqr!('L', 'N', A, τ, N)
    return N
end
