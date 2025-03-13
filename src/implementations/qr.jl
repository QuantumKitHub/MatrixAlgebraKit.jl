# Inputs
# ------
function copy_input(::typeof(qr_full), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
function copy_input(::typeof(qr_compact), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
function copy_input(::typeof(qr_null), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end

function check_input(::typeof(qr_full!), A::AbstractMatrix, QR)
    m, n = size(A)
    Q, R = QR
    (Q isa AbstractMatrix && eltype(Q) == eltype(A) && size(Q) == (m, m)) ||
        throw(DimensionMismatch("Full unitary matrix Q must be square with equal number of rows as A"))
    (R isa AbstractMatrix && eltype(R) == eltype(A) && (isempty(R) || size(R) == (m, n))) ||
        throw(DimensionMismatch("Upper triangular matrix R must have size equal to A"))
    return nothing
end
function check_input(::typeof(qr_compact!), A::AbstractMatrix, QR)
    m, n = size(A)
    if n <= m
        Q, R = QR
        (Q isa AbstractMatrix && eltype(Q) == eltype(A) && size(Q) == (m, n)) ||
            throw(DimensionMismatch("Isometric Q must have size equal to A"))
        (R isa AbstractMatrix && eltype(R) == eltype(A) &&
         (isempty(R) || size(R) == (n, n))) ||
            throw(DimensionMismatch("Upper triangular matrix R must be square with equal number of columns as A"))
    else
        check_input(qr_full!, A, QR)
    end
end
function check_input(::typeof(qr_null!), A::AbstractMatrix, N)
    m, n = size(A)
    minmn = min(m, n)
    (N isa AbstractMatrix && eltype(N) == eltype(A) && size(N) == (m, m - minmn)) ||
        throw(DimensionMismatch("Matrix N must have a the same eltype as A and a size such that [A N] is square"))
    return nothing
end

# Outputs
# -------
function initialize_output(::typeof(qr_full!), A::AbstractMatrix, ::LAPACK_HouseholderQR)
    m, n = size(A)
    Q = similar(A, (m, m))
    R = similar(A, (m, n))
    return (Q, R)
end
function initialize_output(::typeof(qr_compact!), A::AbstractMatrix, ::LAPACK_HouseholderQR)
    m, n = size(A)
    minmn = min(m, n)
    Q = similar(A, (m, minmn))
    R = similar(A, (minmn, n))
    return (Q, R)
end
function initialize_output(::typeof(qr_null!), A::AbstractMatrix, ::LAPACK_HouseholderQR)
    m, n = size(A)
    minmn = min(m, n)
    N = similar(A, (m, m - minmn))
    return N
end

# Implementation
# --------------
# actual implementation
function qr_full!(A::AbstractMatrix, QR, alg::LAPACK_HouseholderQR)
    check_input(qr_full!, A, QR)
    Q, R = QR
    _lapack_qr!(A, Q, R; alg.kwargs...)
    return Q, R
end
function qr_compact!(A::AbstractMatrix, QR, alg::LAPACK_HouseholderQR)
    check_input(qr_compact!, A, QR)
    Q, R = QR
    _lapack_qr!(A, Q, R; alg.kwargs...)
    return Q, R
end
function qr_null!(A::AbstractMatrix, N, alg::LAPACK_HouseholderQR)
    check_input(qr_null!, A, N)
    _lapack_qr_null!(A, N; alg.kwargs...)
    return N
end

function _lapack_qr!(A::AbstractMatrix, Q::AbstractMatrix, R::AbstractMatrix;
                     positive=false,
                     pivoted=false,
                     blocksize=((pivoted || A === Q) ? 1 : YALAPACK.default_qr_blocksize(A)))
    m, n = size(A)
    minmn = min(m, n)
    computeR = length(R) > 0
    inplaceQ = Q === A

    if pivoted && (blocksize > 1)
        throw(ArgumentError("LAPACK does not provide a blocked implementation for a pivoted QR decomposition"))
    end
    if inplaceQ && (computeR || positive || blocksize > 1 || m < n)
        throw(ArgumentError("inplace Q only supported if matrix is tall (`m >= n`), R is not required, and using the unblocked algorithm (`blocksize=1`) with `positive=false`"))
    end

    if blocksize > 1
        nb = min(minmn, blocksize)
        if computeR # first use R as space for T
            A, T = YALAPACK.geqrt!(A, view(R, 1:nb, 1:minmn))
        else
            A, T = YALAPACK.geqrt!(A, similar(A, nb, minmn))
        end
        Q = YALAPACK.gemqrt!('L', 'N', A, T, one!(Q))
    else
        if pivoted
            A, τ, jpvt = YALAPACK.geqp3!(A)
        else
            A, τ = YALAPACK.geqrf!(A)
        end
        if inplaceQ
            Q = YALAPACK.ungqr!(A, τ)
        else
            Q = YALAPACK.unmqr!('L', 'N', A, τ, one!(Q))
        end
    end

    if positive # already fix Q even if we do not need R
        @inbounds for j in 1:minmn
            s = sign_safe(A[j, j])
            @simd for i in 1:m
                Q[i, j] *= s
            end
        end
    end

    if computeR
        R̃ = triu!(view(A, axes(R)...))
        if positive
            @inbounds for j in n:-1:1
                @simd for i in 1:min(minmn, j)
                    R̃[i, j] = R̃[i, j] * conj(sign_safe(R̃[i, i]))
                end
            end
        end
        if !pivoted
            copyto!(R, R̃)
        else
            # probably very inefficient in terms of memory access
            copyto!(view(R, :, jpvt), R̃)
        end
    end
    return Q, R
end

function _lapack_qr_null!(A::AbstractMatrix, N::AbstractMatrix;
                          positive=false,
                          pivoted=false,
                          blocksize=YALAPACK.default_qr_blocksize(A))
    m, n = size(A)
    minmn = min(m, n)
    fill!(N, zero(eltype(N)))
    one!(view(N, (minmn + 1):m, 1:(m - minmn)))
    if blocksize > 1
        nb = min(minmn, blocksize)
        A, T = YALAPACK.geqrt!(A, similar(A, nb, minmn))
        N = YALAPACK.gemqrt!('L', 'N', A, T, N)
    else
        A, τ = YALAPACK.geqrf!(A)
        N = YALAPACK.unmqr!('L', 'N', A, τ, N)
    end
    return N
end
