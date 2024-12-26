function qr_full!(A::AbstractMatrix, QR=qr_full_init(A); kwargs...)
    return qr_full!(A, QR, default_algorithm(qr_full!, A; kwargs...))
end
function qr_compact!(A::AbstractMatrix, QR=qr_compact_init(A); kwargs...)
    return qr_compact!(A, QR, default_algorithm(qr_compact!, A; kwargs...))
end

function qr_full_init(A::AbstractMatrix)
    m, n = size(A)
    Q = similar(A, (m, m))
    R = similar(A, (m, n))
    return (Q, R)
end
function qr_compact_init(A::AbstractMatrix)
    m, n = size(A)
    minmn = min(m, n)
    Q = similar(A, (m, minmn))
    R = similar(A, (minmn, n))
    return (Q, R)
end

function default_algorithm(::typeof(qr_full!), A::AbstractMatrix; kwargs...)
    return default_qr_algorithm(A; kwargs...)
end
function default_algorithm(::typeof(qr_compact!), A::AbstractMatrix; kwargs...)
    return default_qr_algorithm(A; kwargs...)
end

function default_qr_algorithm(A::StridedMatrix{T}; kwargs...) where {T<:BlasFloat}
    return LAPACK_HouseholderQR(; kwargs...)
end

function check_qr_full_input(A::AbstractMatrix, QR)
    m, n = size(A)
    Q, R = QR
    (Q isa AbstractMatrix && eltype(Q) == eltype(A) && size(Q) == (m, m)) ||
        throw(DimensionMismatch("Full unitary matrix `Q` must be square with equal number of rows as A"))
    (R isa AbstractMatrix && eltype(R) == eltype(A) && (isempty(R) || size(R) == (m, n))) ||
        throw(DimensionMismatch("Upper triangular matrix `R` must have size equal to A"))
    return nothing
end
function check_qr_compact_input(A::AbstractMatrix, QR)
    m, n = size(A)
    if n <= m
        Q, R = QR
        (Q isa AbstractMatrix && eltype(Q) == eltype(A) && size(Q) == (m, n)) ||
            throw(DimensionMismatch("Isometric `Q` must have size equal to A"))
        (R isa AbstractMatrix && eltype(R) == eltype(A) &&
         (isempty(R) || size(R) == (n, n))) ||
            throw(DimensionMismatch("Upper triangular matrix `R` must be square with equal number of columns as A"))
    else
        check_qr_full_input(A, QR)
    end
end

function qr_full!(A::AbstractMatrix, QR, alg::LAPACK_HouseholderQR)
    check_qr_full_input(A, QR)
    Q, R = QR
    _lapack_qr!(A, Q, R; alg.kwargs...)
    return Q, R
end
function qr_compact!(A::AbstractMatrix, QR, alg::LAPACK_HouseholderQR)
    check_qr_compact_input(A, QR)
    Q, R = QR
    _lapack_qr!(A, Q, R; alg.kwargs...)
    return Q, R
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
            Q = YALAPACK.orgqr!(A, τ)
        else
            Q = YALAPACK.ormqr!('L', 'N', A, τ, one!(Q))
        end
    end

    if positive # already fix Q even if we do not need R
        @inbounds for j in 1:minmn
            s = safesign(A[j, j])
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
                    R̃[i, j] = R̃[i, j] * conj(safesign(R̃[i, i]))
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
