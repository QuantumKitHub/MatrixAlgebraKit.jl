# Inputs
# ------
copy_input(::typeof(qr_full), A::AbstractMatrix) = copy!(similar(A, float(eltype(A))), A)
copy_input(::typeof(qr_compact), A) = copy_input(qr_full, A)
copy_input(::typeof(qr_null), A) = copy_input(qr_full, A)

copy_input(::typeof(qr_full), A::Diagonal) = copy(A)

function check_input(::typeof(qr_full!), A::AbstractMatrix, QR, ::AbstractAlgorithm)
    m, n = size(A)
    Q, R = QR
    @assert Q isa AbstractMatrix && R isa AbstractMatrix
    @check_size(Q, (m, m))
    @check_scalar(Q, A)
    isempty(R) || @check_size(R, (m, n))
    @check_scalar(R, A)
    return nothing
end
function check_input(::typeof(qr_compact!), A::AbstractMatrix, QR, ::AbstractAlgorithm)
    m, n = size(A)
    minmn = min(m, n)
    Q, R = QR
    @assert Q isa AbstractMatrix && R isa AbstractMatrix
    @check_size(Q, (m, minmn))
    @check_scalar(Q, A)
    isempty(R) || @check_size(R, (minmn, n))
    @check_scalar(R, A)
    return nothing
end
function check_input(::typeof(qr_null!), A::AbstractMatrix, N, ::AbstractAlgorithm)
    m, n = size(A)
    minmn = min(m, n)
    @assert N isa AbstractMatrix
    @check_size(N, (m, m - minmn))
    @check_scalar(N, A)
    return nothing
end

function check_input(::typeof(qr_full!), A::AbstractMatrix, (Q, R), alg::DiagonalAlgorithm)
    m, n = size(A)
    @assert m == n && isdiag(A)
    @assert Q isa Diagonal && R isa Diagonal
    @check_size(Q, (m, n))
    @check_scalar(Q, A)
    isempty(R) || @check_size(R, (m, n))
    @check_scalar(R, A)
    return nothing
end
function check_input(::typeof(qr_compact!), A::AbstractMatrix, QR, alg::DiagonalAlgorithm)
    return check_input(qr_full!, A, QR, alg)
end
function check_input(::typeof(qr_null!), A::AbstractMatrix, N, ::DiagonalAlgorithm)
    m, n = size(A)
    @assert m == n && isdiag(A)
    @assert N isa AbstractMatrix
    @check_size(N, (m, 0))
    @check_scalar(N, A)
    return nothing
end

# Outputs
# -------
function initialize_output(::typeof(qr_full!), A::AbstractMatrix, ::AbstractAlgorithm)
    m, n = size(A)
    Q = similar(A, (m, m))
    R = similar(A, (m, n))
    return (Q, R)
end
function initialize_output(::typeof(qr_compact!), A::AbstractMatrix, ::AbstractAlgorithm)
    m, n = size(A)
    minmn = min(m, n)
    Q = similar(A, (m, minmn))
    R = similar(A, (minmn, n))
    return (Q, R)
end
function initialize_output(::typeof(qr_null!), A::AbstractMatrix, ::AbstractAlgorithm)
    m, n = size(A)
    minmn = min(m, n)
    N = similar(A, (m, m - minmn))
    return N
end

for f! in (:qr_full!, :qr_compact!)
    @eval function initialize_output(::typeof($f!), A::AbstractMatrix, ::DiagonalAlgorithm)
        return A, similar(A)
    end
end

# Implementation
# --------------
# actual implementation
function qr_full!(A::AbstractMatrix, QR, alg::LAPACK_HouseholderQR)
    check_input(qr_full!, A, QR, alg)
    Q, R = QR
    _lapack_qr!(A, Q, R; alg.kwargs...)
    return Q, R
end
function qr_compact!(A::AbstractMatrix, QR, alg::LAPACK_HouseholderQR)
    check_input(qr_compact!, A, QR, alg)
    Q, R = QR
    _lapack_qr!(A, Q, R; alg.kwargs...)
    return Q, R
end
function qr_null!(A::AbstractMatrix, N, alg::LAPACK_HouseholderQR)
    check_input(qr_null!, A, N, alg)
    _lapack_qr_null!(A, N; alg.kwargs...)
    return N
end

function qr_full!(A::AbstractMatrix, QR, alg::DiagonalAlgorithm)
    check_input(qr_full!, A, QR, alg)
    Q, R = QR
    _diagonal_qr!(A, Q, R; alg.kwargs...)
    return Q, R
end
function qr_compact!(A::AbstractMatrix, QR, alg::DiagonalAlgorithm)
    check_input(qr_compact!, A, QR, alg)
    Q, R = QR
    _diagonal_qr!(A, Q, R; alg.kwargs...)
    return Q, R
end
function qr_null!(A::AbstractMatrix, N, alg::DiagonalAlgorithm)
    check_input(qr_null!, A, N, alg)
    _diagonal_qr_null!(A, N; alg.kwargs...)
    return N
end

# LAPACK logic
# ------------
function _lapack_qr!(
        A::AbstractMatrix, Q::AbstractMatrix, R::AbstractMatrix;
        positive = false, pivoted = false,
        blocksize = ((pivoted || A === Q) ? 1 : YALAPACK.default_qr_blocksize(A))
    )
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
        R̃ = uppertriangular!(view(A, axes(R)...))
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

function _lapack_qr_null!(
        A::AbstractMatrix, N::AbstractMatrix;
        positive = false, pivoted = false, blocksize = YALAPACK.default_qr_blocksize(A)
    )
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

# Diagonal logic
# --------------
function _diagonal_qr!(
        A::AbstractMatrix, Q::AbstractMatrix, R::AbstractMatrix; positive::Bool = false
    )
    # note: Ad and Qd might share memory here so order of operations is important
    Ad = diagview(A)
    Qd = diagview(Q)
    Rd = diagview(R)
    if positive
        @. Rd = abs(Ad)
        @. Qd = sign_safe(Ad)
    else
        Rd .= Ad
        one!(Q)
    end
    return Q, R
end

_diagonal_qr_null!(A::AbstractMatrix, N; positive::Bool = false) = N

# GPU logic
# --------------
# placed here to avoid code duplication since much of the logic is replicable across CUDA and AMDGPU
function MatrixAlgebraKit.qr_full!(
        A::AbstractMatrix, QR, alg::Union{CUSOLVER_HouseholderQR, ROCSOLVER_HouseholderQR}
    )
    check_input(qr_full!, A, QR, alg)
    Q, R = QR
    _gpu_qr!(A, Q, R; alg.kwargs...)
    return Q, R
end
function MatrixAlgebraKit.qr_compact!(
        A::AbstractMatrix, QR, alg::Union{CUSOLVER_HouseholderQR, ROCSOLVER_HouseholderQR}
    )
    check_input(qr_compact!, A, QR, alg)
    Q, R = QR
    _gpu_qr!(A, Q, R; alg.kwargs...)
    return Q, R
end
function MatrixAlgebraKit.qr_null!(
        A::AbstractMatrix, N, alg::Union{CUSOLVER_HouseholderQR, ROCSOLVER_HouseholderQR}
    )
    check_input(qr_null!, A, N, alg)
    _gpu_qr_null!(A, N; alg.kwargs...)
    return N
end

_gpu_geqrf!(A::AbstractMatrix) = throw(MethodError(_gpu_geqrf!, (A,)))
_gpu_ungqr!(A::AbstractMatrix, τ::AbstractVector) = throw(MethodError(_gpu_ungqr!, (A, τ)))
function _gpu_unmqr!(
        side::AbstractChar, trans::AbstractChar, A::AbstractMatrix, τ::AbstractVector, C
    )
    throw(MethodError(_gpu_unmqr!, (side, trans, A, τ, C)))
end

function _gpu_qr!(
        A::AbstractMatrix, Q::AbstractMatrix, R::AbstractMatrix; positive = false, blocksize = 1, pivoted = false
    )
    blocksize > 1 &&
        throw(ArgumentError("CUSOLVER/ROCSOLVER does not provide a blocked implementation for a QR decomposition"))
    pivoted &&
        throw(ArgumentError("CUSOLVER/ROCSOLVER does not provide a pivoted implementation for a QR decomposition"))
    m, n = size(A)
    minmn = min(m, n)
    computeR = length(R) > 0
    inplaceQ = Q === A
    if inplaceQ && (computeR || positive || m < n)
        throw(ArgumentError("inplace Q only supported if matrix is tall (`m >= n`), R is not required and using `positive=false`"))
    end

    A, τ = _gpu_geqrf!(A)
    if inplaceQ
        Q = _gpu_ungqr!(A, τ)
    else
        Q = _gpu_unmqr!('L', 'N', A, τ, one!(Q))
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

function _gpu_qr_null!(
        A::AbstractMatrix, N::AbstractMatrix; positive = false, blocksize = 1, pivoted = false
    )
    blocksize > 1 &&
        throw(ArgumentError("CUSOLVER/ROCSOLVER does not provide a blocked implementation for a QR decomposition"))
    pivoted &&
        throw(ArgumentError("CUSOLVER/ROCSOLVER does not provide a pivoted implementation for a QR decomposition"))
    m, n = size(A)
    minmn = min(m, n)
    fill!(N, zero(eltype(N)))
    one!(view(N, (minmn + 1):m, 1:(m - minmn)))
    A, τ = _gpu_geqrf!(A)
    N = _gpu_unmqr!('L', 'N', A, τ, N)
    return N
end

# Native logic
# --------------
function qr_full!(A::AbstractMatrix, QR, alg::Native_HouseholderQR)
    check_input(qr_full!, A, QR, alg)
    Q, R = QR
    A === Q &&
        throw(ArgumentError("inplace Q not supported with native QR implementation"))
    _native_qr!(A, Q, R; alg.kwargs...)
    return Q, R
end
function qr_compact!(A::AbstractMatrix, QR, alg::Native_HouseholderQR)
    check_input(qr_compact!, A, QR, alg)
    Q, R = QR
    A === Q &&
        throw(ArgumentError("inplace Q not supported with native QR implementation"))
    _native_qr!(A, Q, R; alg.kwargs...)
    return Q, R
end
function qr_null!(A::AbstractMatrix, N, alg::Native_HouseholderQR)
    check_input(qr_null!, A, N, alg)
    _native_qr_null!(A, N; alg.kwargs...)
    return N
end

function _native_qr!(
        A::AbstractMatrix, Q::AbstractMatrix, R::AbstractMatrix;
        positive::Bool = true # always true regardless of setting
    )
    m, n = size(A)
    minmn = min(m, n)
    @inbounds for j in 1:minmn
        for i in 1:(j - 1)
            R[i, j] = A[i, j]
        end
        β, v, R[j, j] = _householder!(view(A, j:m, j), 1)
        for i in (j + 1):size(R, 1)
            R[i, j] = 0
        end
        H = Householder(β, v, j:m)
        lmul!(H, A; cols = (j + 1):n)
        # A[j,j] == 1; store β instead
        A[j, j] = β
    end
    # copy remaining columns if m < n
    @inbounds for j in (minmn + 1):n
        for i in 1:size(R, 1)
            R[i, j] = A[i, j]
        end
    end
    # build Q
    one!(Q)
    @inbounds for j in minmn:-1:1
        β = A[j, j]
        A[j, j] = 1
        Hᴴ = Householder(conj(β), view(A, j:m, j), j:m)
        lmul!(Hᴴ, Q)
    end
    return Q, R
end

function _native_qr_null!(A::AbstractMatrix, N::AbstractMatrix; positive::Bool = true)
    m, n = size(A)
    minmn = min(m, n)
    @inbounds for j in 1:minmn
        β, v, ν = _householder!(view(A, j:m, j), 1)
        H = Householder(β, v, j:m)
        lmul!(H, A; cols = (j + 1):n)
        # A[j,j] == 1; store β instead
        A[j, j] = β
    end
    # build N
    fill!(N, zero(eltype(N)))
    one!(view(N, (minmn + 1):m, 1:(m - minmn)))
    @inbounds for j in minmn:-1:1
        β = A[j, j]
        A[j, j] = 1
        Hᴴ = Householder(conj(β), view(A, j:m, j), j:m)
        lmul!(Hᴴ, N)
    end
    return N
end
