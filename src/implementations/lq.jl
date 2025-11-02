# Inputs
# ------
copy_input(::typeof(lq_full), A::AbstractMatrix) = copy!(similar(A, float(eltype(A))), A)
copy_input(::typeof(lq_compact), A) = copy_input(lq_full, A)
copy_input(::typeof(lq_null), A) = copy_input(lq_full, A)

copy_input(::typeof(lq_full), A::Diagonal) = copy(A)

function check_input(::typeof(lq_full!), A::AbstractMatrix, LQ, ::AbstractAlgorithm)
    m, n = size(A)
    L, Q = LQ
    @assert L isa AbstractMatrix && Q isa AbstractMatrix
    isempty(L) || @check_size(L, (m, n))
    @check_scalar(L, A)
    @check_size(Q, (n, n))
    @check_scalar(Q, A)
    return nothing
end
function check_input(::typeof(lq_compact!), A::AbstractMatrix, LQ, ::AbstractAlgorithm)
    m, n = size(A)
    minmn = min(m, n)
    L, Q = LQ
    @assert L isa AbstractMatrix && Q isa AbstractMatrix
    isempty(L) || @check_size(L, (m, minmn))
    @check_scalar(L, A)
    @check_size(Q, (minmn, n))
    @check_scalar(Q, A)
    return nothing
end
function check_input(::typeof(lq_null!), A::AbstractMatrix, Nᴴ, ::AbstractAlgorithm)
    m, n = size(A)
    minmn = min(m, n)
    @assert Nᴴ isa AbstractMatrix
    @check_size(Nᴴ, (n - minmn, n))
    @check_scalar(Nᴴ, A)
    return nothing
end

function check_input(::typeof(lq_full!), A::AbstractMatrix, (L, Q), ::DiagonalAlgorithm)
    m, n = size(A)
    @assert m == n && isdiag(A)
    @assert Q isa Diagonal && L isa Diagonal
    isempty(L) || @check_size(L, (m, n))
    @check_scalar(L, A)
    @check_size(Q, (n, n))
    @check_scalar(Q, A)
    return nothing
end
function check_input(::typeof(lq_compact!), A::AbstractMatrix, LQ, alg::DiagonalAlgorithm)
    return check_input(lq_full!, A, LQ, alg)
end
function check_input(::typeof(lq_null!), A::AbstractMatrix, N, ::DiagonalAlgorithm)
    m, n = size(A)
    @assert m == n && isdiag(A)
    @assert N isa AbstractMatrix
    @check_size(N, (0, m))
    @check_scalar(N, A)
    return nothing
end

# Outputs
# -------
function initialize_output(::typeof(lq_full!), A::AbstractMatrix, ::AbstractAlgorithm)
    m, n = size(A)
    L = similar(A, (m, n))
    Q = similar(A, (n, n))
    return (L, Q)
end
function initialize_output(::typeof(lq_compact!), A::AbstractMatrix, ::AbstractAlgorithm)
    m, n = size(A)
    minmn = min(m, n)
    L = similar(A, (m, minmn))
    Q = similar(A, (minmn, n))
    return (L, Q)
end
function initialize_output(::typeof(lq_null!), A::AbstractMatrix, ::AbstractAlgorithm)
    m, n = size(A)
    minmn = min(m, n)
    Nᴴ = similar(A, (n - minmn, n))
    return Nᴴ
end

for f! in (:lq_full!, :lq_compact!)
    @eval function initialize_output(::typeof($f!), A::AbstractMatrix, ::DiagonalAlgorithm)
        return similar(A), A
    end
end

# Implementation
# --------------
function lq_full!(A::AbstractMatrix, LQ, alg::LAPACK_HouseholderLQ)
    check_input(lq_full!, A, LQ, alg)
    L, Q = LQ
    _lapack_lq!(A, L, Q; alg.kwargs...)
    return L, Q
end
function lq_compact!(A::AbstractMatrix, LQ, alg::LAPACK_HouseholderLQ)
    check_input(lq_compact!, A, LQ, alg)
    L, Q = LQ
    _lapack_lq!(A, L, Q; alg.kwargs...)
    return L, Q
end
function lq_null!(A::AbstractMatrix, Nᴴ, alg::LAPACK_HouseholderLQ)
    check_input(lq_null!, A, Nᴴ, alg)
    _lapack_lq_null!(A, Nᴴ; alg.kwargs...)
    return Nᴴ
end

function lq_full!(A::AbstractMatrix, LQ, alg::LQViaTransposedQR)
    check_input(lq_full!, A, LQ, alg)
    L, Q = LQ
    lq_via_qr!(A, L, Q, alg.qr_alg)
    return L, Q
end
function lq_compact!(A::AbstractMatrix, LQ, alg::LQViaTransposedQR)
    check_input(lq_compact!, A, LQ, alg)
    L, Q = LQ
    lq_via_qr!(A, L, Q, alg.qr_alg)
    return L, Q
end
function lq_null!(A::AbstractMatrix, Nᴴ, alg::LQViaTransposedQR)
    check_input(lq_null!, A, Nᴴ, alg)
    lq_null_via_qr!(A, Nᴴ, alg.qr_alg)
    return Nᴴ
end

function lq_full!(A::AbstractMatrix, LQ, alg::DiagonalAlgorithm)
    check_input(lq_full!, A, LQ, alg)
    L, Q = LQ
    _diagonal_lq!(A, L, Q; alg.kwargs...)
    return L, Q
end
function lq_compact!(A::AbstractMatrix, LQ, alg::DiagonalAlgorithm)
    check_input(lq_compact!, A, LQ, alg)
    L, Q = LQ
    _diagonal_lq!(A, L, Q; alg.kwargs...)
    return L, Q
end
function lq_null!(A::AbstractMatrix, N, alg::DiagonalAlgorithm)
    check_input(lq_null!, A, N, alg)
    return _diagonal_lq_null!(A, N; alg.kwargs...)
end

# LAPACK logic
# ------------
function _lapack_lq!(
        A::AbstractMatrix, L::AbstractMatrix, Q::AbstractMatrix;
        positive = false, pivoted = false,
        blocksize = ((pivoted || A === Q) ? 1 : YALAPACK.default_qr_blocksize(A))
    )
    m, n = size(A)
    minmn = min(m, n)
    computeL = length(L) > 0
    inplaceQ = Q === A

    if pivoted && (blocksize > 1)
        throw(ArgumentError("LAPACK does not provide a blocked implementation for a pivoted LQ decomposition"))
    end
    if inplaceQ && (computeL || positive || blocksize > 1 || n < m)
        throw(ArgumentError("inplace Q only supported if matrix is wide (`m <= n`), L is not required, and using the unblocked algorithm (`blocksize=1`) with `positive=false`"))
    end

    if blocksize > 1
        mb = min(minmn, blocksize)
        if computeL # first use L as space for T
            A, T = YALAPACK.gelqt!(A, view(L, 1:mb, 1:minmn))
        else
            A, T = YALAPACK.gelqt!(A, similar(A, mb, minmn))
        end
        Q = YALAPACK.gemlqt!('R', 'N', A, T, one!(Q))
    else
        A, τ = YALAPACK.gelqf!(A)
        if inplaceQ
            Q = YALAPACK.unglq!(A, τ)
        else
            Q = YALAPACK.unmlq!('R', 'N', A, τ, one!(Q))
        end
    end

    if positive # already fix Q even if we do not need R
        @inbounds for j in 1:n
            @simd for i in 1:minmn
                s = sign_safe(A[i, i])
                Q[i, j] *= s
            end
        end
    end

    if computeL
        L̃ = lowertriangular!(view(A, axes(L)...))
        if positive
            @inbounds for j in 1:minmn
                s = conj(sign_safe(L̃[j, j]))
                @simd for i in j:m
                    L̃[i, j] = L̃[i, j] * s
                end
            end
        end
        copyto!(L, L̃)
    end
    return L, Q
end

function _lapack_lq_null!(
        A::AbstractMatrix, Nᴴ::AbstractMatrix;
        positive = false, pivoted = false, blocksize = YALAPACK.default_qr_blocksize(A)
    )
    m, n = size(A)
    minmn = min(m, n)
    fill!(Nᴴ, zero(eltype(Nᴴ)))
    one!(view(Nᴴ, 1:(n - minmn), (minmn + 1):n))
    if blocksize > 1
        mb = min(minmn, blocksize)
        A, T = YALAPACK.gelqt!(A, similar(A, mb, minmn))
        Nᴴ = YALAPACK.gemlqt!('R', 'N', A, T, Nᴴ)
    else
        A, τ = YALAPACK.gelqf!(A)
        Nᴴ = YALAPACK.unmlq!('R', 'N', A, τ, Nᴴ)
    end
    return Nᴴ
end

# LQ via transposition and QR
# ---------------------------
function lq_via_qr!(
        A::AbstractMatrix, L::AbstractMatrix, Q::AbstractMatrix, qr_alg::AbstractAlgorithm
    )
    m, n = size(A)
    minmn = min(m, n)
    At = adjoint!(similar(A'), A)::AbstractMatrix
    Qt = (A === Q) ? At : similar(Q')
    Lt = similar(L')
    if size(Q) == (n, n)
        Qt, Lt = qr_full!(At, (Qt, Lt), qr_alg)
    else
        Qt, Lt = qr_compact!(At, (Qt, Lt), qr_alg)
    end
    adjoint!(Q, Qt)
    !isempty(L) && adjoint!(L, Lt)
    return L, Q
end

function lq_null_via_qr!(A::AbstractMatrix, N::AbstractMatrix, qr_alg::AbstractAlgorithm)
    m, n = size(A)
    minmn = min(m, n)
    At = adjoint!(similar(A'), A)::AbstractMatrix
    Nt = similar(N')
    Nt = qr_null!(At, Nt, qr_alg)
    !isempty(N) && adjoint!(N, Nt)
    return N
end

# Diagonal logic
# --------------
function _diagonal_lq!(
        A::AbstractMatrix, L::AbstractMatrix, Q::AbstractMatrix; positive::Bool = false
    )
    # note: Ad and Qd might share memory here so order of operations is important
    Ad = diagview(A)
    Ld = diagview(L)
    Qd = diagview(Q)
    if positive
        @. Ld = abs(Ad)
        @. Qd = sign_safe(Ad)
    else
        Ld .= Ad
        one!(Q)
    end
    return L, Q
end

_diagonal_lq_null!(A::AbstractMatrix, N; positive::Bool = false) = N

# Native logic
# -------------
function lq_full!(A::AbstractMatrix, LQ, alg::Native_HouseholderLQ)
    check_input(lq_full!, A, LQ, alg)
    L, Q = LQ
    A === Q &&
        throw(ArgumentError("inplace Q not supported with native LQ implementation"))
    _native_lq!(A, L, Q; alg.kwargs...)
    return L, Q
end
function lq_compact!(A::AbstractMatrix, LQ, alg::Native_HouseholderLQ)
    check_input(lq_compact!, A, LQ, alg)
    L, Q = LQ
    A === Q &&
        throw(ArgumentError("inplace Q not supported with native LQ implementation"))
    _native_lq!(A, L, Q; alg.kwargs...)
    return L, Q
end
function lq_null!(A::AbstractMatrix, N, alg::Native_HouseholderLQ)
    check_input(lq_null!, A, N, alg)
    _native_lq_null!(A, N; alg.kwargs...)
    return N
end

function _native_lq!(
        A::AbstractMatrix, L::AbstractMatrix, Q::AbstractMatrix;
        positive::Bool = true # always true regardless of setting
    )
    m, n = size(A)
    minmn = min(m, n)
    @inbounds for i in 1:minmn
        for j in 1:(i - 1)
            L[i, j] = A[i, j]
        end
        β, v, L[i, i] = _householder!(conj!(view(A, i, i:n)), 1)
        for j in (i + 1):size(L, 2)
            L[i, j] = 0
        end
        H = Householder(conj(β), v, i:n)
        rmul!(A, H; rows = (i + 1):m)
        # A[i, i] == 1; store β instead
        A[i, i] = β
    end
    # copy remaining rows for m > n
    @inbounds for j in 1:size(L, 2)
        for i in (minmn + 1):m
            L[i, j] = A[i, j]
        end
    end
    # build Q
    one!(Q)
    @inbounds for i in minmn:-1:1
        β = A[i, i]
        A[i, i] = 1
        Hᴴ = Householder(β, view(A, i, i:n), i:n)
        rmul!(Q, Hᴴ)
    end
    return L, Q
end

function _native_lq_null!(A::AbstractMatrix, Nᴴ::AbstractMatrix; positive::Bool = true)
    m, n = size(A)
    minmn = min(m, n)
    @inbounds for i in 1:minmn
        β, v, ν = _householder!(conj!(view(A, i, i:n)), 1)
        H = Householder(conj(β), v, i:n)
        rmul!(A, H; rows = (i + 1):m)
        # A[i, i] == 1; store β instead
        A[i, i] = β
    end
    # build Nᴴ
    fill!(Nᴴ, zero(eltype(Nᴴ)))
    one!(view(Nᴴ, 1:(n - minmn), (minmn + 1):n))
    @inbounds for i in minmn:-1:1
        β = A[i, i]
        A[i, i] = 1
        Hᴴ = Householder(β, view(A, i, i:n), i:n)
        rmul!(Nᴴ, Hᴴ)
    end
    return Nᴴ
end
