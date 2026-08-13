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

# DefaultAlgorithm intercepts
# ---------------------------
for f! in (:qr_full!, :qr_compact!, :qr_null!)
    @eval function $f!(A::AbstractMatrix, alg::DefaultAlgorithm)
        return $f!(A, select_algorithm($f!, A, nothing; alg.kwargs...))
    end
    @eval function $f!(A::AbstractMatrix, out, alg::DefaultAlgorithm)
        return $f!(A, out, select_algorithm($f!, A, nothing; alg.kwargs...))
    end
end

# ==========================
#      IMPLEMENTATIONS
# ==========================

# Householder
# -----------
function qr_full!(A::AbstractMatrix, QR, alg::Householder)
    check_input(qr_full!, A, QR, alg)
    return qr_householder!(A, QR...; alg.kwargs...)
end
function qr_compact!(A::AbstractMatrix, QR, alg::Householder)
    check_input(qr_compact!, A, QR, alg)
    return qr_householder!(A, QR...; alg.kwargs...)
end
function qr_null!(A::AbstractMatrix, N, alg::Householder)
    check_input(qr_null!, A, N, alg)
    return qr_null_householder!(A, N; alg.kwargs...)
end


# dispatch helpers
for f in (:geqrt!, :gemqrt!, :geqp3!, :geqrf!, :ungqr!, :unmqr!)
    @eval begin
        $f(driver::Driver, args...) = throw(MethodError($f, (driver, args...))) # make JET not complain
        $f(::LAPACK, args...) = YALAPACK.$f(args...)
    end
end

# cuSOLVER generates Q faster with ungqr! than by applying the reflectors with unmqr!,
# and avoids the large workspace of ormqr, whose 32-bit size query fails for large matrices
prefers_ungqr(::Driver) = false

# copy R out of the packed factorization, leaving the reflectors in A intact
function _qr_copyR!(R::AbstractMatrix, A::AbstractMatrix, jpvt = nothing)
    Rp = isnothing(jpvt) ? R : view(R, :, jpvt)
    copyto!(Rp, view(A, axes(R)...))
    uppertriangular!(Rp)
    return R
end

function _qr_buildQ!(driver::Driver, Q::AbstractMatrix, A::AbstractMatrix, τ, minmn::Int)
    if prefers_ungqr(driver)
        # build Q in its own space: copy in the reflectors, unit vectors elsewhere
        size(Q, 2) > minmn && one!(Q)
        copyto!(view(Q, :, 1:minmn), view(A, :, 1:minmn))
        ungqr!(driver, Q, τ)
    else
        Q = unmqr!(driver, 'L', 'N', A, τ, one!(Q))
    end
    return Q
end

@inline qr_householder!(A, Q, R; driver::Driver = DefaultDriver(), kwargs...) =
    qr_householder!(driver, A, Q, R; kwargs...)
qr_householder!(::DefaultDriver, A, Q, R; kwargs...) =
    qr_householder!(default_driver(Householder, A), A, Q, R; kwargs...)
function qr_householder!(
        driver::Union{LAPACK, CUSOLVER, ROCSOLVER}, A::AbstractMatrix, Q::AbstractMatrix, R::AbstractMatrix;
        positive::Bool = true, pivoted::Bool = false,
        blocksize::Int = 0
    )
    blocksize = blocksize > 0 ? blocksize : ((driver !== LAPACK() || pivoted || A === Q) ? 1 : YALAPACK.default_qr_blocksize(A))

    # error messages for disallowing driver - setting combinations
    (blocksize == 1 || driver === LAPACK()) ||
        throw(ArgumentError(lazy"$driver does not provide a blocked QR decomposition"))
    (!pivoted || driver === LAPACK()) ||
        throw(ArgumentError(lazy"$driver does not provide a pivoted QR decomposition"))
    pivoted && (blocksize > 1) &&
        throw(ArgumentError(lazy"$driver does not provide a blocked pivoted QR decomposition"))

    m, n = size(A)
    minmn = min(m, n)
    computeR = length(R) > 0
    inplaceQ = Q === A

    if inplaceQ
        # ungqr! builds Q in the space of A, so R has to be extracted first and cannot alias A
        (blocksize == 1 && m >= n) ||
            throw(ArgumentError(lazy"in-place Q is only supported if matrix is tall (`$m >= $n`) and using the unblocked algorithm (`blocksize = $blocksize`)"))
        (computeR && Base.mightalias(R, A)) &&
            throw(ArgumentError("in-place Q is only supported if R does not share memory with A"))
    end

    # Compute QR in packed form
    if blocksize > 1
        # R doubles as workspace for T, so Q is constructed before R is extracted
        nb = min(minmn, blocksize)
        if computeR # first use R as space for T
            A, T = geqrt!(driver, A, view(R, 1:nb, 1:minmn))
        else
            A, T = geqrt!(driver, A, similar(A, nb, minmn))
        end
        Q = gemqrt!(driver, 'L', 'N', A, T, one!(Q))
        computeR && _qr_copyR!(R, A)
        positive && gaugefix!(qr_householder!, Q, computeR ? R : nothing, diagview(A))
    else
        if pivoted
            A, τ, jpvt = geqp3!(driver, A)
            computeR && _qr_copyR!(R, A, jpvt)
        else
            A, τ = geqrf!(driver, A)
            computeR && _qr_copyR!(R, A)
        end
        Rf = computeR ? R : nothing # gaugefix! rescales rows, which commutes with the pivoting
        if inplaceQ
            Rd = positive ? copy(diagview(A)) : nothing # ungqr! destroys the diagonal of A
            ungqr!(driver, A, τ) # Q === A, so no need to rebind Q
            positive && gaugefix!(qr_householder!, Q, Rf, Rd)
        else
            Q = _qr_buildQ!(driver, Q, A, τ, minmn)
            positive && gaugefix!(qr_householder!, Q, Rf, diagview(A))
        end
    end

    return Q, R
end
function qr_householder!(
        driver::Native, A::AbstractMatrix, Q::AbstractMatrix, R::AbstractMatrix;
        positive::Bool = true, pivoted::Bool = false, blocksize::Int = 0
    )
    # error messages for disallowing driver - setting combinations
    blocksize <= 1 ||
        throw(ArgumentError(lazy"$driver does not provide a blocked QR decomposition"))
    pivoted &&
        throw(ArgumentError(lazy"$driver does not provide a pivoted QR decomposition"))
    Q === A &&
        throw(ArgumentError(lazy"$driver does not provide an in-place Q"))
    # positive = true regardless of setting

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
        H = HouseholderReflection(β, v, j:m)
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
        Hᴴ = HouseholderReflection(conj(β), view(A, j:m, j), j:m)
        lmul!(Hᴴ, Q)
    end
    return Q, R
end

@inline qr_null_householder!(A, N; driver::Driver = DefaultDriver(), kwargs...) =
    qr_null_householder!(driver, A, N; kwargs...)
qr_null_householder!(::DefaultDriver, A, N; kwargs...) =
    qr_null_householder!(default_driver(Householder, A), A, N; kwargs...)
function qr_null_householder!(
        driver::Union{LAPACK, CUSOLVER, ROCSOLVER}, A::AbstractMatrix, N::AbstractMatrix;
        positive::Bool = true, pivoted::Bool = false, blocksize::Int = 0
    )
    blocksize = blocksize > 0 ? blocksize : ((driver !== LAPACK() || pivoted) ? 1 : YALAPACK.default_qr_blocksize(A))
    # error messages for disallowing driver - setting combinations
    (blocksize == 1 || driver === LAPACK()) ||
        throw(ArgumentError(lazy"$driver does not provide a blocked QR decomposition"))
    (!pivoted || driver === LAPACK()) ||
        throw(ArgumentError(lazy"$driver does not provide a pivoted QR decomposition"))
    pivoted && (blocksize > 1) &&
        throw(ArgumentError(lazy"$driver does not provide a blocked pivoted QR decomposition"))

    m, n = size(A)
    minmn = min(m, n)
    zero!(N)
    one!(view(N, (minmn + 1):m, 1:(m - minmn)))

    if blocksize > 1
        nb = min(minmn, blocksize)
        A, T = geqrt!(driver, A, similar(A, nb, minmn))
        N = gemqrt!(driver, 'L', 'N', A, T, N)
    else
        A, τ = geqrf!(driver, A)
        N = unmqr!(driver, 'L', 'N', A, τ, N)
    end
    return N
end
function qr_null_householder!(
        driver::Native, A::AbstractMatrix, N::AbstractMatrix;
        positive::Bool = true, pivoted::Bool = false, blocksize::Int = 0
    )
    # error messages for disallowing driver - setting combinations
    blocksize <= 1 ||
        throw(ArgumentError(lazy"$driver does not provide a blocked QR decomposition"))
    pivoted &&
        throw(ArgumentError(lazy"$driver does not provide a pivoted QR decomposition"))

    m, n = size(A)
    minmn = min(m, n)

    @inbounds for j in 1:minmn
        β, v, ν = _householder!(view(A, j:m, j), 1)
        H = HouseholderReflection(β, v, j:m)
        lmul!(H, A; cols = (j + 1):n)
        # A[j, j] == 1; store β instead
        A[j, j] = β
    end

    # build N
    zero!(N)
    one!(view(N, (minmn + 1):m, 1:(m - minmn)))
    @inbounds for j in minmn:-1:1
        β = A[j, j]
        A[j, j] = 1
        Hᴴ = HouseholderReflection(conj(β), view(A, j:m, j), j:m)
        lmul!(Hᴴ, N)
    end
    return N
end


# Diagonal
# --------
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

function _diagonal_qr!(
        A::AbstractMatrix, Q::AbstractMatrix, R::AbstractMatrix; positive::Bool = true
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

_diagonal_qr_null!(A::AbstractMatrix, N; positive::Bool = true) = N

# Deprecations
# ------------
for drivertype in (:LAPACK, :CUSOLVER, :ROCSOLVER, :Native, :GLA)
    algtype = Symbol(drivertype, :_HouseholderQR)
    @eval begin
        Base.@deprecate(
            qr_full!(A::AbstractMatrix, QR, alg::$algtype),
            qr_full!(A, QR, Householder(; driver = $drivertype(), alg.kwargs...))
        )
        Base.@deprecate(
            qr_compact!(A::AbstractMatrix, QR, alg::$algtype),
            qr_compact!(A, QR, Householder(; driver = $drivertype(), alg.kwargs...))
        )
        Base.@deprecate(
            qr_null!(A::AbstractMatrix, N, alg::$algtype),
            qr_null!(A, N, Householder(; driver = $drivertype(), alg.kwargs...))
        )
    end
end
