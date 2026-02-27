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

# ==========================
#      IMPLEMENTATIONS
# ==========================

# Householder
# -----------
function qr_full!(A, QR, alg::Householder)
    check_input(qr_full!, A, QR, alg)
    return householder_qr!(alg.driver, A, QR...; alg.kwargs...)
end
function qr_compact!(A, QR, alg::Householder)
    check_input(qr_compact!, A, QR, alg)
    return householder_qr!(alg.driver, A, QR...; alg.kwargs...)
end
function qr_null!(A, N, alg::Householder)
    check_input(qr_null!, A, N, alg)
    return householder_qr_null!(alg.driver, A, N; alg.kwargs...)
end

householder_qr!(::DefaultDriver, A, Q, R; kwargs...) =
    householder_qr!(default_householder_driver(A), A, Q, R; kwargs...)
householder_qr_null!(::DefaultDriver, A, N; kwargs...) =
    householder_qr_null!(default_householder_driver(A), A, N; kwargs...)

# dispatch helpers
for f in (:geqrt!, :gemqrt!, :geqp3!, :geqrf!, :ungqr!, :unmqr!)
    @eval begin
        $f(driver::Driver, args...) = throw(MethodError($f, (driver, args...))) # make JET not complain
        $f(::LAPACK, args...) = YALAPACK.$f(args...)
    end
end

function householder_qr!(
        driver::Union{LAPACK, CUSOLVER, ROCSOLVER}, A::AbstractMatrix, Q::AbstractMatrix, R::AbstractMatrix;
        positive::Bool = true, pivoted::Bool = false,
        blocksize::Int = ((driver !== LAPACK() || pivoted || A === Q) ? 1 : YALAPACK.default_qr_blocksize(A))
    )
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

    (inplaceQ && (computeR || positive || blocksize > 1 || m < n)) &&
        throw(ArgumentError("inplace Q only supported if matrix is tall (`m >= n`), R is not required, and using the unblocked algorithm (`blocksize = 1`) with `positive = false`"))

    # Compute QR in packed form
    if blocksize > 1
        nb = min(minmn, blocksize)
        if computeR # first use R as space for T
            A, T = geqrt!(driver, A, view(R, 1:nb, 1:minmn))
        else
            A, T = geqrt!(driver, A, similar(A, nb, minmn))
        end
        Q = gemqrt!(driver, 'L', 'N', A, T, one!(Q))
    else
        if pivoted
            A, τ, jpvt = geqp3!(driver, A)
        else
            A, τ = geqrf!(driver, A)
        end
        if inplaceQ
            Q = ungqr!(driver, A, τ)
        else
            Q = unmqr!(driver, 'L', 'N', A, τ, one!(Q))
        end
    end

    if positive # already fix Q even if we do not need R
        if driver === LAPACK()
            @inbounds for j in 1:minmn
                s = sign_safe(A[j, j])
                @simd for i in 1:m
                    Q[i, j] *= s
                end
            end
        else
            # guaranteed τ exists and no longer needed
            τ .= sign_safe.(diagview(A))
            Qf = view(Q, 1:m, 1:minmn) # first minmn columns of Q
            Qf .= Qf .* transpose(τ)
        end
    end

    if computeR
        R̃ = uppertriangular!(view(A, axes(R)...))
        if positive
            if driver === LAPACK()
                @inbounds for j in n:-1:1
                    @simd for i in 1:min(minmn, j)
                        R̃[i, j] = R̃[i, j] * conj(sign_safe(R̃[i, i]))
                    end
                end
            else
                R̃f = view(R̃, 1:minmn, 1:n) # first minmn rows of R
                R̃f .= conj.(τ) .* R̃f
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
function householder_qr!(
        driver::Native, A::AbstractMatrix, Q::AbstractMatrix, R::AbstractMatrix;
        positive::Bool = true, pivoted::Bool = false, blocksize::Int = 1
    )
    # error messages for disallowing driver - setting combinations
    blocksize == 1 ||
        throw(ArgumentError(lazy"$driver does not provide a blocked QR decomposition"))
    pivoted &&
        throw(ArgumentError(lazy"$driver does not provide a pivoted QR decomposition"))
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

function householder_qr_null!(
        driver::Union{LAPACK, CUSOLVER, ROCSOLVER}, A::AbstractMatrix, N::AbstractMatrix;
        positive::Bool = true, pivoted::Bool = false,
        blocksize::Int = ((driver !== LAPACK() || pivoted) ? 1 : YALAPACK.default_qr_blocksize(A))
    )
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
function householder_qr_null!(
        driver::Native, A::AbstractMatrix, N::AbstractMatrix;
        positive::Bool = true, pivoted::Bool = false, blocksize::Int = 1
    )
    # error messages for disallowing driver - setting combinations
    blocksize == 1 ||
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
function qr_full!(A, QR, alg::DiagonalAlgorithm)
    check_input(qr_full!, A, QR, alg)
    Q, R = QR
    _diagonal_qr!(A, Q, R; alg.kwargs...)
    return Q, R
end
function qr_compact!(A, QR, alg::DiagonalAlgorithm)
    check_input(qr_compact!, A, QR, alg)
    Q, R = QR
    _diagonal_qr!(A, Q, R; alg.kwargs...)
    return Q, R
end
function qr_null!(A, N, alg::DiagonalAlgorithm)
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
            qr_full!(A, QR, alg::$algtype),
            qr_full!(A, QR, Householder($drivertype(), alg.kwargs))
        )
        Base.@deprecate(
            qr_compact!(A, QR, alg::$algtype),
            qr_compact!(A, QR, Householder($drivertype(), alg.kwargs))
        )
        Base.@deprecate(
            qr_null!(A, N, alg::$algtype),
            qr_null!(A, N, Householder($drivertype(), alg.kwargs))
        )
    end
end
