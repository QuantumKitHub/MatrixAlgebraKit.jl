# Inputs
# ------
copy_input(::typeof(left_polar), A) = copy_input(svd_full, A)
copy_input(::typeof(right_polar), A) = copy_input(svd_full, A)

function check_input(::typeof(left_polar!), A::AbstractMatrix, WP, ::AbstractAlgorithm)
    m, n = size(A)
    W, P = WP
    m ≥ n ||
        throw(ArgumentError("input matrix needs at least as many rows ($m) as columns ($n)"))
    @assert W isa AbstractMatrix && P isa AbstractMatrix
    @check_size(W, (m, n))
    @check_scalar(W, A)
    isempty(P) || @check_size(P, (n, n))
    @check_scalar(P, A)
    return nothing
end
function check_input(::typeof(right_polar!), A::AbstractMatrix, PWᴴ, ::AbstractAlgorithm)
    m, n = size(A)
    P, Wᴴ = PWᴴ
    n ≥ m ||
        throw(ArgumentError("input matrix needs at least as many columns ($n) as rows ($m)"))
    @assert P isa AbstractMatrix && Wᴴ isa AbstractMatrix
    isempty(P) || @check_size(P, (m, m))
    @check_scalar(P, A)
    @check_size(Wᴴ, (m, n))
    @check_scalar(Wᴴ, A)
    return nothing
end

# Outputs
# -------
function initialize_output(::typeof(left_polar!), A::AbstractMatrix, ::AbstractAlgorithm)
    m, n = size(A)
    W = similar(A)
    P = similar(A, (n, n))
    return (W, P)
end
function initialize_output(::typeof(right_polar!), A::AbstractMatrix, ::AbstractAlgorithm)
    m, n = size(A)
    P = similar(A, (m, m))
    Wᴴ = similar(A)
    return (P, Wᴴ)
end

# Implementation via SVD
# -----------------------
function left_polar!(A::AbstractMatrix, WP, alg::PolarViaSVD)
    check_input(left_polar!, A, WP, alg)
    U, S, Vᴴ = svd_compact!(A, alg.svd_alg)
    W, P = WP
    W = mul!(W, U, Vᴴ)
    if !isempty(P)
        S .= sqrt.(S)
        SsqrtVᴴ = lmul!(S, Vᴴ)
        P = mul!(P, SsqrtVᴴ', SsqrtVᴴ)
    end
    return (W, P)
end
function right_polar!(A::AbstractMatrix, PWᴴ, alg::PolarViaSVD)
    check_input(right_polar!, A, PWᴴ, alg)
    U, S, Vᴴ = svd_compact!(A, alg.svd_alg)
    P, Wᴴ = PWᴴ
    Wᴴ = mul!(Wᴴ, U, Vᴴ)
    if !isempty(P)
        S .= sqrt.(S)
        USsqrt = rmul!(U, S)
        P = mul!(P, USsqrt, USsqrt')
    end
    return (P, Wᴴ)
end

# Implementation via Newton
# --------------------------
function left_polar!(A::AbstractMatrix, WP, alg::PolarNewton)
    check_input(left_polar!, A, WP, alg)
    W, P = WP
    if isempty(P)
        W = _left_polarnewton!(A, W, P; alg.kwargs...)
        return W, P
    else
        W = _left_polarnewton!(copy(A), W, P; alg.kwargs...)
        # we still need `A` to compute `P`
        P = project_hermitian!(mul!(P, W', A))
        return W, P
    end
end

function right_polar!(A::AbstractMatrix, PWᴴ, alg::PolarNewton)
    check_input(right_polar!, A, PWᴴ, alg)
    P, Wᴴ = PWᴴ
    if isempty(P)
        Wᴴ = _right_polarnewton!(A, Wᴴ, P; alg.kwargs...)
        return P, Wᴴ
    else
        Wᴴ = _right_polarnewton!(copy(A), Wᴴ, P; alg.kwargs...)
        # we still need `A` to compute `P`
        P = project_hermitian!(mul!(P, A, Wᴴ'))
        return P, Wᴴ
    end
end

# these methods only compute W and destroy A in the process
function _left_polarnewton!(A::AbstractMatrix, W, P = similar(A, (0, 0)); tol = defaulttol(A), maxiter = 10)
    m, n = size(A) # we must have m >= n
    Rᴴinv = isempty(P) ? similar(P, (n, n)) : P # use P as workspace when available
    if m > n # initial QR
        Q, R = qr_compact!(A)
        Rc = view(A, 1:n, 1:n)
        Rc .= R
        Rᴴinv = ldiv!(UpperTriangular(Rc)', one!(Rᴴinv))
    else # m == n
        R = A
        Rc = view(W, 1:n, 1:n)
        Rc .= R
        Rᴴinv = ldiv!(lu!(Rc)', one!(Rᴴinv))
    end
    γ = sqrt(norm(Rᴴinv) / norm(R)) # scaling factor
    rmul!(R, γ)
    rmul!(Rᴴinv, 1 / γ)
    R, Rᴴinv = _avgdiff!(R, Rᴴinv)
    Rc .= R
    i = 1
    conv = norm(Rᴴinv, Inf)
    while i < maxiter && conv > tol
        Rᴴinv = ldiv!(lu!(Rc)', one!(Rᴴinv))
        γ = sqrt(norm(Rᴴinv) / norm(R)) # scaling factor
        rmul!(R, γ)
        rmul!(Rᴴinv, 1 / γ)
        R, Rᴴinv = _avgdiff!(R, Rᴴinv)
        Rc .= R
        conv = norm(Rᴴinv, Inf)
        i += 1
    end
    if conv > tol
        @warn "`left_polar!` via Newton iteration did not converge within $maxiter iterations (final residual: $conv)"
    end
    if m > n
        return mul!(W, Q, Rc)
    end
    return W
end

function _right_polarnewton!(A::AbstractMatrix, Wᴴ, P = similar(A, (0, 0)); tol = defaulttol(A), maxiter = 10)
    m, n = size(A) # we must have m <= n
    Lᴴinv = isempty(P) ? similar(P, (m, m)) : P # use P as workspace when available
    if m < n # initial QR
        L, Q = lq_compact!(A)
        Lc = view(A, 1:m, 1:m)
        copy!(Lc, L)
        Lᴴinv = ldiv!(LowerTriangular(Lc)', one!(Lᴴinv))
    else # m == n
        L = A
        Lc = view(Wᴴ, 1:m, 1:m)
        Lc .= L
        Lᴴinv = ldiv!(lu!(Lc)', one!(Lᴴinv))
    end
    γ = sqrt(norm(Lᴴinv) / norm(L)) # scaling factor
    rmul!(L, γ)
    rmul!(Lᴴinv, 1 / γ)
    L, Lᴴinv = _avgdiff!(L, Lᴴinv)
    copy!(Lc, L)
    i = 1
    conv = norm(Lᴴinv, Inf)
    while i < maxiter && conv > tol
        Lᴴinv = ldiv!(lu!(Lc)', one!(Lᴴinv))
        γ = sqrt(norm(Lᴴinv) / norm(L)) # scaling factor
        rmul!(L, γ)
        rmul!(Lᴴinv, 1 / γ)
        L, Lᴴinv = _avgdiff!(L, Lᴴinv)
        Lc .= L
        conv = norm(Lᴴinv, Inf)
        i += 1
    end
    if conv > tol
        @warn "`right_polar!` via Newton iteration did not converge within $maxiter iterations (final residual: $conv)"
    end
    if m < n
        return mul!(Wᴴ, Lc, Q)
    end
    return Wᴴ
end

# in place computation of the average and difference of two arrays
function _avgdiff!(A::AbstractArray, B::AbstractArray)
    axes(A) == axes(B) || throw(DimensionMismatch())
    @simd for I in eachindex(A, B)
        @inbounds begin
            a = A[I]
            b = B[I]
            A[I] = (a + b) / 2
            B[I] = b - a
        end
    end
    return A, B
end
