# Inputs
# ------
copy_input(::typeof(left_polar), A::AbstractMatrix) = copy_input(svd_full, A)
copy_input(::typeof(right_polar), A::AbstractMatrix) = copy_input(svd_full, A)

function check_input(::typeof(left_polar!), A::AbstractMatrix, WP)
    m, n = size(A)
    W, P = WP
    m >= n ||
        throw(ArgumentError("input matrix needs at least as many rows as columns"))
    @assert W isa AbstractMatrix && P isa AbstractMatrix
    @check_size(W, (m, n))
    @check_scalar(W, A)
    @check_size(P, (n, n))
    @check_scalar(P, A)
    return nothing
end
function check_input(::typeof(right_polar!), A::AbstractMatrix, PWᴴ)
    m, n = size(A)
    P, Wᴴ = PWᴴ
    n >= m ||
        throw(ArgumentError("input matrix needs at least as many columns as rows"))
    @assert P isa AbstractMatrix && Wᴴ isa AbstractMatrix
    @check_size(P, (m, m))
    @check_scalar(P, A)
    @check_size(Wᴴ, (m, n))
    @check_scalar(Wᴴ, A)
    return nothing
end

# Outputs
# -------
function initialize_output(::typeof(left_polar!), A::AbstractMatrix, ::PolarViaSVD)
    m, n = size(A)
    W = similar(A)
    P = similar(A, (n, n))
    return (W, P)
end
function initialize_output(::typeof(right_polar!), A::AbstractMatrix, ::PolarViaSVD)
    m, n = size(A)
    P = similar(A, (m, m))
    Wᴴ = similar(A)
    return (P, Wᴴ)
end

# Implementation
# --------------
function left_polar!(A::AbstractMatrix, WP, alg::PolarViaSVD)
    check_input(left_polar!, A, WP)
    U, S, Vᴴ = svd_compact!(A, alg.svdalg)
    W, P = WP
    W = mul!(W, U, Vᴴ)
    S .= sqrt.(S)
    SsqrtVᴴ = lmul!(S, Vᴴ)
    P = mul!(P, SsqrtVᴴ', SsqrtVᴴ)
    return (W, P)
end
function right_polar!(A::AbstractMatrix, PWᴴ, alg::PolarViaSVD)
    check_input(right_polar!, A, PWᴴ)
    U, S, Vᴴ = svd_compact!(A, alg.svdalg)
    P, Wᴴ = PWᴴ
    Wᴴ = mul!(Wᴴ, U, Vᴴ)
    S .= sqrt.(S)
    USsqrt = rmul!(U, S)
    P = mul!(P, USsqrt, USsqrt')
    return (P, Wᴴ)
end
