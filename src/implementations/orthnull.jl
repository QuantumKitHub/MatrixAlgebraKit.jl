# Inputs
# ------
copy_input(::typeof(left_orth), A::AbstractMatrix) = copy_input(qr_compact, A) # do we ever need anything else
copy_input(::typeof(right_orth), A::AbstractMatrix) = copy_input(lq_compact, A) # do we ever need anything else
copy_input(::typeof(left_null), A::AbstractMatrix) = copy_input(qr_null, A) # do we ever need anything else
copy_input(::typeof(right_null), A::AbstractMatrix) = copy_input(lq_null, A) # do we ever need anything else

function check_input(::typeof(left_orth!), A::AbstractMatrix, VC)
    m, n = size(A)
    minmn = min(m, n)
    V, C = VC
    @assert V isa AbstractMatrix && C isa AbstractMatrix
    @check_size(V, (m, minmn))
    @check_scalar(V, A)
    if !isempty(C)
        @check_size(C, (minmn, n))
        @check_scalar(C, A)
    end
    return nothing
end
function check_input(::typeof(right_orth!), A::AbstractMatrix, CVᴴ)
    m, n = size(A)
    minmn = min(m, n)
    C, Vᴴ = CVᴴ
    @assert C isa AbstractMatrix && Vᴴ isa AbstractMatrix
    if !isempty(C)
        @check_size(C, (m, minmn))
        @check_scalar(C, A)
    end
    @check_size(Vᴴ, (minmn, n))
    @check_scalar(Vᴴ, A)
    return nothing
end

function check_input(::typeof(left_null!), A::AbstractMatrix, N)
    m, n = size(A)
    minmn = min(m, n)
    @assert N isa AbstractMatrix
    @check_size(N, (m, m - minmn))
    @check_scalar(N, A)
    return nothing
end
function check_input(::typeof(right_null!), A::AbstractMatrix, Nᴴ)
    m, n = size(A)
    minmn = min(m, n)
    @assert Nᴴ isa AbstractMatrix
    @check_size(Nᴴ, (n - minmn, n))
    @check_scalar(Nᴴ, A)
    return nothing
end

# Outputs
# -------
function initialize_output(::typeof(left_orth!), A::AbstractMatrix)
    m, n = size(A)
    minmn = min(m, n)
    V = similar(A, (m, minmn))
    C = similar(A, (minmn, n))
    return (V, C)
end
function initialize_output(::typeof(right_orth!), A::AbstractMatrix)
    m, n = size(A)
    minmn = min(m, n)
    C = similar(A, (m, minmn))
    Vᴴ = similar(A, (minmn, n))
    return (C, Vᴴ)
end

function initialize_output(::typeof(left_null!), A::AbstractMatrix)
    m, n = size(A)
    minmn = min(m, n)
    N = similar(A, (m, m - minmn))
    return N
end
function initialize_output(::typeof(right_null!), A::AbstractMatrix)
    m, n = size(A)
    minmn = min(m, n)
    Nᴴ = similar(A, (n - minmn, n))
    return Nᴴ
end

# Implementation of orth functions
# --------------------------------
function left_orth!(A::AbstractMatrix, VC; kwargs...)
    check_input(left_orth!, A, VC)
    atol = get(kwargs, :atol, 0)
    rtol = get(kwargs, :rtol, 0)
    kind = get(kwargs, :kind, iszero(atol) && iszero(rtol) ? :qrpos : :svd)
    if !(iszero(atol) && iszero(rtol)) && kind != :svd
        throw(ArgumentError("nonzero tolerance not supported for left_orth with kind=$kind"))
    end
    if kind == :qr
        alg = get(kwargs, :alg, select_algorithm(qr_compact!, A))
        return qr_compact!(A, VC, alg)
    elseif kind == :qrpos
        alg = get(kwargs, :alg, select_algorithm(qr_compact!, A; positive=true))
        return qr_compact!(A, VC, alg)
    elseif kind == :polar
        size(A, 1) >= size(A, 2) ||
            throw(ArgumentError("`left_orth!` with `kind = :polar` only possible for `(m, n)` matrix with `m >= n`"))
        alg = get(kwargs, :alg, select_algorithm(left_polar!, A))
        return left_polar!(A, VC, alg)
    elseif kind == :svd && iszero(atol) && iszero(rtol)
        alg = get(kwargs, :alg, select_algorithm(svd_compact!, A))
        V, C = VC
        S = Diagonal(initialize_output(svd_vals!, A, alg))
        U, S, Vᴴ = svd_compact!(A, (V, S, C), alg)
        return U, lmul!(S, Vᴴ)
    elseif kind == :svd
        alg_svd = select_algorithm(svd_compact!, A)
        trunc = TruncationKeepAbove(atol, rtol)
        alg = get(kwargs, :alg, TruncatedAlgorithm(alg_svd, trunc))
        V, C = VC
        S = Diagonal(initialize_output(svd_vals!, A, alg_svd))
        U, S, Vᴴ = svd_trunc!(A, (V, S, C), alg)
        return U, lmul!(S, Vᴴ)
    else
        throw(ArgumentError("`left_orth!` received unknown value `kind = $kind`"))
    end
end

function right_orth!(A::AbstractMatrix, CVᴴ; kwargs...)
    check_input(right_orth!, A, CVᴴ)
    atol = get(kwargs, :atol, 0)
    rtol = get(kwargs, :rtol, 0)
    kind = get(kwargs, :kind, iszero(atol) && iszero(rtol) ? :lqpos : :svd)
    if !(iszero(atol) && iszero(rtol)) && kind != :svd
        throw(ArgumentError("nonzero tolerance not supported for left_orth with kind=$kind"))
    end
    if kind == :lq
        alg = get(kwargs, :alg, select_algorithm(lq_compact!, A))
        return lq_compact!(A, CVᴴ, alg)
    elseif kind == :lqpos
        alg = get(kwargs, :alg, select_algorithm(lq_compact!, A; positive=true))
        return lq_compact!(A, CVᴴ, alg)
    elseif kind == :polar
        size(A, 2) >= size(A, 1) ||
            throw(ArgumentError("`right_orth!` with `kind = :polar` only possible for `(m, n)` matrix with `m <= n`"))
        alg = get(kwargs, :alg, select_algorithm(right_polar!, A))
        return right_polar!(A, CVᴴ, alg)
    elseif kind == :svd && iszero(atol) && iszero(rtol)
        alg = get(kwargs, :alg, select_algorithm(svd_compact!, A))
        C, Vᴴ = CVᴴ
        S = Diagonal(initialize_output(svd_vals!, A, alg))
        U, S, Vᴴ = svd_compact!(A, (C, S, Vᴴ), alg)
        return rmul!(U, S), Vᴴ
    elseif kind == :svd
        alg_svd = select_algorithm(svd_compact!, A)
        trunc = TruncationKeepAbove(atol, rtol)
        alg = get(kwargs, :alg, TruncatedAlgorithm(alg_svd, trunc))
        C, Vᴴ = CVᴴ
        S = Diagonal(initialize_output(svd_vals!, A, alg_svd))
        U, S, Vᴴ = svd_trunc!(A, (C, S, Vᴴ), alg)
        return rmul!(U, S), Vᴴ
    else
        throw(ArgumentError("`right_orth!` received unknown value `kind = $kind`"))
    end
end

# Implementation of null functions
# --------------------------------
function left_null!(A::AbstractMatrix, N; kwargs...)
    check_input(left_null!, A, N)
    atol = get(kwargs, :atol, 0)
    rtol = get(kwargs, :rtol, 0)
    kind = get(kwargs, :kind, iszero(atol) && iszero(rtol) ? :qrpos : :svd)
    if !(iszero(atol) && iszero(rtol)) && kind != :svd
        throw(ArgumentError("nonzero tolerance not supported for left_orth with kind=$kind"))
    end
    if kind == :qr
        alg = get(kwargs, :alg, select_algorithm(qr_null!, A))
        return qr_null!(A, N, alg)
    elseif kind == :qrpos
        alg = get(kwargs, :alg, select_algorithm(qr_null!, A; positive=true))
        return qr_null!(A, N, alg)
    elseif kind == :svd && iszero(atol) && iszero(rtol)
        alg = get(kwargs, :alg, select_algorithm(svd_full!, A))
        U, _, _ = svd_full!(A, alg)
        (m, n) = size(A)
        return copy!(N, view(U, 1:m, (n + 1):m))
    elseif kind == :svd
        alg = get(kwargs, :alg, select_algorithm(svd_full!, A))
        U, S, _ = svd_full!(A, alg)
        trunc = TruncationKeepBelow(atol, rtol)
        return truncate!(left_null!, (U, S), trunc)
    else
        throw(ArgumentError("`left_null!` received unknown value `kind = $kind`"))
    end
end

function right_null!(A::AbstractMatrix, Nᴴ; kwargs...)
    check_input(right_null!, A, Nᴴ)
    atol = get(kwargs, :atol, 0)
    rtol = get(kwargs, :rtol, 0)
    kind = get(kwargs, :kind, iszero(atol) && iszero(rtol) ? :lqpos : :svd)
    if !(iszero(atol) && iszero(rtol)) && kind != :svd
        throw(ArgumentError("nonzero tolerance not supported for left_orth with kind=$kind"))
    end
    if kind == :lq
        alg = get(kwargs, :alg, select_algorithm(lq_null!, A))
        return lq_null!(A, Nᴴ, alg)
    elseif kind == :lqpos
        alg = get(kwargs, :alg, select_algorithm(lq_null!, A; positive=true))
        return lq_null!(A, Nᴴ, alg)
    elseif kind == :svd && iszero(atol) && iszero(rtol)
        alg = get(kwargs, :alg, select_algorithm(svd_full!, A))
        _, _, Vᴴ = svd_full!(A, alg)
        (m, n) = size(A)
        return copy!(Nᴴ, view(Vᴴ, (m + 1):n, 1:n))
    elseif kind == :svd
        alg = get(kwargs, :alg, select_algorithm(svd_full!, A))
        _, S, Vᴴ = svd_full!(A, alg)
        trunc = TruncationKeepBelow(atol, rtol)
        return truncate!(right_null!, (S, Vᴴ), trunc)
    else
        throw(ArgumentError("`right_null!` received unknown value `kind = $kind`"))
    end
end