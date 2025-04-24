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
function left_orth!(A::AbstractMatrix, VC; trunc=nothing,
                    kind=isnothing(trunc) ? :qr : :svd, alg_qr=(; positive=true),
                    alg_polar=(;), alg_svd=(;))
    check_input(left_orth!, A, VC)
    if !isnothing(trunc) && kind != :svd
        throw(ArgumentError("truncation not supported for left_orth with kind=$kind"))
    end
    if kind == :qr
        alg_qr′ = _select_algorithm(qr_compact!, A, alg_qr)
        return qr_compact!(A, VC, alg_qr′)
    elseif kind == :polar
        size(A, 1) >= size(A, 2) ||
            throw(ArgumentError("`left_orth!` with `kind = :polar` only possible for `(m, n)` matrix with `m >= n`"))
        alg_polar′ = _select_algorithm(left_polar!, A, alg_polar)
        return left_polar!(A, VC, alg_polar′)
    elseif kind == :svd && isnothing(trunc)
        alg_svd′ = _select_algorithm(svd_compact!, A, alg_svd)
        V, C = VC
        S = Diagonal(initialize_output(svd_vals!, A, alg_svd′))
        U, S, Vᴴ = svd_compact!(A, (V, S, C), alg_svd′)
        return U, lmul!(S, Vᴴ)
    elseif kind == :svd
        alg_svd′ = _select_algorithm(svd_compact!, A, alg_svd)
        alg_svd_trunc = select_algorithm(svd_trunc!, A; trunc, alg=alg_svd′)
        V, C = VC
        S = Diagonal(initialize_output(svd_vals!, A, alg_svd_trunc.alg))
        U, S, Vᴴ = svd_trunc!(A, (V, S, C), alg_svd_trunc)
        return U, lmul!(S, Vᴴ)
    else
        throw(ArgumentError("`left_orth!` received unknown value `kind = $kind`"))
    end
end

function right_orth!(A::AbstractMatrix, CVᴴ; trunc=nothing,
                     kind=isnothing(trunc) ? :lq : :svd, alg_lq=(; positive=true),
                     alg_polar=(;), alg_svd=(;))
    check_input(right_orth!, A, CVᴴ)
    if !isnothing(trunc) && kind != :svd
        throw(ArgumentError("truncation not supported for right_orth with kind=$kind"))
    end
    if kind == :lq
        alg_lq′ = _select_algorithm(lq_compact!, A, alg_lq)
        return lq_compact!(A, CVᴴ, alg_lq′)
    elseif kind == :polar
        size(A, 2) >= size(A, 1) ||
            throw(ArgumentError("`right_orth!` with `kind = :polar` only possible for `(m, n)` matrix with `m <= n`"))
        alg_polar′ = _select_algorithm(right_polar!, A, alg_polar)
        return right_polar!(A, CVᴴ, alg_polar′)
    elseif kind == :svd && isnothing(trunc)
        alg_svd′ = _select_algorithm(svd_compact!, A, alg_svd)
        C, Vᴴ = CVᴴ
        S = Diagonal(initialize_output(svd_vals!, A, alg_svd′))
        U, S, Vᴴ = svd_compact!(A, (C, S, Vᴴ), alg_svd′)
        return rmul!(U, S), Vᴴ
    elseif kind == :svd
        alg_svd′ = _select_algorithm(svd_compact!, A, alg_svd)
        alg_svd_trunc = select_algorithm(svd_trunc!, A; trunc, alg=alg_svd′)
        C, Vᴴ = CVᴴ
        S = Diagonal(initialize_output(svd_vals!, A, alg_svd_trunc.alg))
        U, S, Vᴴ = svd_trunc!(A, (C, S, Vᴴ), alg_svd_trunc)
        return rmul!(U, S), Vᴴ
    else
        throw(ArgumentError("`right_orth!` received unknown value `kind = $kind`"))
    end
end

# Implementation of null functions
# --------------------------------
function null_truncation_strategy(; atol=nothing, rtol=nothing, maxnullity=nothing)
    if isnothing(maxnullity) && isnothing(atol) && isnothing(rtol)
        return NoTruncation()
    end
    atol = @something atol 0
    rtol = @something rtol 0
    trunc = TruncationKeepBelow(atol, rtol)
    return !isnothing(maxnullity) ? trunc & truncrank(maxnullity; rev=false) : trunc
end

function left_null!(A::AbstractMatrix, N; trunc=nothing,
                    kind=isnothing(trunc) ? :qr : :svd, alg_qr=(; positive=true),
                    alg_svd=(;))
    check_input(left_null!, A, N)
    if !isnothing(trunc) && kind != :svd
        throw(ArgumentError("truncation not supported for left_null with kind=$kind"))
    end
    if kind == :qr
        alg_qr′ = _select_algorithm(qr_null!, A, alg_qr)
        return qr_null!(A, N, alg_qr′)
    elseif kind == :svd && isnothing(trunc)
        alg_svd′ = _select_algorithm(svd_full!, A, alg_svd)
        U, _, _ = svd_full!(A, alg_svd′)
        (m, n) = size(A)
        return copy!(N, view(U, 1:m, (n + 1):m))
    elseif kind == :svd
        alg_svd′ = _select_algorithm(svd_full!, A, alg_svd)
        U, S, _ = svd_full!(A, alg_svd′)
        trunc′ = trunc isa TruncationStrategy ? trunc :
                 trunc isa NamedTuple ? null_truncation_strategy(; trunc...) :
                 throw(ArgumentError("Unknown truncation strategy: $trunc"))
        return truncate!(left_null!, (U, S), trunc′)
    else
        throw(ArgumentError("`left_null!` received unknown value `kind = $kind`"))
    end
end

function right_null!(A::AbstractMatrix, Nᴴ; trunc=nothing,
                     kind=isnothing(trunc) ? :lq : :svd, alg_lq=(; positive=true),
                     alg_svd=(;))
    check_input(right_null!, A, Nᴴ)
    if !isnothing(trunc) && kind != :svd
        throw(ArgumentError("truncation not supported for right_null with kind=$kind"))
    end
    if kind == :lq
        alg_lq′ = _select_algorithm(lq_null!, A, alg_lq)
        return lq_null!(A, Nᴴ, alg_lq′)
    elseif kind == :svd && isnothing(trunc)
        alg_svd′ = _select_algorithm(svd_full!, A, alg_svd)
        _, _, Vᴴ = svd_full!(A, alg_svd′)
        (m, n) = size(A)
        return copy!(Nᴴ, view(Vᴴ, (m + 1):n, 1:n))
    elseif kind == :svd
        alg_svd′ = _select_algorithm(svd_full!, A, alg_svd)
        _, S, Vᴴ = svd_full!(A, alg_svd′)
        trunc′ = trunc isa TruncationStrategy ? trunc :
                 trunc isa NamedTuple ? null_truncation_strategy(; trunc...) :
                 throw(ArgumentError("Unknown truncation strategy: $trunc"))
        return truncate!(right_null!, (S, Vᴴ), trunc′)
    else
        throw(ArgumentError("`right_null!` received unknown value `kind = $kind`"))
    end
end
