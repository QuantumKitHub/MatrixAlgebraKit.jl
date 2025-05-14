# Inputs
# ------
copy_input(::typeof(left_orth), A) = copy_input(qr_compact, A) # do we ever need anything else
copy_input(::typeof(right_orth), A) = copy_input(lq_compact, A) # do we ever need anything else
copy_input(::typeof(left_null), A) = copy_input(qr_null, A) # do we ever need anything else
copy_input(::typeof(right_null), A) = copy_input(lq_null, A) # do we ever need anything else

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
function left_orth!(A, VC; trunc=nothing,
                    kind=isnothing(trunc) ? :qr : :svd, alg_qr=(; positive=true),
                    alg_polar=(;), alg_svd=(;))
    check_input(left_orth!, A, VC)
    if !isnothing(trunc) && kind != :svd
        throw(ArgumentError("truncation not supported for left_orth with kind=$kind"))
    end
    if kind == :qr
        return left_orth_qr!(A, VC, alg_qr)
    elseif kind == :polar
        return left_orth_polar!(A, VC, alg_polar)
    elseif kind == :svd
        return left_orth_svd!(A, VC, alg_svd, trunc)
    else
        throw(ArgumentError("`left_orth!` received unknown value `kind = $kind`"))
    end
end
function left_orth_qr!(A, VC, alg)
    alg′ = select_algorithm(qr_compact!, A, alg)
    return qr_compact!(A, VC, alg′)
end
function left_orth_polar!(A, VC, alg)
    alg′ = select_algorithm(left_polar!, A, alg)
    return left_polar!(A, VC, alg′)
end
function left_orth_svd!(A, VC, alg, trunc::Nothing=nothing)
    alg′ = select_algorithm(svd_compact!, A, alg)
    V, C = VC
    S = Diagonal(initialize_output(svd_vals!, A, alg′))
    U, S, Vᴴ = svd_compact!(A, (V, S, C), alg′)
    return U, lmul!(S, Vᴴ)
end
function left_orth_svd!(A, VC, alg, trunc)
    alg′ = select_algorithm(svd_compact!, A, alg)
    alg_trunc = select_algorithm(svd_trunc!, A, alg′; trunc)
    V, C = VC
    S = Diagonal(initialize_output(svd_vals!, A, alg_trunc.alg))
    U, S, Vᴴ = svd_trunc!(A, (V, S, C), alg_trunc)
    return U, lmul!(S, Vᴴ)
end

function right_orth!(A, CVᴴ; trunc=nothing,
                     kind=isnothing(trunc) ? :lq : :svd, alg_lq=(; positive=true),
                     alg_polar=(;), alg_svd=(;))
    check_input(right_orth!, A, CVᴴ)
    if !isnothing(trunc) && kind != :svd
        throw(ArgumentError("truncation not supported for right_orth with kind=$kind"))
    end
    if kind == :lq
        return right_orth_lq!(A, CVᴴ, alg_lq)
    elseif kind == :polar
        return right_orth_polar!(A, CVᴴ, alg_polar)
    elseif kind == :svd
        return right_orth_svd!(A, CVᴴ, alg_svd, trunc)
    else
        throw(ArgumentError("`right_orth!` received unknown value `kind = $kind`"))
    end
end
function right_orth_lq!(A, CVᴴ, alg)
    alg′ = select_algorithm(lq_compact!, A, alg)
    return lq_compact!(A, CVᴴ, alg′)
end
function right_orth_polar!(A, CVᴴ, alg)
    alg′ = select_algorithm(right_polar!, A, alg)
    return right_polar!(A, CVᴴ, alg′)
end
function right_orth_svd!(A, CVᴴ, alg, trunc::Nothing=nothing)
    alg′ = select_algorithm(svd_compact!, A, alg)
    C, Vᴴ = CVᴴ
    S = Diagonal(initialize_output(svd_vals!, A, alg′))
    U, S, Vᴴ = svd_compact!(A, (C, S, Vᴴ), alg′)
    return rmul!(U, S), Vᴴ
end
function right_orth_svd!(A, CVᴴ, alg, trunc)
    alg′ = select_algorithm(svd_compact!, A, alg)
    alg_trunc = select_algorithm(svd_trunc!, A, alg′; trunc)
    C, Vᴴ = CVᴴ
    S = Diagonal(initialize_output(svd_vals!, A, alg_trunc.alg))
    U, S, Vᴴ = svd_trunc!(A, (C, S, Vᴴ), alg_trunc)
    return rmul!(U, S), Vᴴ
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

function left_null!(A, N; trunc=nothing,
                    kind=isnothing(trunc) ? :qr : :svd, alg_qr=(; positive=true),
                    alg_svd=(;))
    check_input(left_null!, A, N)
    if !isnothing(trunc) && kind != :svd
        throw(ArgumentError("truncation not supported for left_null with kind=$kind"))
    end
    if kind == :qr
        left_null_qr!(A, N, alg_qr)
    elseif kind == :svd
        left_null_svd!(A, N, alg_svd, trunc)
    else
        throw(ArgumentError("`left_null!` received unknown value `kind = $kind`"))
    end
end
function left_null_qr!(A, N, alg)
    alg′ = select_algorithm(qr_null!, A, alg)
    return qr_null!(A, N, alg′)
end
function left_null_svd!(A, N, alg, trunc::Nothing=nothing)
    alg′ = select_algorithm(svd_full!, A, alg)
    U, _, _ = svd_full!(A, alg′)
    (m, n) = size(A)
    return copy!(N, view(U, 1:m, (n + 1):m))
end
function left_null_svd!(A, N, alg, trunc)
    alg′ = select_algorithm(svd_full!, A, alg)
    U, S, _ = svd_full!(A, alg′)
    trunc′ = trunc isa TruncationStrategy ? trunc :
             trunc isa NamedTuple ? null_truncation_strategy(; trunc...) :
             throw(ArgumentError("Unknown truncation strategy: $trunc"))
    return truncate!(left_null!, (U, S), trunc′)
end

function right_null!(A, Nᴴ; trunc=nothing,
                     kind=isnothing(trunc) ? :lq : :svd, alg_lq=(; positive=true),
                     alg_svd=(;))
    check_input(right_null!, A, Nᴴ)
    if !isnothing(trunc) && kind != :svd
        throw(ArgumentError("truncation not supported for right_null with kind=$kind"))
    end
    if kind == :lq
        return right_null_lq!(A, Nᴴ, alg_lq)
    elseif kind == :svd
        return right_null_svd!(A, Nᴴ, alg_svd)
    else
        throw(ArgumentError("`right_null!` received unknown value `kind = $kind`"))
    end
end
function right_null_lq!(A, Nᴴ, alg)
    alg′ = select_algorithm(lq_null!, A, alg)
    return lq_null!(A, Nᴴ, alg′)
end
function right_null_svd!(A, Nᴴ, alg, trunc::Nothing=nothing)
    alg′ = select_algorithm(svd_full!, A, alg)
    _, _, Vᴴ = svd_full!(A, alg′)
    (m, n) = size(A)
    return copy!(Nᴴ, view(Vᴴ, (m + 1):n, 1:n))
end
function right_null_svd!(A, Nᴴ, alg, trunc)
    alg′ = select_algorithm(svd_full!, A, alg)
    _, S, Vᴴ = svd_full!(A, alg′)
    trunc′ = trunc isa TruncationStrategy ? trunc :
             trunc isa NamedTuple ? null_truncation_strategy(; trunc...) :
             throw(ArgumentError("Unknown truncation strategy: $trunc"))
    return truncate!(right_null!, (S, Vᴴ), trunc′)
end
