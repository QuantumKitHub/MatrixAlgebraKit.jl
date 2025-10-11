# Inputs
# ------
copy_input(::typeof(left_orth), A) = copy_input(qr_compact, A) # do we ever need anything else
copy_input(::typeof(right_orth), A) = copy_input(lq_compact, A) # do we ever need anything else
copy_input(::typeof(left_null), A) = copy_input(qr_null, A) # do we ever need anything else
copy_input(::typeof(right_null), A) = copy_input(lq_null, A) # do we ever need anything else

check_input(::typeof(left_orth!), A, VC, alg::AbstractAlgorithm) =
    check_input(left_orth_kind(alg), A, VC, alg)

check_input(::typeof(right_orth!), A, CVᴴ, alg::AbstractAlgorithm) =
    check_input(right_orth_kind(alg), A, CVᴴ, alg)


check_input(::typeof(left_orth_qr!), A, VC, alg::AbstractAlgorithm) =
    check_input(qr_compact!, A, VC, alg)
check_input(::typeof(left_orth_polar!), A, VC, alg::AbstractAlgorithm) =
    check_input(left_polar!, A, VC, alg)
check_input(::typeof(left_orth_svd!), A, VC, alg::AbstractAlgorithm) =
    check_input(qr_compact!, A, VC, alg)

check_input(::typeof(right_orth_lq!), A, CVᴴ, alg::AbstractAlgorithm) =
    check_input(lq_compact!, A, CVᴴ, alg)
check_input(::typeof(right_orth_polar!), A, CVᴴ, alg::AbstractAlgorithm) =
    check_input(right_polar!, A, CVᴴ, alg)
check_input(::typeof(right_orth_svd!), A, CVᴴ, alg::AbstractAlgorithm) =
    check_input(lq_compact!, A, CVᴴ, alg)


initialize_output(::typeof(left_orth!), A, alg::AbstractAlgorithm) =
    initialize_output(left_orth_kind(alg), A, alg)
initialize_output(::typeof(right_orth!), A, alg::AbstractAlgorithm) =
    initialize_output(right_orth_kind(alg), A, alg)


initialize_output(::typeof(left_orth_qr!), A, alg::AbstractAlgorithm) =
    initialize_output(qr_compact!, A, alg)
initialize_output(::typeof(left_orth_polar!), A, alg::AbstractAlgorithm) =
    initialize_output(left_polar!, A, alg)
initialize_output(::typeof(left_orth_svd!), A, alg::AbstractAlgorithm) =
    initialize_output(qr_compact!, A, alg)

initialize_output(::typeof(right_orth_lq!), A, alg::AbstractAlgorithm) =
    initialize_output(lq_compact!, A, alg)
initialize_output(::typeof(right_orth_polar!), A, alg::AbstractAlgorithm) =
    initialize_output(right_polar!, A, alg)
initialize_output(::typeof(right_orth_svd!), A, alg::AbstractAlgorithm) =
    initialize_output(lq_compact!, A, alg)


function check_input(::typeof(left_null!), A::AbstractMatrix, N, ::AbstractAlgorithm)
    m, n = size(A)
    minmn = min(m, n)
    @assert N isa AbstractMatrix
    @check_size(N, (m, m - minmn))
    @check_scalar(N, A)
    return nothing
end
function check_input(::typeof(right_null!), A::AbstractMatrix, Nᴴ, ::AbstractAlgorithm)
    m, n = size(A)
    minmn = min(m, n)
    @assert Nᴴ isa AbstractMatrix
    @check_size(Nᴴ, (n - minmn, n))
    @check_scalar(Nᴴ, A)
    return nothing
end

# Outputs
# -------

function initialize_orth_svd(A::AbstractMatrix, F, alg)
    S = Diagonal(initialize_output(svd_vals!, A, alg))
    return F[1], S, F[2]
end
# fallback doesn't re-use F at all
initialize_orth_svd(A, F, alg) = initialize_output(svd_compact!, A, alg)

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
left_orth!(A, VC, alg::AbstractAlgorithm) = left_orth_kind(alg)(A, VC, alg)
right_orth!(A, CVᴴ, alg::AbstractAlgorithm) = right_orth_kind(alg)(A, CVᴴ, alg)

left_orth_qr!(A, VC, alg::AbstractAlgorithm) = qr_compact!(A, VC, alg)
right_orth_lq!(A, CVᴴ, alg::AbstractAlgorithm) = lq_compact!(A, CVᴴ, alg)
left_orth_polar!(A, VC, alg::AbstractAlgorithm) = left_polar!(A, VC, alg)
right_orth_polar!(A, CVᴴ, alg::AbstractAlgorithm) = right_polar!(A, CVᴴ, alg)

# orth_svd requires implementations of `lmul!` and `rmul!`
function left_orth_svd!(A, VC, alg::AbstractAlgorithm)
    check_input(left_orth_svd!, A, VC, alg)
    USVᴴ = initialize_orth_svd(A, VC, alg)
    V, S, C = does_truncate(alg) ? svd_trunc!(A, USVᴴ, alg) : svd_compact!(A, USVᴴ, alg)
    lmul!(S, C)
    return V, C
end
function right_orth_svd!(A, CVᴴ, alg::AbstractAlgorithm)
    check_input(right_orth_svd!, A, CVᴴ, alg)
    USVᴴ = initialize_orth_svd(A, CVᴴ, alg)
    C, S, Vᴴ = does_truncate(alg) ? svd_trunc!(A, USVᴴ, alg) : svd_compact!(A, USVᴴ, alg)
    rmul!(C, S)
    return C, Vᴴ
end

# Implementation of null functions
# --------------------------------
function null_truncation_strategy(; atol = nothing, rtol = nothing, maxnullity = nothing)
    if isnothing(maxnullity) && isnothing(atol) && isnothing(rtol)
        return notrunc()
    end
    atol = @something atol 0
    rtol = @something rtol 0
    trunc = trunctol(; atol, rtol, keep_below = true)
    return !isnothing(maxnullity) ? trunc & truncrank(maxnullity; rev = false) : trunc
end

function left_null!(
        A, N;
        trunc = nothing, kind = isnothing(trunc) ? :qr : :svd,
        alg_qr = (; positive = true), alg_svd = (;)
    )
    if !isnothing(trunc) && kind != :svd
        throw(ArgumentError("truncation not supported for left_null with kind=$kind"))
    end
    return if kind == :qr
        left_null_qr!(A, N, alg_qr)
    elseif kind == :svd
        left_null_svd!(A, N, alg_svd, trunc)
    else
        throw(ArgumentError("`left_null!` received unknown value `kind = $kind`"))
    end
end
function left_null_qr!(A, N, alg)
    alg′ = select_algorithm(qr_null!, A, alg)
    check_input(left_null!, A, N, alg′)
    return qr_null!(A, N, alg′)
end
function left_null_svd!(A, N, alg, trunc::Nothing = nothing)
    alg′ = select_algorithm(svd_full!, A, alg)
    check_input(left_null!, A, N, alg′)
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
    N, ind = truncate(left_null!, (U, S), trunc′)
    return N
end

function right_null!(
        A, Nᴴ;
        trunc = nothing, kind = isnothing(trunc) ? :lq : :svd,
        alg_lq = (; positive = true), alg_svd = (;)
    )
    if !isnothing(trunc) && kind != :svd
        throw(ArgumentError("truncation not supported for right_null with kind=$kind"))
    end
    if kind == :lq
        return right_null_lq!(A, Nᴴ, alg_lq)
    elseif kind == :svd
        return right_null_svd!(A, Nᴴ, alg_svd, trunc)
    else
        throw(ArgumentError("`right_null!` received unknown value `kind = $kind`"))
    end
end
function right_null_lq!(A, Nᴴ, alg)
    alg′ = select_algorithm(lq_null!, A, alg)
    check_input(right_null!, A, Nᴴ, alg′)
    return lq_null!(A, Nᴴ, alg′)
end
function right_null_svd!(A, Nᴴ, alg, trunc::Nothing = nothing)
    alg′ = select_algorithm(svd_full!, A, alg)
    check_input(right_null!, A, Nᴴ, alg′)
    _, _, Vᴴ = svd_full!(A, alg′)
    (m, n) = size(A)
    return copy!(Nᴴ, view(Vᴴ, (m + 1):n, 1:n))
end
function right_null_svd!(A, Nᴴ, alg, trunc)
    alg′ = select_algorithm(svd_full!, A, alg)
    check_input(right_null!, A, Nᴴ, alg′)
    _, S, Vᴴ = svd_full!(A, alg′)
    trunc′ = trunc isa TruncationStrategy ? trunc :
        trunc isa NamedTuple ? null_truncation_strategy(; trunc...) :
        throw(ArgumentError("Unknown truncation strategy: $trunc"))
    return first(truncate(right_null!, (S, Vᴴ), trunc′))
end
