# Orthogonalization
# -----------------
copy_input(::typeof(left_orth), A) = copy_input(qr_compact, A) # do we ever need anything else
copy_input(::typeof(right_orth), A) = copy_input(lq_compact, A) # do we ever need anything else
copy_input(::typeof(left_null), A) = copy_input(qr_null, A) # do we ever need anything else
copy_input(::typeof(right_null), A) = copy_input(lq_null, A) # do we ever need anything else

check_input(::typeof(left_orth!), A, VC, alg::AbstractAlgorithm) =
    check_input(left_orth_kind(alg), A, VC, alg)
check_input(::typeof(left_orth_qr!), A, VC, alg::AbstractAlgorithm) =
    check_input(qr_compact!, A, VC, alg)
check_input(::typeof(left_orth_polar!), A, VC, alg::AbstractAlgorithm) =
    check_input(left_polar!, A, VC, alg)
check_input(::typeof(left_orth_svd!), A, VC, alg::AbstractAlgorithm) =
    check_input(qr_compact!, A, VC, alg)

check_input(::typeof(right_orth!), A, CVᴴ, alg::AbstractAlgorithm) =
    check_input(right_orth_kind(alg), A, CVᴴ, alg)
check_input(::typeof(right_orth_lq!), A, CVᴴ, alg::AbstractAlgorithm) =
    check_input(lq_compact!, A, CVᴴ, alg)
check_input(::typeof(right_orth_polar!), A, CVᴴ, alg::AbstractAlgorithm) =
    check_input(right_polar!, A, CVᴴ, alg)
check_input(::typeof(right_orth_svd!), A, CVᴴ, alg::AbstractAlgorithm) =
    check_input(lq_compact!, A, CVᴴ, alg)

check_input(::typeof(left_null!), A, N, alg::AbstractAlgorithm) =
    check_input(left_null_kind(alg), A, N, alg)
check_input(::typeof(left_null_qr!), A, N, alg::AbstractAlgorithm) =
    check_input(qr_null!, A, N, alg)
check_input(::typeof(left_null_svd!), A, N, alg::AbstractAlgorithm) = nothing

check_input(::typeof(right_null!), A, Nᴴ, alg::AbstractAlgorithm) =
    check_input(right_null_kind(alg), A, Nᴴ, alg)
check_input(::typeof(right_null_lq!), A, Nᴴ, alg::AbstractAlgorithm) =
    check_input(lq_null!, A, Nᴴ, alg)
check_input(::typeof(right_null_svd!), A, Nᴴ, alg::AbstractAlgorithm) = nothing


initialize_output(::typeof(left_orth!), A, alg::AbstractAlgorithm) =
    initialize_output(left_orth_kind(alg), A, alg)
initialize_output(::typeof(left_orth_qr!), A, alg::AbstractAlgorithm) =
    initialize_output(qr_compact!, A, alg)
initialize_output(::typeof(left_orth_polar!), A, alg::AbstractAlgorithm) =
    initialize_output(left_polar!, A, alg)
initialize_output(::typeof(left_orth_svd!), A, alg::AbstractAlgorithm) =
    initialize_output(qr_compact!, A, alg)

initialize_output(::typeof(right_orth!), A, alg::AbstractAlgorithm) =
    initialize_output(right_orth_kind(alg), A, alg)
initialize_output(::typeof(right_orth_lq!), A, alg::AbstractAlgorithm) =
    initialize_output(lq_compact!, A, alg)
initialize_output(::typeof(right_orth_polar!), A, alg::AbstractAlgorithm) =
    initialize_output(right_polar!, A, alg)
initialize_output(::typeof(right_orth_svd!), A, alg::AbstractAlgorithm) =
    initialize_output(lq_compact!, A, alg)

initialize_output(::typeof(left_null!), A, alg::AbstractAlgorithm) =
    initialize_output(left_null_kind(alg), A, alg)
initialize_output(::typeof(left_null_qr!), A, alg::AbstractAlgorithm) =
    initialize_output(qr_null!, A, alg)
initialize_output(::typeof(left_null_svd!), A, alg::AbstractAlgorithm) = nothing

initialize_output(::typeof(right_null!), A, alg::AbstractAlgorithm) =
    initialize_output(right_null_kind(alg), A, alg)
initialize_output(::typeof(right_null_lq!), A, alg::AbstractAlgorithm) =
    initialize_output(lq_null!, A, alg)
initialize_output(::typeof(right_null_svd!), A, alg::AbstractAlgorithm) = nothing

# Outputs
# -------
function initialize_orth_svd(A::AbstractMatrix, F, alg)
    S = Diagonal(initialize_output(svd_vals!, A, alg))
    return F[1], S, F[2]
end
# fallback doesn't re-use F at all
initialize_orth_svd(A, F, alg) = initialize_output(svd_compact!, A, alg)

# Implementation of orth functions
# --------------------------------
left_orth!(A, VC, alg::AbstractAlgorithm) = left_orth_kind(alg)(A, VC, alg)
left_orth_qr!(A, VC, alg::AbstractAlgorithm) = qr_compact!(A, VC, alg)
left_orth_polar!(A, VC, alg::AbstractAlgorithm) = left_polar!(A, VC, alg)

right_orth!(A, CVᴴ, alg::AbstractAlgorithm) = right_orth_kind(alg)(A, CVᴴ, alg)
right_orth_lq!(A, CVᴴ, alg::AbstractAlgorithm) = lq_compact!(A, CVᴴ, alg)
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


left_null!(A, N, alg::AbstractAlgorithm) = left_null_kind(alg)(A, N, alg)
left_null_qr!(A, N, alg::AbstractAlgorithm) = qr_null!(A, N, alg)

right_null!(A, Nᴴ, alg::AbstractAlgorithm) = right_null_kind(alg)(A, Nᴴ, alg)
right_null_lq!(A, Nᴴ, alg::AbstractAlgorithm) = lq_null!(A, Nᴴ, alg)

function left_null_svd!(A, N, alg::TruncatedAlgorithm)
    check_input(left_null_svd!, A, N, alg)
    U, S, _ = svd_full!(A, alg.alg)
    N, _ = truncate(left_null!, (U, S), alg.trunc)
    return N
end
function right_null_svd!(A, Nᴴ, alg::TruncatedAlgorithm)
    check_input(right_null_svd!, A, Nᴴ, alg)
    _, S, Vᴴ = svd_full!(A, alg.alg)
    Nᴴ, _ = truncate(right_null!, (S, Vᴴ), alg.trunc)
    return Nᴴ
end
