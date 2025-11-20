# Inputs
# ------
copy_input(::typeof(left_orth), A) = copy_input(qr_compact, A) # do we ever need anything else
copy_input(::typeof(right_orth), A) = copy_input(lq_compact, A) # do we ever need anything else
copy_input(::typeof(left_null), A) = copy_input(qr_null, A) # do we ever need anything else
copy_input(::typeof(right_null), A) = copy_input(lq_null, A) # do we ever need anything else

check_input(::typeof(left_orth!), A, VC, alg::AbstractAlgorithm) =
    check_input(left_orth!, A, VC, left_orth_alg(alg))
check_input(::typeof(left_orth!), A, VC, alg::LeftOrthAlgorithm) =
    check_input_left_orth!(A, VC, alg)
check_input_left_orth!(A, VC, alg::LeftOrthViaQR) =
    check_input(qr_compact!, A, VC, alg.alg)
check_input_left_orth!(A, VC, alg::LeftOrthViaPolar) =
    check_input(left_polar!, A, VC, alg.alg)
check_input_left_orth!(A, VC, alg::LeftOrthViaSVD) = nothing

check_input(::typeof(right_orth!), A, CVᴴ, alg::AbstractAlgorithm) =
    check_input(right_orth!, A, CVᴴ, right_orth_alg(alg))
check_input(::typeof(right_orth!), A, CVᴴ, alg::RightOrthAlgorithm) =
    check_input_right_orth!(A, CVᴴ, right_orth_alg(alg))
check_input_right_orth!(A, CVᴴ, alg::RightOrthViaLQ) =
    check_input(lq_compact!, A, CVᴴ, alg.alg)
check_input_right_orth!(A, CVᴴ, alg::RightOrthViaPolar) =
    check_input(right_polar!, A, CVᴴ, alg.alg)
check_input_right_orth!(A, CVᴴ, alg::RightOrthViaSVD) = nothing

check_input(::typeof(left_null!), A, N, alg::AbstractAlgorithm) =
    check_input(left_null!, A, N, left_null_alg(alg))
check_input(::typeof(left_null!), A, N, alg::LeftNullAlgorithm) =
    check_input_left_null!(A, N, alg)
check_input_left_null!(A, N, alg::LeftNullViaQR) =
    check_input(qr_null!, A, N, alg.alg)
check_input_left_null!(A, N, alg::LeftNullViaSVD) = nothing

check_input(::typeof(right_null!), A, Nᴴ, alg::AbstractAlgorithm) =
    check_input(right_null!, A, Nᴴ, right_null_alg(alg))
check_input(::typeof(right_null!), A, Nᴴ, alg::RightNullAlgorithm) =
    check_input_right_null!(A, Nᴴ, alg)
check_input_right_null!(A, Nᴴ, alg::RightNullViaLQ) =
    check_input(lq_null!, A, Nᴴ, alg.alg)
check_input_right_null!(A, Nᴴ, alg::RightNullViaSVD) = nothing

# Outputs
# -------
initialize_output(::typeof(left_orth!), A, alg::AbstractAlgorithm) =
    initialize_output(left_orth!, A, left_orth_alg(alg))
initialize_output(::typeof(left_orth!), A, alg::LeftOrthAlgorithm) =
    initialize_output_left_orth!(A, alg)
initialize_output_left_orth!(A, alg::LeftOrthViaQR) =
    initialize_output(qr_compact!, A, alg.alg)
initialize_output_left_orth!(A, alg::LeftOrthViaPolar) =
    initialize_output(left_polar!, A, alg.alg)
initialize_output_left_orth!(A, alg::LeftOrthViaSVD) = nothing

initialize_output(::typeof(right_orth!), A, alg::AbstractAlgorithm) =
    initialize_output(right_orth!, A, right_orth_alg(alg))
initialize_output(::typeof(right_orth!), A, alg::RightOrthAlgorithm) =
    initialize_output_right_orth!(A, alg)
initialize_output_right_orth!(A, alg::RightOrthViaLQ) =
    initialize_output(lq_compact!, A, alg.alg)
initialize_output_right_orth!(A, alg::RightOrthViaPolar) =
    initialize_output(right_polar!, A, alg.alg)
initialize_output_right_orth!(A, alg::RightOrthViaSVD) = nothing

initialize_output(::typeof(left_null!), A, alg::AbstractAlgorithm) =
    initialize_output(left_null!, A, left_null_alg(alg))
initialize_output(::typeof(left_null!), A, alg::LeftNullAlgorithm) =
    initialize_output_left_null!(A, alg)
initialize_output_left_null!(A, alg::LeftNullViaQR) =
    initialize_output(qr_null!, A, alg.alg)
initialize_output_left_null!(A, alg::LeftNullViaSVD) = nothing

initialize_output(::typeof(right_null!), A, alg::AbstractAlgorithm) =
    initialize_output(right_null!, A, right_null_alg(alg))
initialize_output(::typeof(right_null!), A, alg::RightNullAlgorithm) =
    initialize_output_right_null!(A, alg)
initialize_output_right_null!(A, alg::RightNullViaLQ) =
    initialize_output(lq_null!, A, alg.alg)
initialize_output_right_null!(A, alg::RightNullViaSVD) = nothing

# Implementation of orth functions
# --------------------------------
left_orth!(A, VC, alg::AbstractAlgorithm) = left_orth!(A, VC, left_orth_alg(alg))
left_orth!(A, VC, alg::LeftOrthAlgorithm) = left_orth_via!(A, VC, alg)
left_orth_via!(A, VC, alg::LeftOrthViaQR) = qr_compact!(A, VC, alg.alg)
left_orth_via!(A, VC, alg::LeftOrthViaPolar) = left_polar!(A, VC, alg.alg)
function left_orth_via!(A, VC, alg::LeftOrthViaSVD)
    check_input(left_orth!, A, VC, alg)
    V, S, C = does_truncate(alg.alg) ? svd_trunc!(A, alg.alg) : svd_compact!(A, alg.alg)
    lmul!(S, C)
    return V, C
end

right_orth!(A, CVᴴ, alg::AbstractAlgorithm) = right_orth!(A, CVᴴ, right_orth_alg(alg))
right_orth!(A, CVᴴ, alg::RightOrthAlgorithm) = right_orth_via!(A, CVᴴ, alg)
right_orth_via!(A, CVᴴ, alg::RightOrthViaLQ) = lq_compact!(A, CVᴴ, alg.alg)
right_orth_via!(A, CVᴴ, alg::RightOrthViaPolar) = right_polar!(A, CVᴴ, alg.alg)
function right_orth_via!(A, CVᴴ, alg::RightOrthViaSVD)
    check_input(right_orth!, A, CVᴴ, alg)
    C, S, Vᴴ = does_truncate(alg.alg) ? svd_trunc!(A, alg.alg) : svd_compact!(A, alg.alg)
    rmul!(C, S)
    return C, Vᴴ
end

# Implementation of null functions
# --------------------------------
left_null!(A, N, alg::AbstractAlgorithm) = left_null!(A, N, left_null_alg(alg))
left_null!(A, N, alg::LeftNullAlgorithm) = left_null_via!(A, N, alg)
left_null_via!(A, N, alg::LeftNullViaQR) = qr_null!(A, N, alg.alg)
function left_null_via!(A, N, alg::LeftNullViaSVD{<:TruncatedAlgorithm})
    check_input(left_null!, A, N, alg)
    U, S, _ = svd_full!(A, alg.alg.alg)
    N, _ = truncate(left_null!, (U, S), alg.alg.trunc)
    return N
end

right_null!(A, Nᴴ, alg::AbstractAlgorithm) = right_null!(A, Nᴴ, right_null_alg(alg))
right_null!(A, Nᴴ, alg::RightNullAlgorithm) = right_null_via!(A, Nᴴ, alg)
right_null_via!(A, Nᴴ, alg::RightNullViaLQ) = lq_null!(A, Nᴴ, alg.alg)
function right_null_via!(A, Nᴴ, alg::RightNullViaSVD{<:TruncatedAlgorithm})
    check_input(right_null!, A, Nᴴ, alg)
    _, S, Vᴴ = svd_full!(A, alg.alg.alg)
    Nᴴ, _ = truncate(right_null!, (S, Vᴴ), alg.alg.trunc)
    return Nᴴ
end

# randomized algorithms don't currently work for smallest values:
left_null_via!(A, N, alg::LeftNullViaSVD{<:TruncatedAlgorithm{<:GPU_Randomized}}) =
    throw(ArgumentError("Randomized SVD ($alg) cannot be used for null spaces yet"))
right_null_via!(A, Nᴴ, alg::RightNullViaSVD{<:TruncatedAlgorithm{<:GPU_Randomized}}) =
    throw(ArgumentError("Randomized SVD ($alg) cannot be used for null spaces yet"))
