# Randomized SVD implementation
# ------------------------------

# Compute a left orthogonal sketch of A via A ≈ Q * (Q' * A), where Q has `size(Ω, 2)` columns.
# Power iterations improve accuracy for matrices with slowly decaying singular values.
function left_orth(A::AbstractMatrix, Ω::AbstractMatrix; maxiter::Integer = 1, alg_qr = nothing)
    @assert size(Ω, 2) ≤ min(size(A)...)
    Y = A * Ω
    Q = similar(Y, (size(A, 1), size(Ω, 2)))
    R = similar(Q, (0, 0))
    Q, _ = qr_compact!(Y, (Q, R); alg = alg_qr)

    for _ in 2:maxiter
        mul!(Ω, A', Q)
        mul!(Y, A, Ω)
        Q, _ = qr_compact!(Y, (Q, R); alg = alg_qr)
    end

    R = Q' * A
    return Q, R
end

# Extract target rank from truncation strategy (best-effort; falls back to min(m, n))
_rsvd_sketch_rank(::TruncationStrategy, m, n) = min(m, n)
_rsvd_sketch_rank(trunc::TruncationByOrder, m, n) = min(trunc.howmany, min(m, n))

function initialize_output(
        ::Union{typeof(svd_trunc!), typeof(svd_trunc_no_error!)},
        A::AbstractMatrix,
        alg::TruncatedAlgorithm{<:RandomizedSVD},
    )
    return initialize_output(svd_compact!, A, alg.alg)
end

function svd_trunc_no_error!(A::AbstractMatrix, USVᴴ, alg::TruncatedAlgorithm{<:RandomizedSVD})
    m, n = size(A)
    rsvd_alg = alg.alg
    oversampling = rsvd_alg.oversampling
    maxiter = rsvd_alg.maxiter
    alg_qr = rsvd_alg.alg_qr

    k_sketch = min(_rsvd_sketch_rank(alg.trunc, m, n) + oversampling, min(m, n))

    Ω = similar(A, (n, k_sketch))
    Random.randn!(Ω)

    Q, R = left_orth(A, Ω; maxiter, alg_qr)

    # Compact SVD of the small k_sketch × n matrix
    U′, S, Vᴴ = svd_compact!(R)

    # Truncate and combine
    (U′tr, Str, Vᴴtr), _ = truncate(svd_trunc!, (U′, S, Vᴴ), alg.trunc)
    U = Q * U′tr

    return U, Str, Vᴴtr
end

function svd_trunc!(A::AbstractMatrix, USVᴴ, alg::TruncatedAlgorithm{<:RandomizedSVD})
    U, S, Vᴴ = svd_trunc_no_error!(A, USVᴴ, alg)
    normA = norm(A)
    normS = norm(diagview(S))
    ϵ = sqrt(max(zero(normA), (normA + normS) * (normA - normS)))
    return U, S, Vᴴ, ϵ
end
