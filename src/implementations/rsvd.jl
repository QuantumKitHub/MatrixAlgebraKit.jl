function rand_range_finder(A::AbstractMatrix, Ω::AbstractMatrix; maxiter::Integer = 1, alg_qr = nothing)
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


    return Q
end

function rand_svd(A, k::Integer; trunc = truncrank(k), p::Integer = 10, maxiter::Integer = 1)
    m, n = size(A)
    @assert p <= min(m, n)
    Ω = Random.randn!(similar(A, (m, k + p)))
    Q = rand_range_finder(A, Ω; maxiter)
    B = Q' * A

    U′, S, Vᴴ = svd_trunc!(B; trunc)
    U = Q * U′

    return U, S, Vᴴ
end
