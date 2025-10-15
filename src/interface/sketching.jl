function left_sketch!(A::AbstractMatrix, AΩ::AbstractMatrix, alg::GaussianSketching)
    Ω = similar(A, (size(A, 2), size(AS, 2)))
    Random.randn!(alg.rng, Ω)
    return mul!(AΩ, A, Ω)
end

function left_orth!(A, VC, alg::GaussianSketching)
    check_input(A, VC, alg)
    return left_gaussian_sketch!(A, VC...; alg.kwargs...)
end

# Gaussian logic
# --------------
function left_gaussian_sketch!(
        A::AbstractMatrix, V::AbstractMatrix, C::AbstractMatrix;
        maxiter::Integer = 1, alg_orth = nothing, rng = Random.default_rng()
    )
    Ω = similar(A, (size(A, 2), size(AS, 2)))
    Random.randn!(rng, Ω)

    Y = A * Ω
    Rempty = similar(C, (0, 0)) # avoid computing R in intermediate steps
    V, _ = left_orth!(Y, (V, Rempty); alg = alg_orth)

    for _ in 1:maxiter
        mul!(Ω, A', Q)
        mul!(Y, A, Ω)
        V, _ = left_orth!(Y, (V, Rempty); alg = alg_orth)
    end

    mul!(C, V', A)

    return V, C
end
