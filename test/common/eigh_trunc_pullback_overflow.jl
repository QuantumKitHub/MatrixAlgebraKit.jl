using MatrixAlgebraKit
using MatrixAlgebraKit: eigh_pullback!, eigh_trunc_pullback!, diagview
using Test
using StableRNGs
using LinearAlgebra: Diagonal, norm, qr

# Regression test for the Sylvester doubling iteration in `eigh_trunc_pullback!`,
# which previously gave rise to overflow and underflow issues when the gap between the
# absolute values of the largest discarded and the smallest retained eigenvalues
# was small, but the gap between the absolute values of the largest and smallest retained
# eigenvalues was large.

@testset "Regression test for eigh_trunc_pullback! with rho close to 1 ($T)" for T in (Float64, ComplexF64)
    rng = StableRNG(12345)
    n, p = 24, 6
    # construct artifical spectrum with |λ_discarded|max / |λ_kept|min close to 1, but
    # |λ_kept|max / |λ_kept|min large
    λ = vcat([30.0, -30.0, 5.0, -5.0, 1.0, -1.0], collect(range(0.95, 0.1; length = n - p)))
    Q = Matrix(qr(randn(rng, T, n, n)).Q)
    A = Q * Diagonal(T.(λ)) * Q'
    A = (A + A') / 2

    V, D = Q[:, 1:p], real(T).(λ[1:p])       # truncated factors taken straight from the
    Dmat = Diagonal(D)                        # construction: no phase/ordering ambiguity
    @test 0.9 < maximum(abs, λ[(p + 1):end]) / minimum(abs, D) < 1

    ΔD = randn(rng, real(T), p)
    ΔV = randn(rng, T, n, p)
    ΔV .-= V * Diagonal(diagview(V' * ΔV))    # drop the gauge-sensitive component

    ΔA = eigh_trunc_pullback!(zeros(T, n, n), A, (Dmat, V), (Diagonal(ΔD), copy(ΔV)))
    @test all(isfinite, ΔA)
    @test ΔA ≈ ΔA'

    # reference: the same cotangents through the full pullback, which has no series in it
    ΔA_full = eigh_pullback!(
        zeros(T, n, n), A, (Diagonal(real(T).(λ)), Q),
        (Diagonal(vcat(ΔD, zeros(real(T), n - p))), hcat(copy(ΔV), zeros(T, n, n - p)))
    )
    @test ΔA ≈ ΔA_full rtol = 1.0e-8
end

# The final assembly recycles the `AP` buffer with `mul!(AP, Z, V', 1, α)`. `APₖ₊₁` aliases
# that buffer, so it must be OVERWRITTEN (α = 0), not accumulated into. With a well
# separated truncation the loop converges at k = 1, so no squaring has been written yet and
# the buffer still holds the original `AP`; accumulating adds a spurious term of relative
# size |λ_discarded|max / |λ|max.
@testset "eigh_trunc_pullback! does not accumulate into the recycled AP buffer ($T)" for T in
    (Float64, ComplexF64)

    rng = StableRNG(12345)
    n, p = 24, 6
    λ = vcat([30.0, -30.0, 5.0, -5.0, 1.0, -1.0], collect(range(1.0e-6, 1.0e-7; length = n - p)))
    Q = Matrix(qr(randn(rng, T, n, n)).Q)
    A = Q * Diagonal(T.(λ)) * Q'
    A = (A + A') / 2

    V, D = Q[:, 1:p], real(T).(λ[1:p])
    ΔD = randn(rng, real(T), p)
    ΔV = randn(rng, T, n, p)
    ΔV .-= V * Diagonal(diagview(V' * ΔV))

    ΔA = eigh_trunc_pullback!(zeros(T, n, n), A, (Diagonal(D), V), (Diagonal(ΔD), copy(ΔV)))
    ΔA_full = eigh_pullback!(
        zeros(T, n, n), A, (Diagonal(real(T).(λ)), Q),
        (Diagonal(vcat(ΔD, zeros(real(T), n - p))), hcat(copy(ΔV), zeros(T, n, n - p)))
    )
    # the spurious term would be ~3e-8 relative, far above this
    @test ΔA ≈ ΔA_full rtol = 1.0e-10
end
