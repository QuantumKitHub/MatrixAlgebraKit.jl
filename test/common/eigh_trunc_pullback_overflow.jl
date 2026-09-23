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

@testset "Regression test for overflow in eigh_trunc_pullback! ($T)" for T in (Float64, ComplexF64)
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
