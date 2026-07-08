using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using MatrixAlgebraKit: diagview
using LinearAlgebra
using GenericSchur

GenericFloats = (BigFloat, Complex{BigFloat})

@testset "logarithm! for T = $T" for T in GenericFloats
    rng = StableRNG(123)
    m = 24

    # spectrum inside a disk around 1, away from the negative real axis and zero
    A = LinearAlgebra.I + LinearAlgebra.normalize!(randn(rng, T, m, m))
    D, V = @constinferred eig_full(A)

    logA = @constinferred logarithm(A)
    @test eltype(logA) == T
    @test exponential(logA) ≈ A

    algs = (MatrixFunctionViaEig(GS_QRIteration()),)
    @testset "algorithm $alg" for alg in algs
        logA2 = @constinferred logarithm(A; alg)
        @test logA2 ≈ logA

        Dlog, Vlog = @constinferred eig_full(logA2)
        by = x -> (real(x), imag(x))
        @test sort(diagview(Dlog); by) ≈ sort(log.(diagview(D)); by)
    end
end
