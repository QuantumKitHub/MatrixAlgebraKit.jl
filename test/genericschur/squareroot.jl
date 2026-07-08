using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using MatrixAlgebraKit: diagview
using LinearAlgebra
using GenericSchur

GenericFloats = (BigFloat, Complex{BigFloat})

@testset "squareroot! for T = $T" for T in GenericFloats
    rng = StableRNG(123)
    m = 24

    # spectrum inside a disk around 1, away from the negative real axis
    A = LinearAlgebra.I + LinearAlgebra.normalize!(randn(rng, T, m, m))
    D, V = @constinferred eig_full(A)

    sqrtA = @constinferred squareroot(A)
    @test sqrtA * sqrtA ≈ A
    @test eltype(sqrtA) == T

    algs = (MatrixFunctionViaEig(GS_QRIteration()),)
    @testset "algorithm $alg" for alg in algs
        sqrtA2 = @constinferred squareroot(A; alg)
        @test sqrtA2 ≈ sqrtA

        Dsqrt, Vsqrt = @constinferred eig_full(sqrtA2)
        by = x -> (real(x), imag(x))
        @test sort(diagview(Dsqrt); by) ≈ sort(sqrt.(diagview(D)); by)
    end
end
