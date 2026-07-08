using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using MatrixAlgebraKit: diagview
using LinearAlgebra
using GenericSchur

GenericFloats = (BigFloat, Complex{BigFloat})

@testset "power! for T = $T" for T in GenericFloats
    rng = StableRNG(123)
    m = 24

    # spectrum inside a disk around 1, away from the negative real axis and zero
    A = LinearAlgebra.I + LinearAlgebra.normalize!(randn(rng, T, m, m))

    powA = @constinferred power(A, 3)
    @test powA ≈ A * A * A
    @test eltype(powA) == T

    powA = @constinferred power(A, big"0.5")
    @test powA ≈ squareroot(A)

    algs = (MatrixFunctionViaEig(GS_QRIteration()),)
    @testset "algorithm $alg" for alg in algs
        @test power(A, 3, alg) ≈ A * A * A
        @test power(A, -1, alg) * A ≈ LinearAlgebra.I
        @test power(A, big"0.5", alg) ≈ squareroot(A)
    end
end
