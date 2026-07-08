using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using MatrixAlgebraKit: diagview
using LinearAlgebra
using GenericLinearAlgebra

GenericFloats = (BigFloat, Complex{BigFloat})

@testset "squareroot! for T = $T" for T in GenericFloats
    rng = StableRNG(123)
    m = 24

    X = randn(rng, T, m, m)
    A = project_hermitian!(X * X') + one(real(T)) * LinearAlgebra.I
    D, V = @constinferred eigh_full(A)

    algs = (MatrixFunctionViaEigh(GLA_QRIteration()),)
    @testset "algorithm $alg" for alg in algs
        sqrtA = @constinferred squareroot!(copy(A); alg)
        sqrtA2 = @constinferred squareroot(A; alg)
        @test sqrtA2 ≈ sqrtA
        @test sqrtA * sqrtA ≈ A
        @test LinearAlgebra.ishermitian(sqrtA)

        Dsqrt, Vsqrt = @constinferred eigh_full(sqrtA)
        @test diagview(Dsqrt) ≈ sqrt.(diagview(D))
    end
end
