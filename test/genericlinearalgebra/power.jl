using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using MatrixAlgebraKit: diagview
using LinearAlgebra
using GenericLinearAlgebra

GenericFloats = (BigFloat, Complex{BigFloat})

@testset "power! for T = $T" for T in GenericFloats
    rng = StableRNG(123)
    m = 24

    X = randn(rng, T, m, m)
    A = project_hermitian!(X * X') + one(real(T)) * LinearAlgebra.I
    D, V = @constinferred eigh_full(A)

    algs = (MatrixFunctionViaEigh(GLA_QRIteration()),)
    @testset "algorithm $alg" for alg in algs
        @test power(A, 3, alg) ≈ A * A * A
        @test power(A, -1, alg) * A ≈ LinearAlgebra.I
        @test power(A, big"0.5", alg) ≈ squareroot(A; alg)

        powA = @constinferred power(A, big"0.5", alg)
        Dpow, Vpow = @constinferred eigh_full(powA)
        @test diagview(Dpow) ≈ sqrt.(diagview(D))
    end
end
