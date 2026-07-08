using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using MatrixAlgebraKit: diagview
using LinearAlgebra
using GenericLinearAlgebra

GenericFloats = (BigFloat, Complex{BigFloat})

@testset "logarithm! for T = $T" for T in GenericFloats
    rng = StableRNG(123)
    m = 24

    X = randn(rng, T, m, m)
    A = project_hermitian!(X * X') + one(real(T)) * LinearAlgebra.I
    D, V = @constinferred eigh_full(A)

    algs = (MatrixFunctionViaEigh(GLA_QRIteration()),)
    @testset "algorithm $alg" for alg in algs
        logA = @constinferred logarithm!(copy(A); alg)
        logA2 = @constinferred logarithm(A; alg)
        @test logA2 ≈ logA

        Dlog, Vlog = @constinferred eigh_full(logA)
        @test diagview(Dlog) ≈ log.(diagview(D))
    end
end
