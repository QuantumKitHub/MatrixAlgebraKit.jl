using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using MatrixAlgebraKit: diagview
using LinearAlgebra

GenericFloats = (BigFloat, Complex{BigFloat})

@testset "exp! for T = $T" for T in GenericFloats
    rng = StableRNG(123)
    m = 2

    A = randn(rng, T, m, m)
    D, V = @constinferred eig_full(A)
    algs = (ExponentialViaEig(GS_QRIteration()),)
    expA_LA = @constinferred exponential(A)
    @testset "algorithm $alg" for alg in algs
        expA = similar(A)

        @constinferred exponential!(copy(A), expA)
        expA2 = @constinferred exponential(A; alg = alg)
        @test expA ≈ expA_LA
        @test expA2 ≈ expA

        Dexp, Vexp = @constinferred eig_full(expA)
        @test diagview(Dexp) ≈ LinearAlgebra.exp.(diagview(D))
    end
end
