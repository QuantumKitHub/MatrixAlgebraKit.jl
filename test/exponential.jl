using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using MatrixAlgebraKit: diagview
using LinearAlgebra

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
GenericFloats = (Float16, ComplexF16, BigFloat, Complex{BigFloat})

@testset "exponential! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 2

    A = randn(rng, T, m, m)
    A = (A + A') / 2
    D, V = @constinferred eigh_full(A)
    algs = (ExponentialViaLA(), ExponentialViaEig(LAPACK_Simple()), ExponentialViaEigh(LAPACK_QRIteration()))
    expA_LA = @constinferred exp(A)
    @testset "algorithm $alg" for alg in algs
        expA = similar(A)

        @constinferred exponential!(copy(A), expA)
        expA2 = @constinferred exponential(A; alg = alg)
        @test expA ≈ expA_LA
        @test expA2 ≈ expA

        Dexp, Vexp = @constinferred eigh_full(expA)
        @test diagview(Dexp) ≈ LinearAlgebra.exp.(diagview(D))
    end
end

@testset "exponential! for Diagonal{$T}" for T in (BLASFloats..., GenericFloats...)
    rng = StableRNG(123)
    atol = sqrt(eps(real(T)))
    m = 54
    Ad = randn(T, m)
    A = Diagonal(Ad)

    expA = similar(A)
    @constinferred exponential!(copy(A), expA)
    expA2 = @constinferred exponential(A; alg = DiagonalAlgorithm())
    @test expA2 ≈ expA

    D, V = @constinferred eig_full(A)
    Dexp, Vexp = @constinferred eig_full(expA)
    @test diagview(Dexp) ≈ LinearAlgebra.exp.(diagview(D))
end
