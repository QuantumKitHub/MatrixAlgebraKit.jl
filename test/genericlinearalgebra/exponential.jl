using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using MatrixAlgebraKit: diagview
using LinearAlgebra

GenericFloats = (BigFloat, Complex{BigFloat})

@testset "exponential! for T = $T" for T in GenericFloats
    rng = StableRNG(123)
    m = 54

    A = randn(rng, T, m, m)
    A = (A + A') / 2
    D, V = @constinferred eigh_full(A)
    algs = (MatrixFunctionViaEigh(GLA_QRIteration()),)
    @testset "algorithm $alg" for alg in algs
        expA = similar(A)

        @constinferred exponential!(copy(A), expA; alg)
        expA2 = @constinferred exponential(A; alg)
        @test expA2 ≈ expA

        Dexp, Vexp = @constinferred eigh_full(expA)
        @test diagview(Dexp) ≈ LinearAlgebra.exp.(diagview(D))
    end
end

using GenericSchur
@testset "exponentiali! for T = $T" for T in GenericFloats
    rng = StableRNG(123)
    m = 2

    A = randn(rng, T, m, m)
    A = (A + A') / 2
    τ = randn(rng, T)

    D, V = @constinferred eigh_full(A)
    algs = (MatrixFunctionViaEigh(GLA_QRIteration()),)
    @testset "algorithm $alg" for alg in algs
        expiτA = similar(complex(A))

        @constinferred exponentiali!(τ, copy(A), expiτA; alg)
        expiτA2 = @constinferred exponentiali(τ, A; alg)
        @test expiτA2 ≈ expiτA

        Dexp, Vexp = @constinferred eig_full(expiτA)
        @test sort(diagview(Dexp); by = imag) ≈ sort(LinearAlgebra.exp.(diagview(D) .* (im*τ)); by = imag)
    end
end
