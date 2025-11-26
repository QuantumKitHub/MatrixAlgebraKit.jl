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
    D, V = @constinferred eig_full(A)
    algs = (MatrixFunctionViaEig(GS_QRIteration()),)
    expA_LA = @constinferred exponential(A)
    @testset "algorithm $alg" for alg in algs
        expA = similar(A)

        @constinferred exponential!(copy(A), expA)
        expA2 = @constinferred exponential(A; alg = alg)
        @test expA ≈ expA_LA
        @test expA2 ≈ expA

        Dexp, Vexp = @constinferred eig_full(expA)
        @test sort(diagview(Dexp); by = imag) ≈ sort(LinearAlgebra.exp.(diagview(D)); by = imag)
    end
end

@testset "exponentiali! for T = $T" for T in GenericFloats
    rng = StableRNG(123)
    m = 54

    A = randn(rng, T, m, m)
    τ = randn(rng, T)

    D, V = @constinferred eig_full(A)
    algs = (MatrixFunctionViaEig(GS_QRIteration()),)
    @testset "algorithm $alg" for alg in algs
        expiτA = similar(complex(A))

        @constinferred exponentiali!(τ, copy(A), expiτA)
        expiτA2 = @constinferred exponentiali(τ, A; alg)
        @test expiτA2 ≈ expiτA

        Dexp, Vexp = @constinferred eig_full(expiτA)
        @test sort(diagview(Dexp); by = imag) ≈ sort(LinearAlgebra.exp.(diagview(D) .* (im*τ)); by = imag)
    end
end
