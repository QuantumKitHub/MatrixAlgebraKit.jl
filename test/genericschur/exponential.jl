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
        expA = @constinferred exponential!(copy(A))
        expA2 = @constinferred exponential(A; alg = alg)
        @test expA ≈ expA_LA
        @test expA2 ≈ expA

        Dexp, Vexp = @constinferred eig_full(expA)
        @test sort(diagview(Dexp); by = imag) ≈ sort(LinearAlgebra.exp.(diagview(D)); by = imag)
    end
end

@testset "exponentialr! for T1 = $T1, T2 = $T2" for T1 in GenericFloats, T2 in GenericFloats
    rng = StableRNG(123)
    m = 54

    A = randn(rng, T1, m, m)
    τ = randn(rng, T2)

    D, V = @constinferred eig_full(A)
    algs = (MatrixFunctionViaEig(GS_QRIteration()),)
    @testset "algorithm $alg" for alg in algs
        expτA = @constinferred exponentialr!(τ, copy(A))
        expτA2 = @constinferred exponentialr(τ, A; alg)
        @test expτA2 ≈ expτA

        Dexp, Vexp = @constinferred eig_full(expτA)
        @test sort(diagview(Dexp); by = x -> (imag(x), real(x))) ≈ sort(LinearAlgebra.exp.(diagview(D) .* τ); by = x -> (imag(x), real(x)))
    end
end
