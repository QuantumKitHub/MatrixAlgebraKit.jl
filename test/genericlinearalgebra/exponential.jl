using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using MatrixAlgebraKit: diagview
using LinearAlgebra
using GenericLinearAlgebra

GenericFloats = (BigFloat, Complex{BigFloat})

@testset "exponential! for T = $T" for T in GenericFloats
    rng = StableRNG(123)
    m = 54

    A = project_hermitian!(randn(rng, T, m, m))
    D, V = @constinferred eigh_full(A)
    algs = (MatrixFunctionViaEigh(GLA_QRIteration()),)
    @testset "algorithm $alg" for alg in algs
        expA = @constinferred exponential!(copy(A); alg)
        expA2 = @constinferred exponential(A; alg)
        @test expA2 ≈ expA

        Dexp, Vexp = @constinferred eigh_full(expA)
        @test diagview(Dexp) ≈ LinearAlgebra.exp.(diagview(D))
    end
end

using GenericSchur
@testset "exponential! for T1 = $T1, T2 = $T2" for T1 in GenericFloats, T2 in GenericFloats
    rng = StableRNG(123)
    m = 54
    A = project_hermitian!(randn(rng, T1, m, m))
    τ = randn(rng, T2)

    D, V = @constinferred eigh_full(A)
    algs = (MatrixFunctionViaEigh(GLA_QRIteration()),)
    @testset "algorithm $alg" for alg in algs
        expτA = @constinferred exponential!((τ, copy(A)); alg)
        expτA2 = @constinferred exponential((τ, A); alg)
        @test expτA2 ≈ expτA

        Dexp, Vexp = @constinferred eig_full(expτA)

        @test sort(diagview(Dexp); by = real) ≈ sort(LinearAlgebra.exp.(diagview(D) .* τ); by = real)
    end
end
