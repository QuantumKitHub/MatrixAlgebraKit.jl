using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using MatrixAlgebraKit: diagview
using LinearAlgebra
using LinearAlgebra: exp

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
GenericFloats = (Float16, ComplexF16, BigFloat, Complex{BigFloat})

@testset "exponential! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54

    A = LinearAlgebra.normalize!(randn(rng, T, m, m))
    Ac = copy(A)
    expA = LinearAlgebra.exp(A)

    expA2 = @constinferred exponential(A)
    @test expA ≈ expA2
    @test A == Ac

    algs = (MatrixFunctionViaLA(), MatrixFunctionViaEig(LAPACK_Simple()))
    @testset "algorithm $alg" for alg in algs
        expA2 = @constinferred exponential(A, alg)
        @test expA ≈ expA2
        @test A == Ac
    end

    @test_throws DomainError exponential(A; alg = MatrixFunctionViaEigh(LAPACK_QRIteration()))
end

@testset "exponential! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54

    A = randn(rng, T, m, m)
    τ = randn(rng, T)
    Ac = copy(A)

    Aτ = A * τ
    expAτ = LinearAlgebra.exp(Aτ)

    expAτ2 = @constinferred exponential((τ, A))
    @test expAτ ≈ expAτ2
    @test A == Ac

    algs = (MatrixFunctionViaLA(), MatrixFunctionViaEig(LAPACK_Simple()))
    @testset "algorithm $alg" for alg in algs
        expAτ2 = @constinferred exponential((τ, A), alg)
        @test expAτ ≈ expAτ2
        @test A == Ac
    end

    @test_throws DomainError exponential((τ, A); alg = MatrixFunctionViaEigh(LAPACK_QRIteration()))
end

@testset "exponential! for Diagonal{$T}" for T in (BLASFloats..., GenericFloats...)
    rng = StableRNG(123)
    m = 54

    A = Diagonal(randn(rng, T, m))
    τ = randn(rng, T)
    Ac = copy(A)

    expA = LinearAlgebra.exp(A)

    expA2 = @constinferred exponential(A)
    @test expA ≈ expA2
    @test A == Ac
end

@testset "exponential! for Diagonal{$T}" for T in (BLASFloats..., GenericFloats...)
    rng = StableRNG(123)
    m = 1

    A = Diagonal(randn(rng, T, m))
    τ = randn(rng, T)
    Ac = copy(A)

    Aτ = A * τ
    expAτ = LinearAlgebra.exp(Aτ)

    expAτ2 = @constinferred exponential((τ, A))
    @test expAτ ≈ expAτ2
    @test A == Ac
end
