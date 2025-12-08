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

@testset "exponentiali! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54

    A = randn(rng, T, m, m)
    τ = randn(rng, T)
    Ac = copy(A)

    Aimτ = A * im * τ
    expAimτ = LinearAlgebra.exp(Aimτ)

    expAimτ2 = @constinferred exponentiali(τ, A)
    @test expAimτ ≈ expAimτ2
    @test A == Ac

    algs = (MatrixFunctionViaLA(), MatrixFunctionViaEig(LAPACK_Simple()))
    @testset "algorithm $alg" for alg in algs
        expAimτ2 = @constinferred exponentiali(τ, A, alg)
        @test expAimτ ≈ expAimτ2
        @test A == Ac
    end

    @test_throws DomainError exponentiali(τ, A; alg = MatrixFunctionViaEigh(LAPACK_QRIteration()))
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

@testset "exponentiali! for Diagonal{$T}" for T in (BLASFloats..., GenericFloats...)
    rng = StableRNG(123)
    m = 54

    A = Diagonal(randn(rng, T, m))
    τ = randn(rng, T)
    Ac = copy(A)

    Aimτ = A * im * τ
    expAimτ = LinearAlgebra.exp(Aimτ)

    expAimτ2 = @constinferred exponentiali(τ, A)
    @test expAimτ ≈ expAimτ2
    @test A == Ac
end
