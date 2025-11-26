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
    m = 54

    A = randn(rng, T, m, m)
    A /= norm(A)

    D, V = @constinferred eig_full(A)
    algs = (MatrixFunctionViaLA(), MatrixFunctionViaEig(LAPACK_Simple()))
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
    @test_throws DomainError exponential(A; alg = MatrixFunctionViaEigh(LAPACK_QRIteration()))
end

@testset "exponentiali! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54

    A = randn(rng, T, m, m)
    τ = randn(rng, T)

    D, V = @constinferred eig_full(A)
    algs = (MatrixFunctionViaLA(), MatrixFunctionViaEig(LAPACK_Simple()))
    expiτA_LA = @constinferred exp(im * τ * A)
    @testset "algorithm $alg" for alg in algs
        expiτA = similar(complex(A))

        @constinferred exponentiali!(τ, copy(A), expiτA; alg)
        expiτA2 = @constinferred exponentiali(τ, A; alg = alg)
        @test expiτA ≈ expiτA_LA
        @test expiτA2 ≈ expiτA

        Dexp, Vexp = @constinferred eig_full(expiτA)
        @test sort(diagview(Dexp); by = imag) ≈ sort(LinearAlgebra.exp.(diagview(D) .* (im * τ)); by = imag)
    end
    @test_throws DomainError exponentiali(τ, A; alg = MatrixFunctionViaEigh(LAPACK_QRIteration()))
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

@testset "exponentiali! for Diagonal{$T}" for T in (BLASFloats..., GenericFloats...)
    rng = StableRNG(123)
    atol = sqrt(eps(real(T)))
    m = 54
    Ad = randn(T, m)
    A = Diagonal(Ad)
    τ = randn(rng, T)

    expiτA = similar(complex(A))
    @constinferred exponentiali!(τ, copy(A), expiτA)
    expiτA2 = @constinferred exponentiali(τ, A; alg = DiagonalAlgorithm())
    @test expiτA2 ≈ expiτA

    D, V = @constinferred eig_full(A)
    Dexp, Vexp = @constinferred eig_full(expiτA)
    @test diagview(Dexp) ≈ LinearAlgebra.exp.(diagview(D) .* (im * τ))
end
