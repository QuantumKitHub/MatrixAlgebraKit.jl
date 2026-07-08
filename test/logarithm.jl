using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using MatrixAlgebraKit: diagview
using LinearAlgebra
using LinearAlgebra: exp

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
GenericFloats = (Float16, ComplexF16, BigFloat, Complex{BigFloat})

@testset "logarithm! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54

    # spectrum inside a disk around 1, away from the negative real axis and zero
    A = LinearAlgebra.I + LinearAlgebra.normalize!(randn(rng, T, m, m))
    Ac = copy(A)
    logA = LinearAlgebra.log(A)

    logA2 = @constinferred logarithm(A)
    @test logA ≈ logA2
    @test exp(logA2) ≈ A
    @test A == Ac

    algs = (MatrixFunctionViaLA(), MatrixFunctionViaEig(LAPACK_Simple()))
    @testset "algorithm $alg" for alg in algs
        logA2 = @constinferred logarithm(A, alg)
        @test logA ≈ logA2
        @test A == Ac
    end

    @test_throws DomainError logarithm(A; alg = MatrixFunctionViaEigh(LAPACK_QRIteration()))

    # roundtrip with exponential
    @test logarithm(exponential(logA)) ≈ logA
end

@testset "logarithm! for hermitian T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54

    X = randn(rng, T, m, m)
    A = Matrix(LinearAlgebra.Hermitian(X * X' + one(real(T)) * LinearAlgebra.I))
    Ac = copy(A)
    logA = LinearAlgebra.log(LinearAlgebra.Hermitian(A))

    algs = (
        MatrixFunctionViaLA(), MatrixFunctionViaEigh(LAPACK_QRIteration()),
        MatrixFunctionViaEig(LAPACK_Simple()),
    )
    @testset "algorithm $alg" for alg in algs
        logA2 = @constinferred logarithm(A, alg)
        @test logA ≈ logA2
        @test A == Ac
    end
end

@testset "logarithm! domain handling for T = $T" for T in (Float32, Float64)
    rng = StableRNG(123)
    m = 4

    X = randn(rng, T, m, m)
    V = Matrix(LinearAlgebra.qr(X).Q)
    A = V * LinearAlgebra.Diagonal(T[-1, 1, 2, 3]) * V'
    A = (A + A') / 2

    # negative eigenvalue: DomainError for real input, complex principal value otherwise
    @test_throws DomainError logarithm(A)
    @test_throws DomainError logarithm(A; alg = MatrixFunctionViaEigh(LAPACK_QRIteration()))
    @test_throws DomainError logarithm(A; alg = MatrixFunctionViaEig(LAPACK_Simple()))

    logA = @constinferred logarithm(complex.(A))
    @test exp(logA) ≈ A

    # (numerically) zero eigenvalue: no logarithm exists
    Asing = V * LinearAlgebra.Diagonal(T[0, 1, 2, 3]) * V'
    Asing = (Asing + Asing') / 2
    @test_throws DomainError logarithm(Asing; alg = MatrixFunctionViaEigh(LAPACK_QRIteration()))
    @test_throws DomainError logarithm(Asing; alg = MatrixFunctionViaEig(LAPACK_Simple()))
    @test_throws DomainError logarithm(complex.(Asing); alg = MatrixFunctionViaEig(LAPACK_Simple()))
end

@testset "logarithm! for Diagonal{$T}" for T in (BLASFloats..., GenericFloats...)
    rng = StableRNG(123)
    m = 54

    data = T <: Real ? (abs.(randn(rng, T, m)) .+ one(T)) : (randn(rng, T, m) .+ 4 * one(T))
    A = Diagonal(data)
    Ac = copy(A)
    logA = LinearAlgebra.log(A)

    logA2 = @constinferred logarithm(A)
    @test logA2 isa Diagonal
    @test logA ≈ logA2
    @test A == Ac

    if T <: Real
        @test_throws DomainError logarithm(Diagonal(-data))
    end
    @test_throws DomainError logarithm(Diagonal(zeros(T, m)))
end
