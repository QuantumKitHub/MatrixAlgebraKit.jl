using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using MatrixAlgebraKit: diagview
using LinearAlgebra

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
GenericFloats = (Float16, ComplexF16, BigFloat, Complex{BigFloat})

@testset "power! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54

    # spectrum inside a disk around 1, away from the negative real axis and zero
    A = LinearAlgebra.I + LinearAlgebra.normalize!(randn(rng, T, m, m))
    Ac = copy(A)

    # integer powers
    @testset "integer p = $p" for p in (0, 1, 3, -2)
        powA = A^p
        powA2 = @constinferred power(A, p)
        @test powA ≈ powA2
        @test A == Ac

        algs = (MatrixFunctionViaLA(), MatrixFunctionViaEig(LAPACK_Simple()))
        @testset "algorithm $alg" for alg in algs
            powA2 = @constinferred power(A, p, alg)
            @test powA ≈ powA2
            @test A == Ac
        end
    end

    # fractional powers
    @testset "fractional p = $p" for p in (0.5, 0.75, -0.25, 3.5)
        powA = A^p
        powA2 = @constinferred power(A, p)
        @test powA ≈ powA2
        @test A == Ac

        algs = (MatrixFunctionViaLA(), MatrixFunctionViaEig(LAPACK_Simple()))
        @testset "algorithm $alg" for alg in algs
            powA2 = @constinferred power(A, p, alg)
            @test powA ≈ powA2
            @test A == Ac
        end
    end

    @test power(A, 0.5) ≈ squareroot(A)
    @test power(A, 3.0) ≈ power(A, 3)
    @test_throws DomainError power(A, 0.5; alg = MatrixFunctionViaEigh(LAPACK_QRIteration()))
end

@testset "power! for hermitian T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54

    X = randn(rng, T, m, m)
    A = Matrix(LinearAlgebra.Hermitian(X * X' + one(real(T)) * LinearAlgebra.I))
    Ac = copy(A)

    algs = (
        MatrixFunctionViaLA(), MatrixFunctionViaEigh(LAPACK_QRIteration()),
        MatrixFunctionViaEig(LAPACK_Simple()),
    )
    @testset "p = $p, algorithm $alg" for p in (2, -1, 0.5, -0.5), alg in algs
        powA = LinearAlgebra.Hermitian(A)^p
        powA2 = @constinferred power(A, p, alg)
        @test powA ≈ powA2
        @test A == Ac
    end

    @test LinearAlgebra.ishermitian(power(A, 0.5, MatrixFunctionViaEigh(LAPACK_QRIteration())))
end

@testset "power! domain handling for T = $T" for T in (Float32, Float64)
    rng = StableRNG(123)
    m = 4

    X = randn(rng, T, m, m)
    V = Matrix(LinearAlgebra.qr(X).Q)
    A = V * LinearAlgebra.Diagonal(T[-1, 1, 2, 3]) * V'
    A = (A + A') / 2

    # integer powers of matrices with negative eigenvalues are fine
    @test power(A, 3) ≈ A * A * A
    @test power(A, 3, MatrixFunctionViaEigh(LAPACK_QRIteration())) ≈ A * A * A
    @test power(A, 3, MatrixFunctionViaEig(LAPACK_Simple())) ≈ A * A * A

    # fractional powers require staying off the negative real axis for real input
    @test_throws DomainError power(A, 0.5)
    @test_throws DomainError power(A, 0.5, MatrixFunctionViaEigh(LAPACK_QRIteration()))
    @test_throws DomainError power(A, 0.5, MatrixFunctionViaEig(LAPACK_Simple()))
    powA = @constinferred power(complex.(A), 0.5)
    @test powA * powA ≈ A

    # (numerically) singular matrices with negative fractional powers
    Asing = V * LinearAlgebra.Diagonal(T[0, 1, 2, 3]) * V'
    Asing = (Asing + Asing') / 2
    @test_throws DomainError power(Asing, -0.5, MatrixFunctionViaEigh(LAPACK_QRIteration()))
    @test_throws DomainError power(Asing, -0.5, MatrixFunctionViaEig(LAPACK_Simple()))
end

@testset "power! for Diagonal{$T}" for T in (BLASFloats..., GenericFloats...)
    rng = StableRNG(123)
    m = 54

    data = T <: Real ? (abs.(randn(rng, T, m)) .+ one(T)) : (randn(rng, T, m) .+ 4 * one(T))
    A = Diagonal(data)
    Ac = copy(A)

    @testset "p = $p" for p in (2, -1, 0.5)
        powA = A^p
        powA2 = @constinferred power(A, p)
        @test powA2 isa Diagonal
        @test powA ≈ powA2
        @test A == Ac
    end

    if T <: Real
        @test_throws DomainError power(Diagonal(-data), 0.5)
        @test power(Diagonal(-data), 2) ≈ Diagonal(data .^ 2)
    end
    @test_throws SingularException power(Diagonal(zeros(T, m)), -1)
    @test_throws DomainError power(Diagonal(zeros(T, m)), -0.5)
end
