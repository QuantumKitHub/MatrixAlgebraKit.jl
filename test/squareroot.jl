using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using MatrixAlgebraKit: diagview
using LinearAlgebra

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
GenericFloats = (Float16, ComplexF16, BigFloat, Complex{BigFloat})

@testset "squareroot! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54

    # spectrum inside a disk around 1, away from the negative real axis
    A = LinearAlgebra.I + LinearAlgebra.normalize!(randn(rng, T, m, m))
    Ac = copy(A)
    sqrtA = LinearAlgebra.sqrt(A)

    sqrtA2 = @constinferred squareroot(A)
    @test sqrtA ≈ sqrtA2
    @test sqrtA2 * sqrtA2 ≈ A
    @test A == Ac

    algs = (MatrixFunctionViaLA(), MatrixFunctionViaEig(LAPACK_Simple()))
    @testset "algorithm $alg" for alg in algs
        sqrtA2 = @constinferred squareroot(A, alg)
        @test sqrtA ≈ sqrtA2
        @test A == Ac
    end

    @test_throws DomainError squareroot(A; alg = MatrixFunctionViaEigh(LAPACK_QRIteration()))
end

@testset "squareroot! for hermitian T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54

    X = randn(rng, T, m, m)
    A = Matrix(LinearAlgebra.Hermitian(X * X' + one(real(T)) * LinearAlgebra.I))
    Ac = copy(A)
    sqrtA = LinearAlgebra.sqrt(LinearAlgebra.Hermitian(A))

    algs = (
        MatrixFunctionViaLA(), MatrixFunctionViaEigh(LAPACK_QRIteration()),
        MatrixFunctionViaEig(LAPACK_Simple()),
    )
    @testset "algorithm $alg" for alg in algs
        sqrtA2 = @constinferred squareroot(A, alg)
        @test sqrtA ≈ sqrtA2
        @test A == Ac
    end

    @test LinearAlgebra.ishermitian(squareroot(A, MatrixFunctionViaEigh(LAPACK_QRIteration())))
end

@testset "squareroot! domain handling for T = $T" for T in (Float32, Float64)
    rng = StableRNG(123)
    m = 4

    # genuinely negative eigenvalue: DomainError for real input, complex principal value otherwise
    X = randn(rng, T, m, m)
    V = Matrix(LinearAlgebra.qr(X).Q)
    A = V * LinearAlgebra.Diagonal(T[-1, 1, 2, 3]) * V'
    A = (A + A') / 2

    @test_throws DomainError squareroot(A)
    @test_throws DomainError squareroot(A; alg = MatrixFunctionViaEigh(LAPACK_QRIteration()))
    @test_throws DomainError squareroot(A; alg = MatrixFunctionViaEig(LAPACK_Simple()))

    sqrtA = @constinferred squareroot(complex.(A))
    @test sqrtA * sqrtA ≈ A

    # roundoff-scale negative eigenvalue: clamped to zero instead of throwing
    Aclamp = V * LinearAlgebra.Diagonal(T[-10 * eps(T), 1, 2, 3]) * V'
    Aclamp = (Aclamp + Aclamp') / 2
    @testset "algorithm $alg" for alg in (
            MatrixFunctionViaEigh(LAPACK_QRIteration()),
            MatrixFunctionViaEig(LAPACK_Simple()),
        )
        sqrtA2 = @constinferred squareroot(Aclamp, alg)
        @test eltype(sqrtA2) == T
        @test sqrtA2 * sqrtA2 ≈ Aclamp atol = sqrt(eps(T))
    end
end

@testset "squareroot! for Diagonal{$T}" for T in (BLASFloats..., GenericFloats...)
    rng = StableRNG(123)
    m = 54

    data = T <: Real ? (abs.(randn(rng, T, m)) .+ one(T)) : randn(rng, T, m)
    A = Diagonal(data)
    Ac = copy(A)
    sqrtA = LinearAlgebra.sqrt(A)

    sqrtA2 = @constinferred squareroot(A)
    @test sqrtA2 isa Diagonal
    @test sqrtA ≈ sqrtA2
    @test A == Ac

    if T <: Real
        @test_throws DomainError squareroot(Diagonal(-data))
        # roundoff-scale negative entries are clamped
        @test squareroot(Diagonal(T[-eps(T), 1])) ≈ Diagonal(T[0, 1])
    end
end
