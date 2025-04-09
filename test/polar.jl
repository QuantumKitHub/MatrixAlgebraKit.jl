using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: LinearAlgebra, I, isposdef
using MatrixAlgebraKit: PolarViaSVD

@testset "left_polar! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    @testset "size ($m, $n)" for n in (37, m)
        k = min(m, n)
        if LinearAlgebra.LAPACK.version() < v"3.12.0"
            algs = PolarViaSVD.((LAPACK_DivideAndConquer(), LAPACK_QRIteration(),
                                 LAPACK_Bisection()))
        else
            algs = PolarViaSVD.((LAPACK_DivideAndConquer(), LAPACK_QRIteration(),
                                 LAPACK_Bisection(), LAPACK_Jacobi()))
        end
        @testset "algorithm $alg" for alg in algs
            A = randn(rng, T, m, n)

            W, P = left_polar(A; alg)
            @test W isa Matrix{T} && size(W) == (m, n)
            @test P isa Matrix{T} && size(P) == (n, n)
            @test W * P ≈ A
            @test W' * W ≈ I
            @test isposdef(P)

            Ac = similar(A)
            W2, P2 = @constinferred left_polar!(copy!(Ac, A), (W, P), alg)
            @test W2 === W
            @test P2 === P
            @test W * P ≈ A
            @test W' * W ≈ I
            @test isposdef(P)
        end
    end
end

@testset "right_polar! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    n = 54
    @testset "size ($m, $n)" for m in (37, n)
        k = min(m, n)
        algs = PolarViaSVD.((LAPACK_DivideAndConquer(), LAPACK_QRIteration(),
                             LAPACK_Bisection()))
        @testset "algorithm $alg" for alg in algs
            A = randn(rng, T, m, n)

            P, Wᴴ = right_polar(A; alg)
            @test Wᴴ isa Matrix{T} && size(Wᴴ) == (m, n)
            @test P isa Matrix{T} && size(P) == (m, m)
            @test P * Wᴴ ≈ A
            @test Wᴴ * Wᴴ' ≈ I
            @test isposdef(P)

            Ac = similar(A)
            P2, Wᴴ2 = @constinferred right_polar!(copy!(Ac, A), (P, Wᴴ), alg)
            @test P2 === P
            @test Wᴴ2 === Wᴴ
            @test P * Wᴴ ≈ A
            @test Wᴴ * Wᴴ' ≈ I
            @test isposdef(P)
        end
    end
end
