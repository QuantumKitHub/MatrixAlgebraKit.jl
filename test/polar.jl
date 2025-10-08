using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: LinearAlgebra, I, isposdef

@testset "left_polar! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    @testset "size ($m, $n)" for n in (37, m)
        k = min(m, n)
        if LinearAlgebra.LAPACK.version() < v"3.12.0"
            svdalgs = (LAPACK_DivideAndConquer(), LAPACK_QRIteration(), LAPACK_Bisection())
        else
            svdalgs = (LAPACK_DivideAndConquer(), LAPACK_QRIteration(), LAPACK_Bisection(), LAPACK_Jacobi())
        end
        algs = (PolarViaSVD.(svdalgs)..., PolarNewton())
        @testset "algorithm $alg" for alg in algs
            A = randn(rng, T, m, n)

            W, P = left_polar(A; alg)
            @test W isa Matrix{T} && size(W) == (m, n)
            @test P isa Matrix{T} && size(P) == (n, n)
            @test W * P ≈ A
            @test isisometric(W)
            @test isposdef(P)

            Ac = similar(A)
            W2, P2 = @constinferred left_polar!(copy!(Ac, A), (W, P), alg)
            @test W2 === W
            @test P2 === P
            @test W * P ≈ A
            @test isisometric(W)
            @test isposdef(P)

            noP = similar(P, (0, 0))
            W2, P2 = @constinferred left_polar!(copy!(Ac, A), (W, noP), alg)
            @test P2 === noP
            @test W2 === W
            @test isisometric(W)
            P = W' * A # compute P explicitly to verify W correctness
            @test ishermitian(P; rtol = MatrixAlgebraKit.defaulttol(P))
            @test isposdef(project_hermitian!(P))
        end
    end
end

@testset "right_polar! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    n = 54
    @testset "size ($m, $n)" for m in (37, n)
        k = min(m, n)
        svdalgs = (LAPACK_DivideAndConquer(), LAPACK_QRIteration(), LAPACK_Bisection())
        algs = (PolarViaSVD.(svdalgs)..., PolarNewton())
        @testset "algorithm $alg" for alg in algs
            A = randn(rng, T, m, n)

            P, Wᴴ = right_polar(A; alg)
            @test Wᴴ isa Matrix{T} && size(Wᴴ) == (m, n)
            @test P isa Matrix{T} && size(P) == (m, m)
            @test P * Wᴴ ≈ A
            @test isisometric(Wᴴ; side = :right)
            @test isposdef(P)

            Ac = similar(A)
            P2, Wᴴ2 = @constinferred right_polar!(copy!(Ac, A), (P, Wᴴ), alg)
            @test P2 === P
            @test Wᴴ2 === Wᴴ
            @test P * Wᴴ ≈ A
            @test isisometric(Wᴴ; side = :right)
            @test isposdef(P)

            noP = similar(P, (0, 0))
            P2, Wᴴ2 = @constinferred right_polar!(copy!(Ac, A), (noP, Wᴴ), alg)
            @test P2 === noP
            @test Wᴴ2 === Wᴴ
            @test isisometric(Wᴴ; side = :right)
            P = A * Wᴴ' # compute P explicitly to verify W correctness
            @test ishermitian(P; rtol = MatrixAlgebraKit.defaulttol(P))
            @test isposdef(project_hermitian!(P))
        end
    end
end
