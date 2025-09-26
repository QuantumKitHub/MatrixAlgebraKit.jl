using MatrixAlgebraKit
using MatrixAlgebraKit: diagview
using LinearAlgebra: Diagonal, isposdef, opnorm
using Test
using TestExtras
using StableRNGs
using CUDA

include(joinpath("..", "utilities.jl"))

@testset "svd_compact! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    @testset "size ($m, $n)" for n in (37, m, 63)
        k = min(m, n)
        algs = (CUSOLVER_QRIteration(), CUSOLVER_SVDPolar(), CUSOLVER_Jacobi())
        @testset "algorithm $alg" for alg in algs
            n > m && alg isa CUSOLVER_QRIteration && continue # not supported
            minmn = min(m, n)
            A = CuArray(randn(rng, T, m, n))

            U, S, Vᴴ = svd_compact(A; alg)
            @test U isa CuMatrix{T} && size(U) == (m, minmn)
            @test S isa Diagonal{real(T),<:CuVector} && size(S) == (minmn, minmn)
            @test Vᴴ isa CuMatrix{T} && size(Vᴴ) == (minmn, n)
            @test U * S * Vᴴ ≈ A
            @test isapproxone(U' * U)
            @test isapproxone(Vᴴ * Vᴴ')
            @test isposdef(S)

            Ac = similar(A)
            U2, S2, V2ᴴ = @constinferred svd_compact!(copy!(Ac, A), (U, S, Vᴴ), alg)
            @test U2 === U
            @test S2 === S
            @test V2ᴴ === Vᴴ
            @test U * S * Vᴴ ≈ A
            @test isapproxone(U' * U)
            @test isapproxone(Vᴴ * Vᴴ')
            @test isposdef(S)

            Sd = svd_vals(A, alg)
            @test CuArray(diagview(S)) ≈ Sd
            # CuArray is necessary because norm of CuArray view with non-unit step is broken
        end
    end
end

@testset "svd_full! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    @testset "size ($m, $n)" for n in (37, m, 63)
        algs = (CUSOLVER_QRIteration(), CUSOLVER_SVDPolar(), CUSOLVER_Jacobi())
        @testset "algorithm $alg" for alg in algs
            n > m && alg isa CUSOLVER_QRIteration && continue # not supported
            A = CuArray(randn(rng, T, m, n))
            U, S, Vᴴ = svd_full(A; alg)
            @test U isa CuMatrix{T} && size(U) == (m, m)
            @test S isa CuMatrix{real(T)} && size(S) == (m, n)
            @test Vᴴ isa CuMatrix{T} && size(Vᴴ) == (n, n)
            @test U * S * Vᴴ ≈ A
            @test isapproxone(U' * U)
            @test isapproxone(U * U')
            @test isapproxone(Vᴴ * Vᴴ')
            @test isapproxone(Vᴴ' * Vᴴ)
            @test all(isposdef, diagview(S))

            Ac = similar(A)
            U2, S2, V2ᴴ = @constinferred svd_full!(copy!(Ac, A), (U, S, Vᴴ), alg)
            @test U2 === U
            @test S2 === S
            @test V2ᴴ === Vᴴ
            @test U * S * Vᴴ ≈ A
            @test isapproxone(U' * U)
            @test isapproxone(U * U')
            @test isapproxone(Vᴴ * Vᴴ')
            @test isapproxone(Vᴴ' * Vᴴ)
            @test all(isposdef, diagview(S))

            minmn = min(m, n)
            Sc = similar(A, real(T), minmn)
            Sc2 = svd_vals!(copy!(Ac, A), Sc, alg)
            @test Sc === Sc2
            @test CuArray(diagview(S)) ≈ Sc
            # CuArray is necessary because norm of CuArray view with non-unit step is broken
        end
        @testset "algorithm $alg" for alg in algs
        end
    end
end

@testset "svd_trunc! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    @testset "size ($m, $n)" for n in (37, m, 63)
        k = min(m, n) - 20
        p = min(m, n) - k - 1
        algs = (CUSOLVER_QRIteration(), CUSOLVER_SVDPolar(), CUSOLVER_Jacobi(), CUSOLVER_Randomized(; k=k, p=p, niters=100),)
        @testset "algorithm $alg" for alg in algs
            n > m && alg isa CUSOLVER_QRIteration && continue # not supported
            hA = randn(rng, T, m, n)
            S₀ = svd_vals(hA)
            A = CuArray(hA)
            minmn = min(m, n)
            r = k 

            U1, S1, V1ᴴ = @constinferred svd_trunc(A; alg, trunc=truncrank(r))
            @test length(S1.diag) == r
            @test opnorm(A - U1 * S1 * V1ᴴ) ≈ S₀[r + 1]

            if !(alg isa CUSOLVER_Randomized)
                s = 1 + sqrt(eps(real(T)))
                trunc2 = trunctol(; atol=s * S₀[r + 1])

                U2, S2, V2ᴴ = @constinferred svd_trunc(A; alg, trunc=trunctol(; atol=s * S₀[r + 1]))
                @test length(S2.diag) == r
                @test U1 ≈ U2
                @test parent(S1) ≈ parent(S2)
                @test V1ᴴ ≈ V2ᴴ
            end
        end
    end
end
