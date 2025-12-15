using MatrixAlgebraKit
using MatrixAlgebraKit: diagview
using LinearAlgebra: Diagonal, isposdef, norm, opnorm
using Test
using TestExtras
using StableRNGs
using CUDA

include(joinpath("..", "utilities.jl"))

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)

@testset "svd_compact! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54
    @testset "size ($m, $n)" for n in (37, m, 63)
        k = min(m, n)
        algs(::CuArray) = (CUSOLVER_QRIteration(), CUSOLVER_SVDPolar(), CUSOLVER_Jacobi())
        algs(::Diagonal) = (DiagonalAlgorithm(),)
        As = m == n ? (CuArray(randn(rng, T, m, n)), Diagonal(CuArray(randn(rng, T, m)))) : (CuArray(randn(rng, T, m, n)),)
        for A in As
            @testset "algorithm $alg" for alg in algs(A)
                minmn = min(m, n)
                U, S, Vᴴ = svd_compact(A; alg)
                @test U isa CuMatrix{T} && size(U) == (m, minmn)
                @test S isa Diagonal{real(T), <:CuVector} && size(S) == (minmn, minmn)
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
                if alg isa CUSOLVER_QRIteration
                    @test_warn "invalid keyword arguments for GPU_QRIteration" svd_compact!(copy!(Ac, A), (U, S, Vᴴ), CUSOLVER_QRIteration(; bad = "bad"))
                end
            end
        end
    end
end

@testset "svd_full! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54
    algs(::CuArray) = (CUSOLVER_QRIteration(), CUSOLVER_SVDPolar(), CUSOLVER_Jacobi())
    algs(::Diagonal) = (DiagonalAlgorithm(),)
    @testset "size ($m, $n)" for n in (37, m, 63)
        As = m == n ? (CuArray(randn(rng, T, m, n)), Diagonal(CuArray(randn(rng, T, m)))) : (CuArray(randn(rng, T, m, n)),)
        for A in As
            @testset "algorithm $alg" for alg in algs(A)
                minmn = min(m, n)
                U, S, Vᴴ = svd_full(A; alg)
                @test U isa CuMatrix{T} && size(U) == (m, m)
                if A isa Diagonal
                    @test S isa Diagonal{real(T), <:CuVector{real(T)}} && size(S) == (m, n)
                else
                    @test S isa CuMatrix{real(T)} && size(S) == (m, n)
                end
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
                if alg isa CUSOLVER_QRIteration
                    @test_warn "invalid keyword arguments for GPU_QRIteration" svd_full!(copy!(Ac, A), (U, S, Vᴴ), CUSOLVER_QRIteration(; bad = "bad"))
                    @test_warn "invalid keyword arguments for GPU_QRIteration" svd_vals!(copy!(Ac, A), Sc, CUSOLVER_QRIteration(; bad = "bad"))
                end
            end
        end
    end
    @testset "size (0, 0)" begin
        for A in (CuArray(randn(rng, T, 0, 0)), Diagonal(CuArray(randn(rng, T, 0))))
            @testset "algorithm $alg" for alg in algs(A)
                U, S, Vᴴ = svd_full(A; alg)
                @test U isa CuMatrix{T} && size(U) == (0, 0)
                @test size(S) == (0, 0)
                if isa(A, Diagonal)
                    @test S isa Diagonal{real(T), <:CuVector{real(T)}}
                else
                    @test S isa CuMatrix{real(T)}
                end
                @test Vᴴ isa CuMatrix{T} && size(Vᴴ) == (0, 0)
                @test U * S * Vᴴ ≈ A
                @test isapproxone(U' * U)
                @test isapproxone(U * U')
                @test isapproxone(Vᴴ * Vᴴ')
                @test isapproxone(Vᴴ' * Vᴴ)
                @test all(isposdef, diagview(S))
            end
        end
    end
end

@testset "svd_trunc! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    @testset "size ($m, $n)" for n in (37, m, 63)
        k = min(m, n) - 20
        p = min(m, n) - k - 1
        algs(::CuArray) = (CUSOLVER_QRIteration(), CUSOLVER_SVDPolar(), CUSOLVER_Jacobi(), CUSOLVER_Randomized(; k = k, p = p, niters = 100))
        algs(::Diagonal) = (DiagonalAlgorithm(),)
        hAs = m == n ? (randn(rng, T, m, n), Diagonal(randn(rng, T, m))) : (randn(rng, T, m, n),)
        minmn = min(m, n)
        for hA in hAs
            A = CuArray(hA)
            @testset "algorithm $alg" for alg in algs(A)
                S₀ = svd_vals(hA)
                r = k

                U1, S1, V1ᴴ, ϵ1 = @constinferred svd_trunc(A; alg, trunc = truncrank(r))
                @test length(S1.diag) == r
                @test opnorm(A - U1 * S1 * V1ᴴ) ≈ S₀[r + 1]
                @test norm(A - U1 * S1 * V1ᴴ) ≈ ϵ1
                U1, S1, V1ᴴ = @constinferred svd_trunc_no_error(A; alg, trunc = truncrank(r))
                @test length(S1.diag) == r
                @test opnorm(A - U1 * S1 * V1ᴴ) ≈ S₀[r + 1]

                if !(alg isa CUSOLVER_Randomized)
                    s = 1 + sqrt(eps(real(T)))
                    trunc2 = trunctol(; atol = s * S₀[r + 1])

                    U2, S2, V2ᴴ, ϵ2 = @constinferred svd_trunc(A; alg, trunc = trunctol(; atol = s * S₀[r + 1]))
                    @test length(S2.diag) == r
                    @test U1 ≈ U2
                    @test parent(S1) ≈ parent(S2)
                    @test V1ᴴ ≈ V2ᴴ

                    U2, S2, V2ᴴ = @constinferred svd_trunc_no_error(A; alg, trunc = trunctol(; atol = s * S₀[r + 1]))
                    @test length(S2.diag) == r
                    @test U1 ≈ U2
                    @test parent(S1) ≈ parent(S2)
                    @test V1ᴴ ≈ V2ᴴ
                end
            end
        end
    end
end
