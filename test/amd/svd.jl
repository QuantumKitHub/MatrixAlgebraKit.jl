using MatrixAlgebraKit
using MatrixAlgebraKit: diagview
using LinearAlgebra: Diagonal, isposdef
using Test
using TestExtras
using StableRNGs
using AMDGPU

include(joinpath("..", "utilities.jl"))

@testset "svd_compact! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    @testset "size ($m, $n)" for n in (37, m, 63)
        k = min(m, n)
        algs = (ROCSOLVER_QRIteration(), ROCSOLVER_Jacobi())
        @testset "algorithm $alg" for alg in algs
            n > m && alg isa ROCSOLVER_QRIteration && continue # not supported
            minmn = min(m, n)
            A = ROCArray(randn(rng, T, m, n))

            U, S, Vᴴ = svd_compact(A; alg)
            @test U isa ROCMatrix{T} && size(U) == (m, minmn)
            @test S isa Diagonal{real(T), <:ROCVector} && size(S) == (minmn, minmn)
            @test Vᴴ isa ROCMatrix{T} && size(Vᴴ) == (minmn, n)
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
            @test ROCArray(diagview(S)) ≈ Sd
            # ROCArray is necessary because norm of ROCArray view with non-unit step is broken
        end
    end
end

@testset "svd_full! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    @testset "size ($m, $n)" for n in (37, m, 63)
        algs = (ROCSOLVER_QRIteration(), ROCSOLVER_Jacobi())
        @testset "algorithm $alg" for alg in algs
            n > m && alg isa ROCSOLVER_QRIteration && continue # not supported
            A = ROCArray(randn(rng, T, m, n))
            U, S, Vᴴ = svd_full(A; alg)
            @test U isa ROCMatrix{T} && size(U) == (m, m)
            @test S isa ROCMatrix{real(T)} && size(S) == (m, n)
            @test Vᴴ isa ROCMatrix{T} && size(Vᴴ) == (n, n)
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

            Sc = similar(A, real(T), min(m, n))
            Sc2 = svd_vals!(copy!(Ac, A), Sc, alg)
            @test Sc === Sc2
            @test ROCArray(diagview(S)) ≈ Sc
            # ROCArray is necessary because norm of ROCArray view with non-unit step is broken
        end
    end
end

# @testset "svd_trunc! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
#     rng = StableRNG(123)
#     m = 54
#     if LinearAlgebra.LAPACK.version() < v"3.12.0"
#         algs = (LAPACK_DivideAndConquer(), LAPACK_QRIteration(), LAPACK_Bisection())
#     else
#         algs = (LAPACK_DivideAndConquer(), LAPACK_QRIteration(), LAPACK_Bisection(),
#                 LAPACK_Jacobi())
#     end

#     @testset "size ($m, $n)" for n in (37, m, 63)
#         @testset "algorithm $alg" for alg in algs
#             n > m && alg isa LAPACK_Jacobi && continue # not supported
#             A = randn(rng, T, m, n)
#             S₀ = svd_vals(A)
#             minmn = min(m, n)
#             r = minmn - 2

#             U1, S1, V1ᴴ, ϵ1 = @constinferred svd_trunc(A; alg, trunc=truncrank(r))
#             @test length(S1.diag) == r
#             @test LinearAlgebra.opnorm(A - U1 * S1 * V1ᴴ) ≈ S₀[r + 1]

#             s = 1 + sqrt(eps(real(T)))
#             trunc2 = trunctol(; atol=s * S₀[r + 1])

#             U2, S2, V2ᴴ, ϵ2 = @constinferred svd_trunc(A; alg, trunc=trunctol(; atol=s * S₀[r + 1]))
#             @test length(S2.diag) == r
#             @test U1 ≈ U2
#             @test S1 ≈ S2
#             @test V1ᴴ ≈ V2ᴴ
#         end
#     end
# end
