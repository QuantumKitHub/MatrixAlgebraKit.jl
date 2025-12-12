using MatrixAlgebraKit
using MatrixAlgebraKit: diagview
using LinearAlgebra: Diagonal, isposdef
using Test
using TestExtras
using StableRNGs
using AMDGPU

include(joinpath("..", "utilities.jl"))

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)

@testset "svd_compact! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54
    @testset "size ($m, $n)" for n in (37, m, 63)
        k = min(m, n)
        algs(::ROCArray) = (ROCSOLVER_QRIteration(), ROCSOLVER_Jacobi())
        algs(::Diagonal) = (DiagonalAlgorithm(),)
        As = m == n ? (ROCArray(randn(rng, T, m, n)), Diagonal(ROCArray(randn(rng, T, m)))) : (ROCArray(randn(rng, T, m, n)),)
        for A in As
            @testset "algorithm $alg" for alg in algs(A)
                minmn = min(m, n)

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
                if alg isa ROCSOLVER_QRIteration
                    @test_warn "invalid keyword arguments for GPU_QRIteration" svd_compact!(copy!(Ac, A), (U, S, Vᴴ), ROCSOLVER_QRIteration(; bad = "bad"))
                end
            end
        end
    end
end

@testset "svd_full! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54
    algs(::ROCArray) = (ROCSOLVER_QRIteration(), ROCSOLVER_Jacobi())
    algs(::Diagonal) = (DiagonalAlgorithm(),)
    @testset "size ($m, $n)" for n in (37, m, 63)
        As = m == n ? (ROCArray(randn(rng, T, m, n)), Diagonal(ROCArray(randn(rng, T, m)))) : (ROCArray(randn(rng, T, m, n)),)
        for A in As
            @testset "algorithm $alg" for alg in algs(A)
                U, S, Vᴴ = svd_full(A; alg)
                @test U isa ROCMatrix{T} && size(U) == (m, m)
                if A isa Diagonal
                    @test S isa Diagonal{real(T), <:ROCVector{real(T)}} && size(S) == (m, n)
                else
                    @test S isa ROCMatrix{real(T)} && size(S) == (m, n)
                end
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
                if alg isa ROCSOLVER_QRIteration
                    @test_warn "invalid keyword arguments for GPU_QRIteration" svd_full!(copy!(Ac, A), (U, S, Vᴴ), ROCSOLVER_QRIteration(; bad = "bad"))
                    @test_warn "invalid keyword arguments for GPU_QRIteration" svd_vals!(copy!(Ac, A), Sc, ROCSOLVER_QRIteration(; bad = "bad"))
                end
            end
        end
    end
    @testset "size (0, 0)" begin
        for A in (ROCArray(randn(rng, T, 0, 0)), Diagonal(ROCArray(randn(rng, T, 0))))
            @testset "algorithm $alg" for alg in algs(A)
                U, S, Vᴴ = svd_full(A; alg)
                @test U isa ROCMatrix{T} && size(U) == (0, 0)
                if isa(A, Diagonal)
                    @test S isa Diagonal{real(T), <:ROCVector{real(T)}}
                else
                    @test S isa ROCMatrix{real(T)}
                end
                @test Vᴴ isa ROCMatrix{T} && size(Vᴴ) == (0, 0)
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

# @testset "svd_trunc! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
#     rng = StableRNG(123)
#     m = 54
#     if LinearAlgebra.LAPACK.version() < v"3.12.0"
#         algs = (LAPACK_DivideAndConquer(), LAPACK_QRIteration(), LAPACK_Bisection())
#     else
#         algs = (LAPACK_DivideAndConquer(), LAPACK_QRIteration(), LAPACK_Bisection(),
#                 LAPACK_Jacobi())
#     end
#
#     @testset "size ($m, $n)" for n in (37, m, 63)
#         @testset "algorithm $alg" for alg in algs
#             n > m && alg isa LAPACK_Jacobi && continue # not supported
#             A = randn(rng, T, m, n)
#             S₀ = svd_vals(A)
#             minmn = min(m, n)
#             r = minmn - 2
#
#             U1, S1, V1ᴴ, ϵ1 = @constinferred svd_trunc_with_err(A; alg, trunc=truncrank(r))
#             @test length(S1.diag) == r
#             @test LinearAlgebra.opnorm(A - U1 * S1 * V1ᴴ) ≈ S₀[r + 1]
#
#             s = 1 + sqrt(eps(real(T)))
#             trunc2 = trunctol(; atol=s * S₀[r + 1])
#
#             U2, S2, V2ᴴ, ϵ2 = @constinferred svd_trunc_with_err(A; alg, trunc=trunctol(; atol=s * S₀[r + 1]))
#             @test length(S2.diag) == r
#             @test U1 ≈ U2
#             @test S1 ≈ S2
#             @test V1ᴴ ≈ V2ᴴ
#         end
#     end
# end
