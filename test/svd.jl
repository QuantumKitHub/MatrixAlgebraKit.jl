using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: LinearAlgebra, Diagonal, I, isposdef, norm
using MatrixAlgebraKit: TruncatedAlgorithm, diagview, isisometric

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
GenericFloats = (Float16, BigFloat, Complex{BigFloat})

@testset "svd_compact! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54
    @testset "size ($m, $n)" for n in (37, m, 63, 0)
        k = min(m, n)
        if LinearAlgebra.LAPACK.version() < v"3.12.0"
            algs = (
                LAPACK_DivideAndConquer(), LAPACK_QRIteration(), LAPACK_Bisection(),
                LAPACK_DivideAndConquer, :LAPACK_DivideAndConquer,
            )
        else
            algs = (
                LAPACK_DivideAndConquer(), LAPACK_QRIteration(), LAPACK_Bisection(),
                LAPACK_Jacobi(), LAPACK_DivideAndConquer, :LAPACK_DivideAndConquer,
            )
        end
        @testset "algorithm $alg" for alg in algs
            n > m && alg isa LAPACK_Jacobi && continue # not supported
            minmn = min(m, n)
            A = randn(rng, T, m, n)

            if VERSION < v"1.11"
                # This is type unstable on older versions of Julia.
                U, S, Vᴴ = svd_compact(A; alg)
            else
                U, S, Vᴴ = @constinferred svd_compact(A; alg = ($alg))
            end
            @test U isa Matrix{T} && size(U) == (m, minmn)
            @test S isa Diagonal{real(T)} && size(S) == (minmn, minmn)
            @test Vᴴ isa Matrix{T} && size(Vᴴ) == (minmn, n)
            @test U * S * Vᴴ ≈ A
            @test isisometric(U)
            @test isisometric(Vᴴ; side = :right)
            @test isposdef(S)

            Ac = similar(A)
            Sc = similar(A, real(T), min(m, n))
            alg′ = @constinferred MatrixAlgebraKit.select_algorithm(svd_compact!, A, $alg)
            U2, S2, V2ᴴ = @constinferred svd_compact!(copy!(Ac, A), (U, S, Vᴴ), alg′)
            @test U2 === U
            @test S2 === S
            @test V2ᴴ === Vᴴ
            @test U * S * Vᴴ ≈ A
            @test isisometric(U)
            @test isisometric(Vᴴ; side = :right)
            @test isposdef(S)

            Sd = @constinferred svd_vals(A, alg′)
            @test S ≈ Diagonal(Sd)
        end
    end
end

@testset "svd_full! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54
    @testset "size ($m, $n)" for n in (37, m, 63, 0)
        @testset "algorithm $alg" for alg in
            (LAPACK_DivideAndConquer(), LAPACK_QRIteration())
            A = randn(rng, T, m, n)
            U, S, Vᴴ = svd_full(A; alg)
            @test U isa Matrix{T} && size(U) == (m, m)
            @test S isa Matrix{real(T)} && size(S) == (m, n)
            @test Vᴴ isa Matrix{T} && size(Vᴴ) == (n, n)
            @test U * S * Vᴴ ≈ A
            @test isunitary(U)
            @test isunitary(Vᴴ)
            @test all(isposdef, diagview(S))

            Ac = similar(A)
            U2, S2, V2ᴴ = @constinferred svd_full!(copy!(Ac, A), (U, S, Vᴴ), alg)
            @test U2 === U
            @test S2 === S
            @test V2ᴴ === Vᴴ
            @test U * S * Vᴴ ≈ A
            @test isunitary(U)
            @test isunitary(Vᴴ)
            @test all(isposdef, diagview(S))

            Sc = similar(A, real(T), min(m, n))
            Sc2 = svd_vals!(copy!(Ac, A), Sc, alg)
            @test Sc === Sc2
            @test diagview(S) ≈ Sc
        end
    end
    @testset "size (0, 0)" begin
        @testset "algorithm $alg" for alg in
            (LAPACK_DivideAndConquer(), LAPACK_QRIteration())
            A = randn(rng, T, 0, 0)
            U, S, Vᴴ = svd_full(A; alg)
            @test U isa Matrix{T} && size(U) == (0, 0)
            @test S isa Matrix{real(T)} && size(S) == (0, 0)
            @test Vᴴ isa Matrix{T} && size(Vᴴ) == (0, 0)
            @test U * S * Vᴴ ≈ A
            @test isunitary(U)
            @test isunitary(Vᴴ)
            @test all(isposdef, diagview(S))
        end
    end
end

@testset "svd_trunc! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54
    atol = sqrt(eps(real(T)))
    if LinearAlgebra.LAPACK.version() < v"3.12.0"
        algs = (LAPACK_DivideAndConquer(), LAPACK_QRIteration(), LAPACK_Bisection())
    else
        algs = (
            LAPACK_DivideAndConquer(), LAPACK_QRIteration(), LAPACK_Bisection(), LAPACK_Jacobi(),
        )
    end

    @testset "size ($m, $n)" for n in (37, m, 63)
        @testset "algorithm $alg" for alg in algs
            n > m && alg isa LAPACK_Jacobi && continue # not supported
            A = randn(rng, T, m, n)
            S₀ = svd_vals(A)
            minmn = min(m, n)
            r = minmn - 2

            U1, S1, V1ᴴ, ϵ1 = @constinferred svd_trunc(A; alg, trunc = truncrank(r))
            @test length(diagview(S1)) == r
            @test diagview(S1) ≈ S₀[1:r]
            @test LinearAlgebra.opnorm(A - U1 * S1 * V1ᴴ) ≈ S₀[r + 1]
            # Test truncation error
            @test ϵ1 ≈ norm(view(S₀, (r + 1):minmn)) atol = atol

            s = 1 + sqrt(eps(real(T)))
            trunc = trunctol(; atol = s * S₀[r + 1])

            U2, S2, V2ᴴ, ϵ2 = @constinferred svd_trunc(A; alg, trunc)
            @test length(diagview(S2)) == r
            @test U1 ≈ U2
            @test S1 ≈ S2
            @test V1ᴴ ≈ V2ᴴ
            @test ϵ2 ≈ norm(view(S₀, (r + 1):minmn)) atol = atol

            trunc = truncerror(; atol = s * norm(@view(S₀[(r + 1):end])))
            U3, S3, V3ᴴ, ϵ3 = @constinferred svd_trunc(A; alg, trunc)
            @test length(diagview(S3)) == r
            @test U1 ≈ U3
            @test S1 ≈ S3
            @test V1ᴴ ≈ V3ᴴ
            @test ϵ3 ≈ norm(view(S₀, (r + 1):minmn)) atol = atol
        end
    end
end

@testset "svd_trunc! mix maxrank and tol for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    if LinearAlgebra.LAPACK.version() < v"3.12.0"
        algs = (LAPACK_DivideAndConquer(), LAPACK_QRIteration(), LAPACK_Bisection())
    else
        algs = (
            LAPACK_DivideAndConquer(), LAPACK_QRIteration(), LAPACK_Bisection(), LAPACK_Jacobi(),
        )
    end
    m = 4
    @testset "algorithm $alg" for alg in algs
        U = qr_compact(randn(rng, T, m, m))[1]
        S = Diagonal(T[0.9, 0.3, 0.1, 0.01])
        Vᴴ = qr_compact(randn(rng, T, m, m))[1]
        A = U * S * Vᴴ

        for trunc_fun in (
                (rtol, maxrank) -> (; rtol, maxrank),
                (rtol, maxrank) -> truncrank(maxrank) & trunctol(; rtol),
            )
            U1, S1, V1ᴴ, ϵ1 = svd_trunc(A; alg, trunc = trunc_fun(0.2, 1))
            @test length(diagview(S1)) == 1
            @test diagview(S1) ≈ diagview(S)[1:1]

            U2, S2, V2ᴴ = svd_trunc_no_error(A; alg, trunc = trunc_fun(0.2, 3))
            @test length(diagview(S2)) == 2
            @test diagview(S2) ≈ diagview(S)[1:2]
        end
    end
end

@testset "svd_trunc! specify truncation algorithm T = $T" for T in BLASFloats
    rng = StableRNG(123)
    atol = sqrt(eps(real(T)))
    m = 4
    U = qr_compact(randn(rng, T, m, m))[1]
    S = Diagonal(real(T)[0.9, 0.3, 0.1, 0.01])
    Vᴴ = qr_compact(randn(rng, T, m, m))[1]
    A = U * S * Vᴴ
    alg = TruncatedAlgorithm(LAPACK_DivideAndConquer(), trunctol(; atol = 0.2))
    U2, S2, V2ᴴ, ϵ2 = @constinferred svd_trunc(A; alg)
    @test diagview(S2) ≈ diagview(S)[1:2]
    @test ϵ2 ≈ norm(diagview(S)[3:4]) atol = atol
    U2, S2, V2ᴴ = @constinferred svd_trunc_no_error(A; alg)
    @test diagview(S2) ≈ diagview(S)[1:2]
    @test_throws ArgumentError svd_trunc(A; alg, trunc = (; maxrank = 2))
    @test_throws ArgumentError svd_trunc_no_error(A; alg, trunc = (; maxrank = 2))
end

@testset "svd for Diagonal{$T}" for T in (BLASFloats..., GenericFloats...)
    rng = StableRNG(123)
    atol = sqrt(eps(real(T)))
    for m in (54, 0)
        Ad = randn(T, m)
        A = Diagonal(Ad)

        U, S, Vᴴ = @constinferred svd_compact(A)
        @test U isa AbstractMatrix{T} && size(U) == size(A)
        @test Vᴴ isa AbstractMatrix{T} && size(Vᴴ) == size(A)
        @test S isa Diagonal{real(T)} && size(S) == size(A)
        @test isunitary(U)
        @test isunitary(Vᴴ)
        @test all(≥(0), diagview(S))
        @test A ≈ U * S * Vᴴ

        U, S, Vᴴ = @constinferred svd_full(A)
        @test U isa AbstractMatrix{T} && size(U) == size(A)
        @test Vᴴ isa AbstractMatrix{T} && size(Vᴴ) == size(A)
        @test S isa Diagonal{real(T)} && size(S) == size(A)
        @test isunitary(U)
        @test isunitary(Vᴴ)
        @test all(≥(0), diagview(S))
        @test A ≈ U * S * Vᴴ

        S2 = @constinferred svd_vals(A)
        @test S2 isa AbstractVector{real(T)} && length(S2) == m
        @test S2 ≈ diagview(S)

        alg = TruncatedAlgorithm(DiagonalAlgorithm(), truncrank(2))
        U3, S3, Vᴴ3, ϵ3 = @constinferred svd_trunc(A; alg)
        @test diagview(S3) ≈ S2[1:min(m, 2)]
        @test ϵ3 ≈ norm(S2[(min(m, 2) + 1):m]) atol = atol
        U3, S3, Vᴴ3 = @constinferred svd_trunc_no_error(A; alg)
        @test diagview(S3) ≈ S2[1:min(m, 2)]
    end
end
