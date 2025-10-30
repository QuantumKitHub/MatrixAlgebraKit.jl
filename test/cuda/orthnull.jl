using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: LinearAlgebra, I, mul!, diagm, norm
using MatrixAlgebraKit: GPU_SVDAlgorithm, check_input, copy_input, default_svd_algorithm,
    initialize_output, AbstractAlgorithm
using CUDA

# testing non-AbstractArray codepaths:
include(joinpath("..", "linearmap.jl"))

eltypes = (Float32, Float64, ComplexF32, ComplexF64)

@testset "left_orth and left_null for T = $T" for T in eltypes
    rng = StableRNG(123)
    m = 54
    @testset for n in (37, m, 63)
        minmn = min(m, n)
        A = CuArray(randn(rng, T, m, n))
        V, C = @constinferred left_orth(A)
        N = @constinferred left_null(A)
        @test V isa CuMatrix{T} && size(V) == (m, minmn)
        @test C isa CuMatrix{T} && size(C) == (minmn, n)
        @test N isa CuMatrix{T} && size(N) == (m, m - minmn)
        @test V * C ≈ A
        @test isisometric(V)
        @test norm(A' * N) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(N)
        hV = collect(V)
        hN = collect(N)
        @test hV * hV' + hN * hN' ≈ I

        M = LinearMap(A)
        VM, CM = @constinferred left_orth(M; alg = :svd)
        @test parent(VM) * parent(CM) ≈ A

        if m > n
            nullity = 5
            V, C = @constinferred left_orth(A)
            CUDA.@allowscalar begin
                N = @constinferred left_null(A; trunc = (; maxnullity = nullity))
            end
            @test V isa CuMatrix{T} && size(V) == (m, minmn)
            @test C isa CuMatrix{T} && size(C) == (minmn, n)
            @test N isa CuMatrix{T} && size(N) == (m, nullity)
            @test V * C ≈ A
            @test isisometric(V)
            @test LinearAlgebra.norm(A' * N) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
            @test isisometric(N)
        end

        # passing a kind and some kwargs
        V, C = @constinferred left_orth(A; alg = :qr, positive = true)
        N = @constinferred left_null(A; alg = :qr, positive = true)
        @test V isa CuMatrix{T} && size(V) == (m, minmn)
        @test C isa CuMatrix{T} && size(C) == (minmn, n)
        @test N isa CuMatrix{T} && size(N) == (m, m - minmn)
        @test V * C ≈ A
        @test isisometric(V)
        @test LinearAlgebra.norm(A' * N) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(N)
        hV = collect(V)
        hN = collect(N)
        @test hV * hV' + hN * hN' ≈ I

        # passing an algorithm
        V, C = @constinferred left_orth(A; alg = CUSOLVER_HouseholderQR())
        N = @constinferred left_null(A; alg = :qr, positive = true)
        @test V isa CuMatrix{T} && size(V) == (m, minmn)
        @test C isa CuMatrix{T} && size(C) == (minmn, n)
        @test N isa CuMatrix{T} && size(N) == (m, m - minmn)
        @test V * C ≈ A
        @test isisometric(V)
        @test LinearAlgebra.norm(A' * N) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(N)
        hV = collect(V)
        hN = collect(N)
        @test hV * hV' + hN * hN' ≈ I

        Ac = similar(A)
        V2, C2 = @constinferred left_orth!(copy!(Ac, A), (V, C))
        N2 = @constinferred left_null!(copy!(Ac, A), N)
        @test V2 * C2 ≈ A
        @test isisometric(V2)
        @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(N2)
        hV2 = collect(V2)
        hN2 = collect(N2)
        @test hV2 * hV2' + hN2 * hN2' ≈ I

        atol = eps(real(T))
        V2, C2 = @constinferred left_orth!(copy!(Ac, A), (V, C); trunc = (; atol = atol))
        N2 = @constinferred left_null!(copy!(Ac, A), N; trunc = (; atol = atol))
        @test V2 * C2 ≈ A
        @test isisometric(V2)
        @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(N2)
        hV2 = collect(V2)
        hN2 = collect(N2)
        @test hV2 * hV2' + hN2 * hN2' ≈ I

        rtol = eps(real(T))
        for (trunc_orth, trunc_null) in (
                ((; rtol = rtol), (; rtol = rtol)),
                (trunctol(; rtol), trunctol(; rtol, keep_below = true)),
            )
            V2, C2 = @constinferred left_orth!(copy!(Ac, A), (V, C); trunc = trunc_orth)
            N2 = @constinferred left_null!(copy!(Ac, A), N; trunc = trunc_null)
            @test V2 * C2 ≈ A
            @test isisometric(V2)
            @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
            @test isisometric(N2)
            hV2 = collect(V2)
            hN2 = collect(N2)
            @test hV2 * hV2' + hN2 * hN2' ≈ I
        end

        @testset for alg in (:qr, :polar, :svd) # explicit alg kwarg
            m < n && alg == :polar && continue
            V2, C2 = @constinferred left_orth!(copy!(Ac, A), (V, C); alg = $(QuoteNode(alg)))
            @test V2 * C2 ≈ A
            @test isisometric(V2)
            if alg != :polar
                N2 = @constinferred left_null!(copy!(Ac, A), N; alg)
                @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test isisometric(N2)
                hV2 = collect(V2)
                hN2 = collect(N2)
                @test hV2 * hV2' + hN2 * hN2' ≈ I
            end

            # with alg and tol kwargs
            if alg == :svd
                V2, C2 = @constinferred left_orth!(copy!(Ac, A), (V, C); alg = $(QuoteNode(alg)), trunc = (; atol))
                N2 = @constinferred left_null!(copy!(Ac, A), N; alg, trunc = (; atol))
                @test V2 * C2 ≈ A
                @test isisometric(V2)
                @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test isisometric(N2)
                hV2 = collect(V2)
                hN2 = collect(N2)
                @test hV2 * hV2' + hN2 * hN2' ≈ I

                V2, C2 = @constinferred left_orth!(copy!(Ac, A), (V, C); alg = $(QuoteNode(alg)), trunc = (; rtol))
                N2 = @constinferred left_null!(copy!(Ac, A), N; alg, trunc = (; rtol))
                @test V2 * C2 ≈ A
                @test isisometric(V2)
                @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test isisometric(N2)
                hV2 = collect(V2)
                hN2 = collect(N2)
                @test hV2 * hV2' + hN2 * hN2' ≈ I
            else
                @test_throws ArgumentError left_orth!(copy!(Ac, A), (V, C); alg, trunc = (; atol))
                @test_throws ArgumentError left_orth!(copy!(Ac, A), (V, C); alg, trunc = (; rtol))
                alg == :polar && continue
                @test_throws ArgumentError left_null!(copy!(Ac, A), N; alg, trunc = (; atol))
                @test_throws ArgumentError left_null!(copy!(Ac, A), N; alg, trunc = (; rtol))
            end
        end
    end
end

@testset "right_orth and right_null for T = $T" for T in eltypes
    rng = StableRNG(123)
    m = 54
    @testset for n in (37, m, 63)
        minmn = min(m, n)
        A = CuArray(randn(rng, T, m, n))
        C, Vᴴ = @constinferred right_orth(A)
        Nᴴ = @constinferred right_null(A)
        @test C isa CuMatrix{T} && size(C) == (m, minmn)
        @test Vᴴ isa CuMatrix{T} && size(Vᴴ) == (minmn, n)
        @test Nᴴ isa CuMatrix{T} && size(Nᴴ) == (n - minmn, n)
        @test C * Vᴴ ≈ A
        @test isisometric(Vᴴ; side = :right)
        @test LinearAlgebra.norm(A * adjoint(Nᴴ)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(Nᴴ; side = :right)
        hVᴴ = collect(Vᴴ)
        hNᴴ = collect(Nᴴ)
        @test hVᴴ' * hVᴴ + hNᴴ' * hNᴴ ≈ I

        M = LinearMap(A)
        CM, VMᴴ = @constinferred right_orth(M; alg = :svd)
        @test parent(CM) * parent(VMᴴ) ≈ A

        Ac = similar(A)
        C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ))
        Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ)
        @test C2 * Vᴴ2 ≈ A
        @test isisometric(Vᴴ2; side = :right)
        @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(Nᴴ; side = :right)
        hVᴴ2 = collect(Vᴴ2)
        hNᴴ2 = collect(Nᴴ2)
        @test hVᴴ2' * hVᴴ2 + hNᴴ2' * hNᴴ2 ≈ I

        atol = eps(real(T))
        rtol = eps(real(T))
        C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ); trunc = (; atol = atol))
        Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ; trunc = (; atol = atol))
        @test C2 * Vᴴ2 ≈ A
        @test isisometric(Vᴴ2; side = :right)
        @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(Nᴴ; side = :right)
        hVᴴ2 = collect(Vᴴ2)
        hNᴴ2 = collect(Nᴴ2)
        @test hVᴴ2' * hVᴴ2 + hNᴴ2' * hNᴴ2 ≈ I

        C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ); trunc = (; rtol = rtol))
        Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ; trunc = (; rtol = rtol))
        @test C2 * Vᴴ2 ≈ A
        @test isisometric(Vᴴ2; side = :right)
        @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(Nᴴ2; side = :right)
        hVᴴ2 = collect(Vᴴ2)
        hNᴴ2 = collect(Nᴴ2)
        @test hVᴴ2' * hVᴴ2 + hNᴴ2' * hNᴴ2 ≈ I

        @testset "alg = $alg" for alg in (:lq, :polar, :svd)
            n < m && alg == :polar && continue
            C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ); alg = $(QuoteNode(alg)))
            @test C2 * Vᴴ2 ≈ A
            @test isisometric(Vᴴ2; side = :right)
            if alg != :polar
                Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ; alg = $(QuoteNode(alg)))
                @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test isisometric(Nᴴ2; side = :right)
                hVᴴ2 = collect(Vᴴ2)
                hNᴴ2 = collect(Nᴴ2)
                @test hVᴴ2' * hVᴴ2 + hNᴴ2' * hNᴴ2 ≈ I
            end

            if alg == :svd
                C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ); alg = $(QuoteNode(alg)), trunc = (; atol))
                Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ; alg = $(QuoteNode(alg)), trunc = (; atol))
                @test C2 * Vᴴ2 ≈ A
                @test isisometric(Vᴴ2; side = :right)
                @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test isisometric(Nᴴ2; side = :right)
                hVᴴ2 = collect(Vᴴ2)
                hNᴴ2 = collect(Nᴴ2)
                @test hVᴴ2' * hVᴴ2 + hNᴴ2' * hNᴴ2 ≈ I

                C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ); alg = $(QuoteNode(alg)), trunc = (; rtol))
                Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ; alg = $(QuoteNode(alg)), trunc = (; rtol))
                @test C2 * Vᴴ2 ≈ A
                @test isisometric(Vᴴ2; side = :right)
                @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test isisometric(Nᴴ2; side = :right)
                hVᴴ2 = collect(Vᴴ2)
                hNᴴ2 = collect(Nᴴ2)
                @test hVᴴ2' * hVᴴ2 + hNᴴ2' * hNᴴ2 ≈ I
            else
                @test_throws ArgumentError right_orth!(copy!(Ac, A), (C, Vᴴ); alg, trunc = (; atol))
                @test_throws ArgumentError right_orth!(copy!(Ac, A), (C, Vᴴ); alg, trunc = (; rtol))
                alg == :polar && continue
                @test_throws ArgumentError right_null!(copy!(Ac, A), Nᴴ; alg, trunc = (; atol))
                @test_throws ArgumentError right_null!(copy!(Ac, A), Nᴴ; alg, trunc = (; rtol))
            end
        end
    end
end
