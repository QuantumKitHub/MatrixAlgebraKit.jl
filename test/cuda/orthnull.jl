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

@testset "left_orth and left_null for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
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
        VM, CM = @constinferred left_orth(M; kind = :svd)
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

        for alg_qr in ((; positive = true), (; positive = false), CUSOLVER_HouseholderQR())
            V, C = @constinferred left_orth(A; alg_qr)
            N = @constinferred left_null(A; alg_qr)
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
        end

        Ac = similar(A)
        V2, C2 = @constinferred left_orth!(copy!(Ac, A), (V, C))
        N2 = @constinferred left_null!(copy!(Ac, A), N)
        @test V2 === V
        @test C2 === C
        @test N2 === N
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
        @test V2 !== V
        @test C2 !== C
        @test N2 !== C
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
            @test V2 !== V
            @test C2 !== C
            @test N2 !== C
            @test V2 * C2 ≈ A
            @test isisometric(V2)
            @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
            @test isisometric(N2)
            hV2 = collect(V2)
            hN2 = collect(N2)
            @test hV2 * hV2' + hN2 * hN2' ≈ I
        end

        @testset for kind in (:qr, :polar, :svd) # explicit kind kwarg
            m < n && kind == :polar && continue
            V2, C2 = @constinferred left_orth!(copy!(Ac, A), (V, C); kind = kind)
            @test V2 === V
            @test C2 === C
            @test V2 * C2 ≈ A
            @test isisometric(V2)
            if kind != :polar
                N2 = @constinferred left_null!(copy!(Ac, A), N; kind = kind)
                @test N2 === N
                @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test isisometric(N2)
                hV2 = collect(V2)
                hN2 = collect(N2)
                @test hV2 * hV2' + hN2 * hN2' ≈ I
            end

            # with kind and tol kwargs
            if kind == :svd
                V2, C2 = @constinferred left_orth!(
                    copy!(Ac, A), (V, C); kind = kind,
                    trunc = (; atol = atol)
                )
                N2 = @constinferred left_null!(
                    copy!(Ac, A), N; kind = kind,
                    trunc = (; atol = atol)
                )
                @test V2 !== V
                @test C2 !== C
                @test N2 !== C
                @test V2 * C2 ≈ A
                @test V2' * V2 ≈ I
                @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test isisometric(N2)
                hV2 = collect(V2)
                hN2 = collect(N2)
                @test hV2 * hV2' + hN2 * hN2' ≈ I

                V2, C2 = @constinferred left_orth!(
                    copy!(Ac, A), (V, C); kind = kind,
                    trunc = (; rtol = rtol)
                )
                N2 = @constinferred left_null!(
                    copy!(Ac, A), N; kind = kind,
                    trunc = (; rtol = rtol)
                )
                @test V2 !== V
                @test C2 !== C
                @test N2 !== C
                @test V2 * C2 ≈ A
                @test isisometric(V2)
                @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test isisometric(N2)
                hV2 = collect(V2)
                hN2 = collect(N2)
                @test hV2 * hV2' + hN2 * hN2' ≈ I
            else
                @test_throws ArgumentError left_orth!(
                    copy!(Ac, A), (V, C); kind = kind,
                    trunc = (; atol = atol)
                )
                @test_throws ArgumentError left_orth!(
                    copy!(Ac, A), (V, C); kind = kind,
                    trunc = (; rtol = rtol)
                )
                @test_throws ArgumentError left_null!(
                    copy!(Ac, A), N; kind = kind,
                    trunc = (; atol = atol)
                )
                @test_throws ArgumentError left_null!(
                    copy!(Ac, A), N; kind = kind,
                    trunc = (; rtol = rtol)
                )
            end
        end
    end
end

@testset "right_orth and right_null for T = $T" for T in (
        Float32, Float64, ComplexF32,
        ComplexF64,
    )
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
        CM, VMᴴ = @constinferred right_orth(M; kind = :svd)
        @test parent(CM) * parent(VMᴴ) ≈ A

        Ac = similar(A)
        C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ))
        Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ)
        @test C2 === C
        @test Vᴴ2 === Vᴴ
        @test Nᴴ2 === Nᴴ
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
        @test C2 !== C
        @test Vᴴ2 !== Vᴴ
        @test Nᴴ2 !== Nᴴ
        @test C2 * Vᴴ2 ≈ A
        @test isisometric(Vᴴ2; side = :right)
        @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(Nᴴ; side = :right)
        hVᴴ2 = collect(Vᴴ2)
        hNᴴ2 = collect(Nᴴ2)
        @test hVᴴ2' * hVᴴ2 + hNᴴ2' * hNᴴ2 ≈ I

        C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ); trunc = (; rtol = rtol))
        Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ; trunc = (; rtol = rtol))
        @test C2 !== C
        @test Vᴴ2 !== Vᴴ
        @test Nᴴ2 !== Nᴴ
        @test C2 * Vᴴ2 ≈ A
        @test isisometric(Vᴴ2; side = :right)
        @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(Nᴴ2; side = :right)
        hVᴴ2 = collect(Vᴴ2)
        hNᴴ2 = collect(Nᴴ2)
        @test hVᴴ2' * hVᴴ2 + hNᴴ2' * hNᴴ2 ≈ I

        @testset "kind = $kind" for kind in (:lq, :polar, :svd)
            n < m && kind == :polar && continue
            C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ); kind = kind)
            @test C2 === C
            @test Vᴴ2 === Vᴴ
            @test C2 * Vᴴ2 ≈ A
            @test isisometric(Vᴴ2; side = :right)
            if kind != :polar
                Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ; kind = kind)
                @test Nᴴ2 === Nᴴ
                @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test isisometric(Nᴴ2; side = :right)
                hVᴴ2 = collect(Vᴴ2)
                hNᴴ2 = collect(Nᴴ2)
                @test hVᴴ2' * hVᴴ2 + hNᴴ2' * hNᴴ2 ≈ I
            end

            if kind == :svd
                C2, Vᴴ2 = @constinferred right_orth!(
                    copy!(Ac, A), (C, Vᴴ); kind = kind,
                    trunc = (; atol = atol)
                )
                Nᴴ2 = @constinferred right_null!(
                    copy!(Ac, A), Nᴴ; kind = kind,
                    trunc = (; atol = atol)
                )
                @test C2 !== C
                @test Vᴴ2 !== Vᴴ
                @test Nᴴ2 !== Nᴴ
                @test C2 * Vᴴ2 ≈ A
                @test isisometric(Vᴴ2; side = :right)
                @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test isisometric(Nᴴ2; side = :right)
                hVᴴ2 = collect(Vᴴ2)
                hNᴴ2 = collect(Nᴴ2)
                @test hVᴴ2' * hVᴴ2 + hNᴴ2' * hNᴴ2 ≈ I

                C2, Vᴴ2 = @constinferred right_orth!(
                    copy!(Ac, A), (C, Vᴴ); kind = kind,
                    trunc = (; rtol = rtol)
                )
                Nᴴ2 = @constinferred right_null!(
                    copy!(Ac, A), Nᴴ; kind = kind,
                    trunc = (; rtol = rtol)
                )
                @test C2 !== C
                @test Vᴴ2 !== Vᴴ
                @test Nᴴ2 !== Nᴴ
                @test C2 * Vᴴ2 ≈ A
                @test isisometric(Vᴴ2; side = :right)
                @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test isisometric(Nᴴ2; side = :right)
                hVᴴ2 = collect(Vᴴ2)
                hNᴴ2 = collect(Nᴴ2)
                @test hVᴴ2' * hVᴴ2 + hNᴴ2' * hNᴴ2 ≈ diagm(ones(T, size(Vᴴ2, 2))) atol = m * n * MatrixAlgebraKit.defaulttol(T)
            else
                @test_throws ArgumentError right_orth!(
                    copy!(Ac, A), (C, Vᴴ); kind = kind,
                    trunc = (; atol = atol)
                )
                @test_throws ArgumentError right_orth!(
                    copy!(Ac, A), (C, Vᴴ); kind = kind,
                    trunc = (; rtol = rtol)
                )
                @test_throws ArgumentError right_null!(
                    copy!(Ac, A), Nᴴ; kind = kind,
                    trunc = (; atol = atol)
                )
                @test_throws ArgumentError right_null!(
                    copy!(Ac, A), Nᴴ; kind = kind,
                    trunc = (; rtol = rtol)
                )
            end
        end
    end
end
