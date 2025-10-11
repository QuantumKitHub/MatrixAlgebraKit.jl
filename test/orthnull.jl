using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: LinearAlgebra, I

# testing non-AbstractArray codepaths:
include("linearmap.jl")

# testing non-AbstractArray codepaths:
include("linearmap.jl")

eltypes = (Float32, Float64, ComplexF32, ComplexF64)

@testset "left_orth and left_null for T = $T" for T in eltypes
    rng = StableRNG(123)
    m = 54
    for n in (37, m, 63)
        minmn = min(m, n)
        A = randn(rng, T, m, n)
        V, C = @constinferred left_orth(A)
        N = @constinferred left_null(A)
        @test V isa Matrix{T} && size(V) == (m, minmn)
        @test C isa Matrix{T} && size(C) == (minmn, n)
        @test N isa Matrix{T} && size(N) == (m, m - minmn)
        @test V * C ≈ A
        @test isisometric(V)
        @test LinearAlgebra.norm(A' * N) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(N)
        @test V * V' + N * N' ≈ I

        M = LinearMap(A)
        VM, CM = @constinferred left_orth(M; kind = :svd)
        @test parent(VM) * parent(CM) ≈ A

        if m > n
            nullity = 5
            V, C = @constinferred left_orth(A)
            N = @constinferred left_null(A; trunc = (; maxnullity = nullity))
            @test V isa Matrix{T} && size(V) == (m, minmn)
            @test C isa Matrix{T} && size(C) == (minmn, n)
            @test N isa Matrix{T} && size(N) == (m, nullity)
            @test V * C ≈ A
            @test isisometric(V)
            @test LinearAlgebra.norm(A' * N) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
            @test isisometric(N)
        end

        for alg_qr in ((; positive = true), (; positive = false), LAPACK_HouseholderQR())
            V, C = @constinferred left_orth(A; alg_qr)
            N = @constinferred left_null(A; alg_qr)
            @test V isa Matrix{T} && size(V) == (m, minmn)
            @test C isa Matrix{T} && size(C) == (minmn, n)
            @test N isa Matrix{T} && size(N) == (m, m - minmn)
            @test V * C ≈ A
            @test isisometric(V)
            @test LinearAlgebra.norm(A' * N) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
            @test isisometric(N)
            @test V * V' + N * N' ≈ I
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
        @test V2 * V2' + N2 * N2' ≈ I

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
        @test V2 * V2' + N2 * N2' ≈ I

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
            @test V2 * V2' + N2 * N2' ≈ I
        end

        for kind in (:qr, :polar, :svd) # explicit kind kwarg
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
                @test V2 * V2' + N2 * N2' ≈ I
            end

            # with kind and tol kwargs
            if kind == :svd
                V2, C2 = @constinferred left_orth!(copy!(Ac, A), (V, C); kind, trunc = (; atol))
                N2 = @constinferred left_null!(copy!(Ac, A), N; kind, trunc = (; atol))
                @test V2 !== V
                @test C2 !== C
                @test N2 !== C
                @test V2 * C2 ≈ A
                @test V2' * V2 ≈ I
                @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test N2' * N2 ≈ I
                @test V2 * V2' + N2 * N2' ≈ I

                V2, C2 = @constinferred left_orth!(copy!(Ac, A), (V, C); kind, trunc = (; rtol))
                N2 = @constinferred left_null!(copy!(Ac, A), N; kind, trunc = (; rtol))
                @test V2 !== V
                @test C2 !== C
                @test N2 !== C
                @test V2 * C2 ≈ A
                @test isisometric(V2)
                @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test isisometric(N2)
                @test V2 * V2' + N2 * N2' ≈ I
            else
                @test_throws ArgumentError left_orth!(copy!(Ac, A), (V, C); kind, trunc = (; atol))
                @test_throws ArgumentError left_orth!(copy!(Ac, A), (V, C); kind, trunc = (; rtol))
                @test_throws ArgumentError left_null!(copy!(Ac, A), N; kind, trunc = (; atol))
                @test_throws ArgumentError left_null!(copy!(Ac, A), N; kind, trunc = (; rtol))
            end
        end
    end
end

@testset "right_orth and right_null for T = $T" for T in eltypes
    rng = StableRNG(123)
    m = 54
    for n in (37, m, 63)
        minmn = min(m, n)
        A = randn(rng, T, m, n)
        C, Vᴴ = @constinferred right_orth(A)
        Nᴴ = @constinferred right_null(A)
        @test C isa Matrix{T} && size(C) == (m, minmn)
        @test Vᴴ isa Matrix{T} && size(Vᴴ) == (minmn, n)
        @test Nᴴ isa Matrix{T} && size(Nᴴ) == (n - minmn, n)
        @test C * Vᴴ ≈ A
        @test isisometric(Vᴴ; side = :right)
        @test LinearAlgebra.norm(A * adjoint(Nᴴ)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(Nᴴ; side = :right)
        @test Vᴴ' * Vᴴ + Nᴴ' * Nᴴ ≈ I

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
        @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I

        atol = eps(real(T))
        C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ); trunc = (; atol))
        Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ; trunc = (; atol))
        @test C2 !== C
        @test Vᴴ2 !== Vᴴ
        @test Nᴴ2 !== Nᴴ
        @test C2 * Vᴴ2 ≈ A
        @test isisometric(Vᴴ2; side = :right)
        @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(Nᴴ; side = :right)
        @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I

        rtol = eps(real(T))
        C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ); trunc = (; rtol))
        Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ; trunc = (; rtol))
        @test C2 !== C
        @test Vᴴ2 !== Vᴴ
        @test Nᴴ2 !== Nᴴ
        @test C2 * Vᴴ2 ≈ A
        @test isisometric(Vᴴ2; side = :right)
        @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(Nᴴ2; side = :right)
        @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I

        for kind in (:lq, :polar, :svd)
            n < m && kind == :polar && continue
            C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ); kind)
            @test C2 === C
            @test Vᴴ2 === Vᴴ
            @test C2 * Vᴴ2 ≈ A
            @test isisometric(Vᴴ2; side = :right)
            if kind != :polar
                Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ; kind)
                @test Nᴴ2 === Nᴴ
                @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test isisometric(Nᴴ2; side = :right)
                @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I
            end

            if kind == :svd
                C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ); kind, trunc = (; atol))
                Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ; kind, trunc = (; atol))
                @test C2 !== C
                @test Vᴴ2 !== Vᴴ
                @test Nᴴ2 !== Nᴴ
                @test C2 * Vᴴ2 ≈ A
                @test isisometric(Vᴴ2; side = :right)
                @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test isisometric(Nᴴ2; side = :right)
                @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I

                C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ); kind, trunc = (; rtol))
                Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ; kind, trunc = (; rtol))
                @test C2 !== C
                @test Vᴴ2 !== Vᴴ
                @test Nᴴ2 !== Nᴴ
                @test C2 * Vᴴ2 ≈ A
                @test isisometric(Vᴴ2; side = :right)
                @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test isisometric(Nᴴ2; side = :right)
                @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I
            else
                @test_throws ArgumentError right_orth!(copy!(Ac, A), (C, Vᴴ); kind, trunc = (; atol))
                @test_throws ArgumentError right_orth!(copy!(Ac, A), (C, Vᴴ); kind, trunc = (; rtol))
                @test_throws ArgumentError right_null!(copy!(Ac, A), Nᴴ; kind, trunc = (; atol))
                @test_throws ArgumentError right_null!(copy!(Ac, A), Nᴴ; kind, trunc = (; rtol))
            end
        end
    end
end
