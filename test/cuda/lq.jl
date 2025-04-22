using MatrixAlgebraKit
using MatrixAlgebraKit: diagview
using Test
using TestExtras
using StableRNGs
using CUDA

function isapproxone(A)
    return (size(A, 1) == size(A, 2)) && (A ≈ MatrixAlgebraKit.one!(similar(A)))
end

@testset "lq_compact! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    for n in (37, m, 63)
        minmn = min(m, n)
        A = CuArray(randn(rng, T, m, n))
        L, Q = @constinferred lq_compact(A)
        @test L isa CuMatrix{T} && size(L) == (m, minmn)
        @test Q isa CuMatrix{T} && size(Q) == (minmn, n)
        @test L * Q ≈ A
        @test isapproxone(Q * Q')
        Nᴴ = @constinferred lq_null(A)
        @test Nᴴ isa CuMatrix{T} && size(Nᴴ) == (n - minmn, n)
        @test maximum(abs, A * Nᴴ') < eps(real(T))^(2 / 3)
        @test isapproxone(Nᴴ * Nᴴ')

        Ac = similar(A)
        L2, Q2 = @constinferred lq_compact!(copy!(Ac, A), (L, Q))
        @test L2 === L
        @test Q2 === Q
        Nᴴ2 = @constinferred lq_null!(copy!(Ac, A), Nᴴ)
        @test Nᴴ2 === Nᴴ

        # noL
        noL = similar(A, 0, minmn)
        Q2 = similar(Q)
        lq_compact!(copy!(Ac, A), (noL, Q2))
        @test Q == Q2

        # positive
        lq_compact!(copy!(Ac, A), (L, Q); positive=true)
        @test L * Q ≈ A
        @test isapproxone(Q * Q')
        @test all(>=(zero(real(T))), real(diagview(L)))
        lq_compact!(copy!(Ac, A), (noL, Q2); positive=true)
        @test Q == Q2

        # explicit blocksize
        lq_compact!(copy!(Ac, A), (L, Q); blocksize=1)
        @test L * Q ≈ A
        @test isapproxone(Q * Q')
        lq_compact!(copy!(Ac, A), (noL, Q2); blocksize=1)
        @test Q == Q2
        lq_null!(copy!(Ac, A), Nᴴ; blocksize=1)
        @test maximum(abs, A * Nᴴ') < eps(real(T))^(2 / 3)
        @test isapproxone(Nᴴ * Nᴴ')
        if m <= n
            lq_compact!(copy!(Q2, A), (noL, Q2); blocksize=1) # in-place Q
            @test Q ≈ Q2
            # these do not work because of the in-place Q
            @test_throws ArgumentError lq_compact!(copy!(Q2, A), (L, Q2); blocksize=1)
            @test_throws ArgumentError lq_compact!(copy!(Q2, A), (noL, Q2); positive=true)
        end
        # no blocked CUDA
        @test_throws ArgumentError lq_compact!(copy!(Q2, A), (L, Q2); blocksize=8)
    end
end

@testset "lq_full! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    for n in (37, m, 63)
        minmn = min(m, n)
        A = CuArray(randn(rng, T, m, n))
        L, Q = lq_full(A)
        @test L isa CuMatrix{T} && size(L) == (m, n)
        @test Q isa CuMatrix{T} && size(Q) == (n, n)
        @test L * Q ≈ A
        @test isapproxone(Q * Q')

        Ac = similar(A)
        L2, Q2 = @constinferred lq_full!(copy!(Ac, A), (L, Q))
        @test L2 === L
        @test Q2 === Q
        @test L * Q ≈ A
        @test isapproxone(Q * Q')

        # noL
        noL = similar(A, 0, n)
        Q2 = similar(Q)
        lq_full!(copy!(Ac, A), (noL, Q2))
        @test Q == Q2

        # positive
        lq_full!(copy!(Ac, A), (L, Q); positive=true)
        @test L * Q ≈ A
        @test isapproxone(Q * Q')
        @test all(>=(zero(real(T))), real(diagview(L)))
        lq_full!(copy!(Ac, A), (noL, Q2); positive=true)
        @test Q == Q2

        # explicit blocksize
        lq_full!(copy!(Ac, A), (L, Q); blocksize=1)
        @test L * Q ≈ A
        @test isapproxone(Q * Q')
        lq_full!(copy!(Ac, A), (noL, Q2); blocksize=1)
        @test Q == Q2
        if n == m
            lq_full!(copy!(Q2, A), (noL, Q2); blocksize=1) # in-place Q
            @test Q ≈ Q2
            # these do not work because of the in-place Q
            @test_throws ArgumentError lq_full!(copy!(Q2, A), (L, Q2); blocksize=1)
            @test_throws ArgumentError lq_full!(copy!(Q2, A), (noL, Q2); positive=true)
        end
        # no blocked CUDA
        @test_throws ArgumentError lq_full!(copy!(Ac, A), (L, Q); blocksize=8)
    end
end
