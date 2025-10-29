using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: diag, I, Diagonal
using GenericLinearAlgebra
using GenericSchur

eltypes = (BigFloat, Complex{BigFloat})

@testset "qr_compact! for T = $T" for T in eltypes

    rng = StableRNG(123)
    m = 54
    for n in (37, m, 63)
        minmn = min(m, n)
        m = 54
        A = randn(rng, T, m, n)
        L, Q = @constinferred lq_compact(A)
        @test L isa Matrix{T} && size(L) == (m, minmn)
        @test Q isa Matrix{T} && size(Q) == (minmn, n)
        @test L * Q ≈ A
        @test isisometric(Q; side = :right)

        Ac = similar(A)
        L2, Q2 = @constinferred lq_compact!(copy!(Ac, A), (L, Q))
        @test L2 === L
        @test Q2 === Q

        noL = similar(A, 0, minmn)
        Q2 = similar(Q)
        lq_compact!(copy!(Ac, A), (noL, Q2))
        @test Q == Q2

        # Transposed QR algorithm
        qr_alg = BigFloat_QR_Householder()
        lq_alg = LQViaTransposedQR(qr_alg)
        L2, Q2 = @constinferred lq_compact!(copy!(Ac, A), (L, Q), lq_alg)
        @test L2 === L
        @test Q2 === Q
        noL = similar(A, 0, minmn)
        Q2 = similar(Q)
        lq_compact!(copy!(Ac, A), (noL, Q2), lq_alg)
        @test Q == Q2

        @test_throws ArgumentError lq_compact(A; blocksize = 2)
        @test_throws ArgumentError lq_compact(A; pivoted = true)

        # positive
        lq_compact!(copy!(Ac, A), (L, Q); positive = true)
        @test L * Q ≈ A
        @test isisometric(Q; side = :right)
        @test all(>=(zero(real(T))), real(diag(L)))
        lq_compact!(copy!(Ac, A), (noL, Q2); positive = true)
        @test Q == Q2
    end
end

@testset "lq_full! for T = $T" for T in eltypes
    rng = StableRNG(123)
    m = 54
    for n in (37, m, 63)
        minmn = min(m, n)
        A = randn(rng, T, m, n)
        L, Q = lq_full(A)
        @test L isa Matrix{T} && size(L) == (m, n)
        @test Q isa Matrix{T} && size(Q) == (n, n)
        @test L * Q ≈ A
        @test isunitary(Q)

        Ac = similar(A)
        L2, Q2 = @constinferred lq_full!(copy!(Ac, A), (L, Q))
        @test L2 === L
        @test Q2 === Q
        @test L * Q ≈ A
        @test isunitary(Q)

        noL = similar(A, 0, n)
        Q2 = similar(Q)
        lq_full!(copy!(Ac, A), (noL, Q2))
        @test Q[1:minmn, n] ≈ Q2[1:minmn, n]

        # Transposed QR algorithm
        qr_alg = BigFloat_QR_Householder()
        lq_alg = LQViaTransposedQR(qr_alg)
        L2, Q2 = @constinferred lq_full!(copy!(Ac, A), (L, Q), lq_alg)
        @test L2 === L
        @test Q2 === Q
        @test L * Q ≈ A
        @test Q * Q' ≈ I
        noL = similar(A, 0, n)
        Q2 = similar(Q)
        lq_full!(copy!(Ac, A), (noL, Q2), lq_alg)
        @test Q[1:minmn, n] ≈ Q2[1:minmn, n]

        # Argument errors for unsupported options
        @test_throws ArgumentError lq_full(A; blocksize = 2)
        @test_throws ArgumentError lq_full(A; pivoted = true)

        # positive
        lq_full!(copy!(Ac, A), (L, Q); positive = true)
        @test L * Q ≈ A
        @test isunitary(Q)
        @test all(>=(zero(real(T))), real(diag(L)))
        lq_full!(copy!(Ac, A), (noL, Q2); positive = true)
        @test Q[1:minmn, n] ≈ Q2[1:minmn, n]

        qr_alg = BigFloat_QR_Householder(; positive = true)
        lq_alg = LQViaTransposedQR(qr_alg)
        lq_full!(copy!(Ac, A), (L, Q), lq_alg)
        @test L * Q ≈ A
        @test Q * Q' ≈ I
        @test all(>=(zero(real(T))), real(diag(L)))
        lq_full!(copy!(Ac, A), (noL, Q2), lq_alg)
        @test Q[1:minmn, n] ≈ Q2[1:minmn, n]

        # positive and blocksize 1
        lq_full!(copy!(Ac, A), (L, Q); positive = true, blocksize = 1)
        @test L * Q ≈ A
        @test isunitary(Q)
        @test all(>=(zero(real(T))), real(diag(L)))
        lq_full!(copy!(Ac, A), (noL, Q2); positive = true, blocksize = 1)
        @test Q[1:minmn, n] ≈ Q2[1:minmn, n]
    end
end

@testset "lq_compact for Diagonal{$T}" for T in eltypes
    rng = StableRNG(123)
    atol = eps(real(T))^(3 / 4)
    for m in (54, 0)
        Ad = randn(rng, T, m)
        A = Diagonal(Ad)

        # compact
        L, Q = @constinferred lq_compact(A)
        @test Q isa Diagonal{T} && size(Q) == (m, m)
        @test L isa Diagonal{T} && size(L) == (m, m)
        @test L * Q ≈ A
        @test isunitary(Q)

        # compact and positive
        Lp, Qp = @constinferred lq_compact(A; positive = true)
        @test Qp isa Diagonal{T} && size(Qp) == (m, m)
        @test Lp isa Diagonal{T} && size(Lp) == (m, m)
        @test Lp * Qp ≈ A
        @test isunitary(Qp)
        @test all(≥(zero(real(T))), real(diag(Lp))) &&
            all(≈(zero(real(T)); atol), imag(diag(Lp)))
    end
end
