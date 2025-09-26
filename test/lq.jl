using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: diag, I, Diagonal
using MatrixAlgebraKit: LQViaTransposedQR, LAPACK_HouseholderQR

eltypes = (Float32, Float64, ComplexF32, ComplexF64)

@testset "lq_compact! for T = $T" for T in eltypes
    rng = StableRNG(123)
    m = 54
    for n in (37, m, 63)
        minmn = min(m, n)
        A = randn(rng, T, m, n)
        L, Q = @constinferred lq_compact(A)
        @test L isa Matrix{T} && size(L) == (m, minmn)
        @test Q isa Matrix{T} && size(Q) == (minmn, n)
        @test L * Q ≈ A
        @test isisometry(Q; side = :right)
        Nᴴ = @constinferred lq_null(A)
        @test Nᴴ isa Matrix{T} && size(Nᴴ) == (n - minmn, n)
        @test maximum(abs, A * Nᴴ') < eps(real(T))^(2 / 3)
        @test isisometry(Nᴴ; side = :right)

        Ac = similar(A)
        L2, Q2 = @constinferred lq_compact!(copy!(Ac, A), (L, Q))
        @test L2 === L
        @test Q2 === Q
        Nᴴ2 = @constinferred lq_null!(copy!(Ac, A), Nᴴ)
        @test Nᴴ2 === Nᴴ

        noL = similar(A, 0, minmn)
        Q2 = similar(Q)
        lq_compact!(copy!(Ac, A), (noL, Q2))
        @test Q == Q2

        # Transposed QR algorithm
        qr_alg = LAPACK_HouseholderQR()
        lq_alg = LQViaTransposedQR(qr_alg)
        L2, Q2 = @constinferred lq_compact!(copy!(Ac, A), (L, Q), lq_alg)
        @test L2 === L
        @test Q2 === Q
        Nᴴ2 = @constinferred lq_null!(copy!(Ac, A), Nᴴ, lq_alg)
        @test Nᴴ2 === Nᴴ
        noL = similar(A, 0, minmn)
        Q2 = similar(Q)
        lq_compact!(copy!(Ac, A), (noL, Q2), lq_alg)
        @test Q == Q2

        # unblocked algorithm
        lq_compact!(copy!(Ac, A), (L, Q); blocksize = 1)
        @test L * Q ≈ A
        @test isisometry(Q; side = :right)
        lq_compact!(copy!(Ac, A), (noL, Q2); blocksize = 1)
        @test Q == Q2
        lq_null!(copy!(Ac, A), Nᴴ; blocksize = 1)
        @test maximum(abs, A * Nᴴ') < eps(real(T))^(2 / 3)
        @test isisometry(Nᴴ; side = :right)
        if m <= n
            lq_compact!(copy!(Q2, A), (noL, Q2); blocksize = 1) # in-place Q
            @test Q ≈ Q2
            @test_throws ArgumentError lq_compact!(copy!(Q2, A), (L, Q2); blocksize = 1)
            @test_throws ArgumentError lq_compact!(copy!(Q2, A), (noL, Q2); positive = true)
            @test_throws ArgumentError lq_compact!(copy!(Q2, A), (noL, Q2); blocksize = 8)
        end
        lq_compact!(copy!(Ac, A), (L, Q); blocksize = 8)
        @test L * Q ≈ A
        @test isisometry(Q; side = :right)
        lq_compact!(copy!(Ac, A), (noL, Q2); blocksize = 8)
        @test Q == Q2
        lq_null!(copy!(Ac, A), Nᴴ; blocksize = 8)
        @test maximum(abs, A * Nᴴ') < eps(real(T))^(2 / 3)
        @test isisometry(Nᴴ; side = :right)
        @test Nᴴ * Nᴴ' ≈ I

        qr_alg = LAPACK_HouseholderQR(; blocksize = 1)
        lq_alg = LQViaTransposedQR(qr_alg)
        lq_compact!(copy!(Ac, A), (L, Q), lq_alg)
        @test L * Q ≈ A
        @test Q * Q' ≈ I
        lq_compact!(copy!(Ac, A), (noL, Q2), lq_alg)
        @test Q == Q2
        lq_null!(copy!(Ac, A), Nᴴ, lq_alg)
        @test maximum(abs, A * Nᴴ') < eps(real(T))^(2 / 3)
        @test Nᴴ * Nᴴ' ≈ I

        # pivoted
        @test_throws ArgumentError lq_compact!(copy!(Ac, A), (L, Q); pivoted = true)

        # positive
        lq_compact!(copy!(Ac, A), (L, Q); positive = true)
        @test L * Q ≈ A
        @test isisometry(Q; side = :right)
        @test all(>=(zero(real(T))), real(diag(L)))
        lq_compact!(copy!(Ac, A), (noL, Q2); positive = true)
        @test Q == Q2

        # positive and blocksize 1
        lq_compact!(copy!(Ac, A), (L, Q); positive = true, blocksize = 1)
        @test L * Q ≈ A
        @test isisometry(Q; side = :right)
        @test all(>=(zero(real(T))), real(diag(L)))
        lq_compact!(copy!(Ac, A), (noL, Q2); positive = true, blocksize = 1)
        @test Q == Q2
        qr_alg = LAPACK_HouseholderQR(; positive = true, blocksize = 1)
        lq_alg = LQViaTransposedQR(qr_alg)
        lq_compact!(copy!(Ac, A), (L, Q), lq_alg)
        @test L * Q ≈ A
        @test Q * Q' ≈ I
        @test all(>=(zero(real(T))), real(diag(L)))
        lq_compact!(copy!(Ac, A), (noL, Q2), lq_alg)
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
        @test Q == Q2

        # Transposed QR algorithm
        qr_alg = LAPACK_HouseholderQR()
        lq_alg = LQViaTransposedQR(qr_alg)
        L2, Q2 = @constinferred lq_full!(copy!(Ac, A), (L, Q), lq_alg)
        @test L2 === L
        @test Q2 === Q
        @test L * Q ≈ A
        @test Q * Q' ≈ I
        noL = similar(A, 0, n)
        Q2 = similar(Q)
        lq_full!(copy!(Ac, A), (noL, Q2), lq_alg)
        @test Q == Q2

        # unblocked algorithm
        lq_full!(copy!(Ac, A), (L, Q); blocksize = 1)
        @test L * Q ≈ A
        @test isunitary(Q)
        lq_full!(copy!(Ac, A), (noL, Q2); blocksize = 1)
        @test Q == Q2
        if n == m
            lq_full!(copy!(Q2, A), (noL, Q2); blocksize = 1) # in-place Q
            @test Q ≈ Q2
        end
        qr_alg = LAPACK_HouseholderQR(; blocksize = 1)
        lq_alg = LQViaTransposedQR(qr_alg)
        lq_full!(copy!(Ac, A), (L, Q), lq_alg)
        @test L * Q ≈ A
        @test Q * Q' ≈ I
        lq_full!(copy!(Ac, A), (noL, Q2), lq_alg)
        @test Q == Q2
        if n == m
            lq_full!(copy!(Q2, A), (noL, Q2), lq_alg) # in-place Q
            @test Q ≈ Q2
        end

        # other blocking
        lq_full!(copy!(Ac, A), (L, Q); blocksize = 18)
        @test L * Q ≈ A
        @test isunitary(Q)
        lq_full!(copy!(Ac, A), (noL, Q2); blocksize = 18)
        @test Q == Q2
        # pivoted
        @test_throws ArgumentError lq_full!(copy!(Ac, A), (L, Q); pivoted = true)
        # positive
        lq_full!(copy!(Ac, A), (L, Q); positive = true)
        @test L * Q ≈ A
        @test isunitary(Q)
        @test all(>=(zero(real(T))), real(diag(L)))
        lq_full!(copy!(Ac, A), (noL, Q2); positive = true)
        @test Q == Q2

        qr_alg = LAPACK_HouseholderQR(; positive = true)
        lq_alg = LQViaTransposedQR(qr_alg)
        lq_full!(copy!(Ac, A), (L, Q), lq_alg)
        @test L * Q ≈ A
        @test Q * Q' ≈ I
        @test all(>=(zero(real(T))), real(diag(L)))
        lq_full!(copy!(Ac, A), (noL, Q2), lq_alg)
        @test Q == Q2

        # positive and blocksize 1
        lq_full!(copy!(Ac, A), (L, Q); positive = true, blocksize = 1)
        @test L * Q ≈ A
        @test isunitary(Q)
        @test all(>=(zero(real(T))), real(diag(L)))
        lq_full!(copy!(Ac, A), (noL, Q2); positive = true, blocksize = 1)
        @test Q == Q2
    end
end

@testset "lq_compact, lq_full and lq_null for Diagonal{$T}" for T in eltypes
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

        # full
        L, Q = @constinferred lq_full(A)
        @test Q isa Diagonal{T} && size(Q) == (m, m)
        @test L isa Diagonal{T} && size(L) == (m, m)
        @test L * Q ≈ A
        @test isunitary(Q)

        # full and positive
        Lp, Qp = @constinferred lq_full(A; positive = true)
        @test Qp isa Diagonal{T} && size(Qp) == (m, m)
        @test Lp isa Diagonal{T} && size(Lp) == (m, m)
        @test Lp * Qp ≈ A
        @test isunitary(Qp)
        @test all(≥(zero(real(T))), real(diag(Lp))) &&
            all(≈(zero(real(T)); atol), imag(diag(Lp)))

        # null
        N = @constinferred lq_null(A)
        @test N isa AbstractMatrix{T} && size(N) == (0, m)
    end
end
