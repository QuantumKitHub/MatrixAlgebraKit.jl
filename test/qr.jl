using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: diag, I, Diagonal

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
GenericFloats = (Float16, BigFloat, Complex{BigFloat})

@testset "qr_compact! and qr_null! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54
    for n in (37, m, 63)
        minmn = min(m, n)
        A = randn(rng, T, m, n)
        Q, R = @constinferred qr_compact(A)
        @test Q isa Matrix{T} && size(Q) == (m, minmn)
        @test R isa Matrix{T} && size(R) == (minmn, n)
        @test Q * R ≈ A
        N = @constinferred qr_null(A)
        @test N isa Matrix{T} && size(N) == (m, m - minmn)
        @test isisometric(Q)
        @test maximum(abs, A' * N) < eps(real(T))^(2 / 3)
        @test isisometric(N)

        Ac = similar(A)
        Q2, R2 = @constinferred qr_compact!(copy!(Ac, A), (Q, R))
        @test Q2 === Q
        @test R2 === R
        N2 = @constinferred qr_null!(copy!(Ac, A), N)
        @test N2 === N

        Q2 = similar(Q)
        noR = similar(A, minmn, 0)
        qr_compact!(copy!(Ac, A), (Q2, noR))
        @test Q == Q2

        # unblocked algorithm
        qr_compact!(copy!(Ac, A), (Q, R); blocksize = 1)
        @test Q * R ≈ A
        @test isisometric(Q)
        qr_compact!(copy!(Ac, A), (Q2, noR); blocksize = 1)
        @test Q == Q2
        qr_compact!(copy!(Ac, A), (Q2, noR); blocksize = 1)
        qr_null!(copy!(Ac, A), N; blocksize = 1)
        @test maximum(abs, A' * N) < eps(real(T))^(2 / 3)
        @test isisometric(N)
        if n <= m
            qr_compact!(copy!(Q2, A), (Q2, noR); blocksize = 1) # in-place Q
            @test Q ≈ Q2
            @test_throws ArgumentError qr_compact!(copy!(Q2, A), (Q2, R); blocksize = 1)
            @test_throws ArgumentError qr_compact!(copy!(Q2, A), (Q2, noR); positive = true)
            @test_throws ArgumentError qr_compact!(copy!(Q2, A), (Q2, noR); blocksize = 8)
        end
        # other blocking
        qr_compact!(copy!(Ac, A), (Q, R); blocksize = 8)
        @test Q * R ≈ A
        @test isisometric(Q)
        qr_compact!(copy!(Ac, A), (Q2, noR); blocksize = 8)
        @test Q == Q2
        qr_null!(copy!(Ac, A), N; blocksize = 8)
        @test maximum(abs, A' * N) < eps(real(T))^(2 / 3)
        @test isisometric(N)

        # pivoted
        qr_compact!(copy!(Ac, A), (Q, R); pivoted = true)
        @test Q * R ≈ A
        @test Q' * Q ≈ I
        qr_compact!(copy!(Ac, A), (Q2, noR); pivoted = true)
        @test Q == Q2
        # positive
        qr_compact!(copy!(Ac, A), (Q, R); positive = true)
        @test Q * R ≈ A
        @test isisometric(Q)
        @test all(>=(zero(real(T))), real(diag(R)))
        qr_compact!(copy!(Ac, A), (Q2, noR); positive = true)
        @test Q == Q2
        # positive and blocksize 1
        qr_compact!(copy!(Ac, A), (Q, R); positive = true, blocksize = 1)
        @test Q * R ≈ A
        @test isisometric(Q)
        @test all(>=(zero(real(T))), real(diag(R)))
        qr_compact!(copy!(Ac, A), (Q2, noR); positive = true, blocksize = 1)
        @test Q == Q2
        # positive and pivoted
        qr_compact!(copy!(Ac, A), (Q, R); positive = true, pivoted = true)
        @test Q * R ≈ A
        @test isisometric(Q)
        if n <= m
            # the following test tries to find the diagonal element (in order to test positivity)
            # before the column permutation. This only works if all columns have a diagonal
            # element
            for j in 1:n
                i = findlast(!iszero, view(R, :, j))
                @test real(R[i, j]) >= zero(real(T))
            end
        end
        qr_compact!(copy!(Ac, A), (Q2, noR); positive = true, pivoted = true)
        @test Q == Q2
    end
end

@testset "qr_full! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54
    for n in (37, m, 63)
        minmn = min(m, n)
        A = randn(rng, T, m, n)
        Q, R = qr_full(A)
        @test Q isa Matrix{T} && size(Q) == (m, m)
        @test R isa Matrix{T} && size(R) == (m, n)
        @test Q * R ≈ A
        @test isunitary(Q)

        Ac = similar(A)
        Q2 = similar(Q)
        noR = similar(A, m, 0)
        Q2, R2 = @constinferred qr_full!(copy!(Ac, A), (Q, R))
        @test Q2 === Q
        @test R2 === R
        @test Q * R ≈ A
        @test isunitary(Q)
        qr_full!(copy!(Ac, A), (Q2, noR))
        @test Q == Q2

        # unblocked algorithm
        qr_full!(copy!(Ac, A), (Q, R); blocksize = 1)
        @test Q * R ≈ A
        @test isunitary(Q)
        qr_full!(copy!(Ac, A), (Q2, noR); blocksize = 1)
        @test Q == Q2
        if n == m
            qr_full!(copy!(Q2, A), (Q2, noR); blocksize = 1) # in-place Q
            @test Q ≈ Q2
        end
        # other blocking
        qr_full!(copy!(Ac, A), (Q, R); blocksize = 8)
        @test Q * R ≈ A
        @test isunitary(Q)
        qr_full!(copy!(Ac, A), (Q2, noR); blocksize = 8)
        @test Q == Q2
        # pivoted
        qr_full!(copy!(Ac, A), (Q, R); pivoted = true)
        @test Q * R ≈ A
        @test isunitary(Q)
        qr_full!(copy!(Ac, A), (Q2, noR); pivoted = true)
        @test Q == Q2
        # positive
        qr_full!(copy!(Ac, A), (Q, R); positive = true)
        @test Q * R ≈ A
        @test isunitary(Q)
        @test all(>=(zero(real(T))), real(diag(R)))
        qr_full!(copy!(Ac, A), (Q2, noR); positive = true)
        @test Q == Q2
        # positive and blocksize 1
        qr_full!(copy!(Ac, A), (Q, R); positive = true, blocksize = 1)
        @test Q * R ≈ A
        @test isunitary(Q)
        @test all(>=(zero(real(T))), real(diag(R)))
        qr_full!(copy!(Ac, A), (Q2, noR); positive = true, blocksize = 1)
        @test Q == Q2
        # positive and pivoted
        qr_full!(copy!(Ac, A), (Q, R); positive = true, pivoted = true)
        @test Q * R ≈ A
        @test isunitary(Q)
        if n <= m
            # the following test tries to find the diagonal element (in order to test positivity)
            # before the column permutation. This only works if all columns have a diagonal
            # element
            for j in 1:n
                i = findlast(!iszero, view(R, :, j))
                @test real(R[i, j]) >= zero(real(T))
            end
        end
        qr_full!(copy!(Ac, A), (Q2, noR); positive = true, pivoted = true)
        @test Q == Q2
    end
end

@testset "qr_compact, qr_full and qr_null for Diagonal{$T}" for T in (BLASFloats..., GenericFloats...)
    rng = StableRNG(123)
    atol = eps(real(T))^(3 / 4)
    for m in (54, 0)
        Ad = randn(rng, T, m)
        A = Diagonal(Ad)

        # compact
        Q, R = @constinferred qr_compact(A)
        @test Q isa Diagonal{T} && size(Q) == (m, m)
        @test R isa Diagonal{T} && size(R) == (m, m)
        @test Q * R ≈ A
        @test isunitary(Q)

        # compact and positive
        Qp, Rp = @constinferred qr_compact(A; positive = true)
        @test Qp isa Diagonal{T} && size(Qp) == (m, m)
        @test Rp isa Diagonal{T} && size(Rp) == (m, m)
        @test Qp * Rp ≈ A
        @test isunitary(Qp)
        @test all(≥(zero(real(T))), real(diag(Rp))) &&
            all(≈(zero(real(T)); atol), imag(diag(Rp)))

        # full
        Q, R = @constinferred qr_full(A)
        @test Q isa Diagonal{T} && size(Q) == (m, m)
        @test R isa Diagonal{T} && size(R) == (m, m)
        @test Q * R ≈ A
        @test isunitary(Q)

        # full and positive
        Qp, Rp = @constinferred qr_full(A; positive = true)
        @test Qp isa Diagonal{T} && size(Qp) == (m, m)
        @test Rp isa Diagonal{T} && size(Rp) == (m, m)
        @test Qp * Rp ≈ A
        @test isunitary(Qp)
        @test all(≥(zero(real(T))), real(diag(Rp))) &&
            all(≈(zero(real(T)); atol), imag(diag(Rp)))

        # null
        N = @constinferred qr_null(A)
        @test N isa AbstractMatrix{T} && size(N) == (m, 0)
    end
end
