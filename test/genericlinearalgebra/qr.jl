using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: diag, I, Diagonal
using GenericLinearAlgebra

eltypes = (BigFloat, Complex{BigFloat})

@testset "qr_compact! for T = $T" for T in eltypes

    rng = StableRNG(123)
    m = 54
    for n in (37, m, 63)
        minmn = min(m, n)
        m = 54
        A = randn(rng, T, m, n)
        Q, R = @constinferred qr_compact(A)
        @test Q isa Matrix{T} && size(Q) == (m, minmn)
        @test R isa Matrix{T} && size(R) == (minmn, n)
        @test Q * R ≈ A

        Ac = similar(A)
        Q2, R2 = @constinferred qr_compact!(copy!(Ac, A), (Q, R))
        @test Q2 === Q
        @test R2 === R

        Q2 = similar(Q)
        noR = similar(A, minmn, 0)
        qr_compact!(copy!(Ac, A), (Q2, noR))
        @test Q == Q2

        @test_throws ArgumentError qr_compact(A; blocksize = 2)
        @test_throws ArgumentError qr_compact(A; pivoted = true)

        # positive
        qr_compact!(copy!(Ac, A), (Q, R); positive = true)
        @test Q * R ≈ A
        @test isisometric(Q)
        @test all(>=(zero(real(T))), real(diag(R)))
        qr_compact!(copy!(Ac, A), (Q2, noR); positive = true)
        @test Q == Q2
    end
end

@testset "qr_full! for T = $T" for T in eltypes
    rng = StableRNG(123)
    m = 54
    for n in (37, m, 63)
        minmn = min(m, n)
        A = randn(rng, T, m, n)
        Q, R = qr_full(A)
        @test Q isa Matrix{T} && size(Q) == (m, m)
        @test R isa Matrix{T} && size(R) == (m, n)
        Qc, Rc = qr_compact(A)
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

        # Argument errors for unsupported options
        @test_throws ArgumentError qr_full(A; blocksize = 2)
        @test_throws ArgumentError qr_compact(A; pivoted = true)

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
        if n <= m
            # the following test tries to find the diagonal element (in order to test positivity)
            # before the column permutation. This only works if all columns have a diagonal
            # element
            for j in 1:n
                i = findlast(!iszero, view(R, :, j))
                @test real(R[i, j]) >= zero(real(T))
            end
        end
    end
end
