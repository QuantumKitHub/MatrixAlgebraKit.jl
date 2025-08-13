using MatrixAlgebraKit
using MatrixAlgebraKit: diagview
using Test
using TestExtras
using StableRNGs
using AMDGPU

include("utilities.jl")

@testset "qr_compact! and qr_null! for T = $T" for T in (Float32, Float64, ComplexF32,
                                                         ComplexF64)
    rng = StableRNG(123)
    m = 54
    for n in (37, m, 63)
        minmn = min(m, n)
        A = ROCArray(randn(rng, T, m, n))
        Q, R = @constinferred qr_compact(A)
        @test Q isa ROCMatrix{T} && size(Q) == (m, minmn)
        @test R isa ROCMatrix{T} && size(R) == (minmn, n)
        @test Q * R ≈ A
        N = @constinferred qr_null(A)
        @test N isa ROCMatrix{T} && size(N) == (m, m - minmn)
        @test isapproxone(Q' * Q)
        @test maximum(abs, A' * N) < eps(real(T))^(2 / 3)
        @test isapproxone(N' * N)

        Ac = similar(A)
        Q2, R2 = @constinferred qr_compact!(copy!(Ac, A), (Q, R))
        @test Q2 === Q
        @test R2 === R
        N2 = @constinferred qr_null!(copy!(Ac, A), N)
        @test N2 === N

        # noR
        Q2 = similar(Q)
        noR = similar(A, minmn, 0)
        qr_compact!(copy!(Ac, A), (Q2, noR))
        @test Q == Q2

        # positive
        qr_compact!(copy!(Ac, A), (Q, R); positive=true)
        @test Q * R ≈ A
        @test isapproxone(Q' * Q)
        @test all(>=(zero(real(T))), real(diagview(R)))
        qr_compact!(copy!(Ac, A), (Q2, noR); positive=true)
        @test Q == Q2

        # explicit blocksize
        qr_compact!(copy!(Ac, A), (Q, R); blocksize=1)
        @test Q * R ≈ A
        @test isapproxone(Q' * Q)
        qr_compact!(copy!(Ac, A), (Q2, noR); blocksize=1)
        @test Q == Q2
        qr_compact!(copy!(Ac, A), (Q2, noR); blocksize=1)
        qr_null!(copy!(Ac, A), N; blocksize=1)
        @test maximum(abs, A' * N) < eps(real(T))^(2 / 3)
        @test isapproxone(N' * N)
        if n <= m
            qr_compact!(copy!(Q2, A), (Q2, noR); blocksize=1) # in-place Q
            @test Q ≈ Q2
            # these do not work because of the in-place Q
            @test_throws ArgumentError qr_compact!(copy!(Q2, A), (Q2, R2))
            @test_throws ArgumentError qr_compact!(copy!(Q2, A), (Q2, noR); positive=true)
        end
        # no blocked CUDA
        @test_throws ArgumentError qr_compact!(copy!(Ac, A), (Q2, R); blocksize=8)
    end
end

@testset "qr_full! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 63
    for n in (37, m, 63)
        minmn = min(m, n)
        A = ROCArray(randn(rng, T, m, n))
        Q, R = qr_full(A)
        @test Q isa ROCMatrix{T} && size(Q) == (m, m)
        @test R isa ROCMatrix{T} && size(R) == (m, n)
        @test Q * R ≈ A
        @test isapproxone(Q' * Q)

        Ac = similar(A)
        Q2 = similar(Q)
        noR = similar(A, m, 0)
        Q2, R2 = @constinferred qr_full!(copy!(Ac, A), (Q, R))
        @test Q2 === Q
        @test R2 === R
        @test Q * R ≈ A
        @test isapproxone(Q' * Q)
        qr_full!(copy!(Ac, A), (Q2, noR))
        @test Q == Q2

        # noR
        noR = similar(A, m, 0)
        Q2 = similar(Q)
        qr_full!(copy!(Ac, A), (Q2, noR))
        @test Q == Q2

        # positive
        qr_full!(copy!(Ac, A), (Q, R); positive=true)
        @test Q * R ≈ A
        @test isapproxone(Q' * Q)
        @test all(>=(zero(real(T))), real(diagview(R)))
        qr_full!(copy!(Ac, A), (Q2, noR); positive=true)
        @test Q == Q2

        # explicit blocksize
        qr_full!(copy!(Ac, A), (Q, R); blocksize=1)
        @test Q * R ≈ A
        @test isapproxone(Q' * Q)
        qr_full!(copy!(Ac, A), (Q2, noR); blocksize=1)
        @test Q == Q2
        if n == m
            qr_full!(copy!(Q2, A), (Q2, noR); blocksize=1) # in-place Q
            @test Q ≈ Q2
            @test_throws ArgumentError qr_full!(copy!(Q2, A), (Q2, R2))
            @test_throws ArgumentError qr_full!(copy!(Q2, A), (Q2, noR); positive=true)
        end
        # no blocked CUDA
        @test_throws ArgumentError qr_full!(copy!(Ac, A), (Q, R); blocksize=8)
    end
end
