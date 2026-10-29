using MatrixAlgebraKit
using MatrixAlgebraKit: Native
using LinearAlgebra: Diagonal
using Test

@testset "Orthogonalization without a basis" begin
    for T in (Float64, ComplexF64), (m, n) in ((7, 4), (4, 4), (4, 7))
        A = randn(T, m, n)
        Q0 = similar(A, 0, 0)
        for positive in (false, true), alg in (
                    Householder(; blocksize = 1, positive),
                    Householder(; blocksize = 2, positive),
                    Householder(; blocksize = 1, pivoted = true, positive),
                    Householder(; driver = Native(), positive),
                )
            Q, R = left_orth(A; alg)
            Q2, R2 = left_orth!(copy(A), (Q0, similar(R)), alg)
            @test Q2 === Q0
            @test R2 ≈ R
            @test Q * R2 ≈ A

            # LQ pivoting is supported through transposed QR.
            lq_alg = LQViaTransposedQR(alg)
            L, Q = right_orth(A; alg = lq_alg)
            L2, Q2 = right_orth!(copy(A), (similar(L), Q0), lq_alg)
            @test Q2 === Q0
            @test L2 ≈ L
            @test L2 * Q ≈ A
        end
        for positive in (false, true), blocksize in (1, 2)
            L, Q = right_orth(A; alg = :lq, positive, blocksize)
            L2, Q2 = right_orth!(copy(A), (similar(L), Q0); alg = :lq, positive, blocksize)
            @test Q2 === Q0
            @test L2 ≈ L
            @test L2 * Q ≈ A
        end
    end
    for T in (Float64, ComplexF64), positive in (false, true)
        A = Diagonal(randn(T, 4))
        Q0 = Diagonal(T[])
        Q, R = left_orth(A; positive)
        Q2, R2 = left_orth!(copy(A), (Q0, similar(R)); positive)
        @test Q2 === Q0
        @test Q * R2 ≈ A
        L, Q = right_orth(A; positive)
        L2, Q2 = right_orth!(copy(A), (similar(L), Q0); positive)
        @test Q2 === Q0
        @test L2 * Q ≈ A
    end
end
