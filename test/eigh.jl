using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: LinearAlgebra, Diagonal, I
using MatrixAlgebraKit: diagview

@testset "eigh_full! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    for alg in (LAPACK_MultipleRelativelyRobustRepresentations(),
                LAPACK_DivideAndConquer(),
                LAPACK_QRIteration(),
                LAPACK_Bisection())
        A = randn(rng, T, m, m)
        A = (A + A') / 2

        D, V = @constinferred eigh_full(A; alg)
        @test A * V ≈ V * D
        @test V' * V ≈ I
        @test V * V' ≈ I
        @test all(isreal, D)

        D2, V2 = eigh_full!(copy(A), (D, V), alg)
        @test D2 === D
        @test V2 === V

        D3 = @constinferred eigh_vals(A, alg)
        @test D ≈ Diagonal(D3)
    end
end

@testset "eigh_trunc! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    for alg in (LAPACK_QRIteration(),
                LAPACK_Bisection(),
                LAPACK_DivideAndConquer(),
                LAPACK_MultipleRelativelyRobustRepresentations())
        A = randn(rng, T, m, m)
        A = A * A'
        A = (A + A') / 2
        Ac = similar(A)
        D₀ = reverse(eigh_vals(A))
        r = m - 2
        s = 1 + sqrt(eps(real(T)))

        D1, V1 = @constinferred eigh_trunc(A; alg, trunc=truncrank(r))
        @test length(diagview(D1)) == r
        @test V1' * V1 ≈ I
        @test A * V1 ≈ V1 * D1
        @test LinearAlgebra.opnorm(A - V1 * D1 * V1') ≈ D₀[r + 1]

        trunc = trunctol(s * D₀[r + 1])
        D2, V2 = @constinferred eigh_trunc(A; alg, trunc)
        @test length(diagview(D2)) == r
        @test V2' * V2 ≈ I
        @test A * V2 ≈ V2 * D2

        # test for same subspace
        @test V1 * (V1' * V2) ≈ V2
        @test V2 * (V2' * V1) ≈ V1
    end
end
