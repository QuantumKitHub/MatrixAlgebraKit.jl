using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: LinearAlgebra, Diagonal, I
using MatrixAlgebraKit: TruncatedAlgorithm, diagview, norm

const BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)

@testset "eigh_full! for T = $T" for T in BLASFloats
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
        @test isunitary(V)
        @test all(isreal, D)

        D2, V2 = eigh_full!(copy(A), (D, V), alg)
        @test D2 === D
        @test V2 === V

        D3 = @constinferred eigh_vals(A, alg)
        @test D ≈ Diagonal(D3)
    end
end

@testset "eigh_trunc! for T = $T" for T in BLASFloats
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
        @test isisometry(V1)
        @test A * V1 ≈ V1 * D1
        @test LinearAlgebra.opnorm(A - V1 * D1 * V1') ≈ D₀[r + 1]

        trunc = trunctol(s * D₀[r + 1])
        D2, V2 = @constinferred eigh_trunc(A; alg, trunc)
        @test length(diagview(D2)) == r
        @test isisometry(V2)
        @test A * V2 ≈ V2 * D2

        s = 1 - sqrt(eps(real(T)))
        trunc = truncerror(; atol=s * norm(@view(D₀[r:end]), 1), p=1)
        D3, V3 = @constinferred eigh_trunc(A; alg, trunc)
        @test length(diagview(D3)) == r
        @test A * V3 ≈ V3 * D3

        # test for same subspace
        @test V1 * (V1' * V2) ≈ V2
        @test V2 * (V2' * V1) ≈ V1
        @test V1 * (V1' * V3) ≈ V3
        @test V3 * (V3' * V1) ≈ V1
    end
end

@testset "eigh_trunc! specify truncation algorithm T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 4
    V = qr_compact(randn(rng, T, m, m))[1]
    D = Diagonal([0.9, 0.3, 0.1, 0.01])
    A = V * D * V'
    A = (A + A') / 2
    alg = TruncatedAlgorithm(LAPACK_QRIteration(), truncrank(2))
    D2, V2 = @constinferred eigh_trunc(A; alg)
    @test diagview(D2) ≈ diagview(D)[1:2] rtol = sqrt(eps(real(T)))
    @test_throws ArgumentError eigh_trunc(A; alg, trunc=(; maxrank=2))

    alg = TruncatedAlgorithm(LAPACK_QRIteration(), truncerror(; atol=0.2))
    D3, V3 = @constinferred eigh_trunc(A; alg)
    @test diagview(D3) ≈ diagview(D)[1:2] rtol = sqrt(eps(real(T)))
end

@testset "eigh for Diagonal{$T}" for T in BLASFloats
    rng = StableRNG(123)
    m = 54
    Ad = randn(rng, T, m)
    Ad .+= conj.(Ad)
    A = Diagonal(Ad)

    D, V = @constinferred eigh_full(A)
    @test D isa Diagonal{real(T)} && size(D) == size(A)
    @test V isa Diagonal{T} && size(V) == size(A)
    @test A * V ≈ V * D

    D2 = @constinferred eigh_vals(A)
    @test D2 isa AbstractVector{real(T)} && length(D2) == m
    @test diagview(D) ≈ D2

    A2 = Diagonal(T[0.9, 0.3, 0.1, 0.01])
    alg = TruncatedAlgorithm(DiagonalAlgorithm(), truncrank(2))
    D2, V2 = @constinferred eigh_trunc(A2; alg)
    @test diagview(D2) ≈ diagview(A2)[1:2]
end
