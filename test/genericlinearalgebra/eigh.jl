using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: LinearAlgebra, Diagonal, I
using MatrixAlgebraKit: TruncatedAlgorithm, diagview, norm
using GenericLinearAlgebra

const eltypes = (BigFloat, Complex{BigFloat})

@testset "eigh_full! for T = $T" for T in eltypes
    rng = StableRNG(123)
    m = 54
    alg = GLA_QRIteration()

    A = randn(rng, T, m, m)
    A = (A + A') / 2

    D, V = @constinferred eigh_full(A; alg)
    @test A * V ≈ V * D
    @test isunitary(V)
    @test all(isreal, D)

    D2, V2 = eigh_full!(copy(A), (D, V), alg)
    @test D2 ≈ D
    @test V2 ≈ V

    D3 = @constinferred eigh_vals(A, alg)
    @test D ≈ Diagonal(D3)
end

@testset "eigh_trunc! for T = $T" for T in eltypes
    rng = StableRNG(123)
    m = 54
    alg = GLA_QRIteration()
    A = randn(rng, T, m, m)
    A = A * A'
    A = (A + A') / 2
    Ac = similar(A)
    D₀ = reverse(eigh_vals(A))

    r = m - 2
    s = 1 + sqrt(eps(real(T)))
    atol = sqrt(eps(real(T)))

    D1, V1, ϵ1 = @constinferred eigh_trunc(A; alg, trunc = truncrank(r))
    Dfull, Vfull = eigh_full(A; alg)
    @test length(diagview(D1)) == r
    @test isisometric(V1)
    @test A * V1 ≈ V1 * D1
    @test LinearAlgebra.opnorm(A - V1 * D1 * V1') ≈ D₀[r + 1]
    @test ϵ1 ≈ norm(view(D₀, (r + 1):m)) atol = atol

    trunc = trunctol(; atol = s * D₀[r + 1])
    D2, V2, ϵ2 = @constinferred eigh_trunc(A; alg, trunc)
    @test length(diagview(D2)) == r
    @test isisometric(V2)
    @test A * V2 ≈ V2 * D2
    @test ϵ2 ≈ norm(view(D₀, (r + 1):m)) atol = atol

    s = 1 - sqrt(eps(real(T)))
    trunc = truncerror(; atol = s * norm(@view(D₀[r:end]), 1), p = 1)
    D3, V3, ϵ3 = @constinferred eigh_trunc(A; alg, trunc)
    @test length(diagview(D3)) == r
    @test A * V3 ≈ V3 * D3
    @test ϵ3 ≈ norm(view(D₀, (r + 1):m)) atol = atol

    # test for same subspace
    @test V1 * (V1' * V2) ≈ V2
    @test V2 * (V2' * V1) ≈ V1
    @test V1 * (V1' * V3) ≈ V3
    @test V3 * (V3' * V1) ≈ V1
end

@testset "eigh_trunc! specify truncation algorithm T = $T" for T in eltypes
    rng = StableRNG(123)
    m = 4
    atol = sqrt(eps(real(T)))
    V = qr_compact(randn(rng, T, m, m))[1]
    D = Diagonal(real(T)[0.9, 0.3, 0.1, 0.01])
    A = V * D * V'
    A = (A + A') / 2
    alg = TruncatedAlgorithm(GLA_QRIteration(), truncrank(2))
    D2, V2, ϵ2 = @constinferred eigh_trunc(A; alg)
    @test diagview(D2) ≈ diagview(D)[1:2]
    @test_throws ArgumentError eigh_trunc(A; alg, trunc = (; maxrank = 2))
    @test ϵ2 ≈ norm(diagview(D)[3:4]) atol = atol

    alg = TruncatedAlgorithm(GLA_QRIteration(), truncerror(; atol = 0.2))
    D3, V3, ϵ3 = @constinferred eigh_trunc(A; alg)
    @test diagview(D3) ≈ diagview(D)[1:2]
    @test ϵ3 ≈ norm(diagview(D)[3:4]) atol = atol
end

@testset "eigh for Diagonal{$T}" for T in eltypes
    rng = StableRNG(123)
    m = 54
    Ad = randn(rng, T, m)
    Ad .+= conj.(Ad)
    A = Diagonal(Ad)
    atol = sqrt(eps(real(T)))

    D, V = @constinferred eigh_full(A)
    @test D isa Diagonal{real(T)} && size(D) == size(A)
    @test V isa Diagonal{T} && size(V) == size(A)
    @test A * V ≈ V * D

    D2 = @constinferred eigh_vals(A)
    @test D2 isa AbstractVector{real(T)} && length(D2) == m
    @test diagview(D) ≈ D2

    A2 = Diagonal(T[0.9, 0.3, 0.1, 0.01])
    alg = TruncatedAlgorithm(DiagonalAlgorithm(), truncrank(2))
    D2, V2, ϵ2 = @constinferred eigh_trunc(A2; alg)
    @test diagview(D2) ≈ diagview(A2)[1:2]
    @test ϵ2 ≈ norm(diagview(A2)[3:4]) atol = atol

end
