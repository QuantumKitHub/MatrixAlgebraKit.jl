using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: Diagonal
using MatrixAlgebraKit: TruncatedAlgorithm, diagview, norm
using GenericSchur

const eltypes = (BigFloat, Complex{BigFloat})

@testset "eig_full! for T = $T" for T in eltypes
    rng = StableRNG(123)
    m = 24
    alg = GS_eig_Francis()
    A = randn(rng, T, m, m)
    Tc = complex(T)

    D, V = @constinferred eig_full(A; alg = ($alg))
    @test eltype(D) == eltype(V) == Tc
    @test A * V ≈ V * D

    alg′ = @constinferred MatrixAlgebraKit.select_algorithm(eig_full!, A, $alg)

    Ac = similar(A)
    D2, V2 = @constinferred eig_full!(copy!(Ac, A), (D, V), alg′)
    @test D2 === D
    @test V2 === V
    @test A * V ≈ V * D

    Dc = @constinferred eig_vals(A, alg′)
    @test eltype(Dc) == Tc
    @test D ≈ Diagonal(Dc)
end

@testset "eig_trunc! for T = $T" for T in eltypes
    rng = StableRNG(123)
    m = 6
    alg = GS_eig_Francis()
    A = randn(rng, T, m, m)
    A *= A' # TODO: deal with eigenvalue ordering etc
    # eigenvalues are sorted by ascending real component...
    D₀ = sort!(eig_vals(A); by = abs, rev = true)
    rmin = findfirst(i -> abs(D₀[end - i]) != abs(D₀[end - i - 1]), 1:(m - 2))
    r = length(D₀) - rmin
    atol = sqrt(eps(real(T)))

    D1, V1, ϵ1 = @constinferred eig_trunc(A; alg, trunc = truncrank(r))
    D1base, V1base = @constinferred eig_full(A; alg)

    @test length(diagview(D1)) == r
    @test A * V1 ≈ V1 * D1
    @test ϵ1 ≈ norm(view(D₀, (r + 1):m)) atol = atol

    s = 1 + sqrt(eps(real(T)))
    trunc = trunctol(; atol = s * abs(D₀[r + 1]))
    D2, V2, ϵ2 = @constinferred eig_trunc(A; alg, trunc)
    @test length(diagview(D2)) == r
    @test A * V2 ≈ V2 * D2
    @test ϵ2 ≈ norm(view(D₀, (r + 1):m)) atol = atol

    s = 1 - sqrt(eps(real(T)))
    trunc = truncerror(; atol = s * norm(@view(D₀[r:end]), 1), p = 1)
    D3, V3, ϵ3 = @constinferred eig_trunc(A; alg, trunc)
    @test length(diagview(D3)) == r
    @test A * V3 ≈ V3 * D3
    @test ϵ3 ≈ norm(view(D₀, (r + 1):m)) atol = atol

    # trunctol keeps order, truncrank might not
    # test for same subspace
    @test V1 * ((V1' * V1) \ (V1' * V2)) ≈ V2
    @test V2 * ((V2' * V2) \ (V2' * V1)) ≈ V1
    @test V1 * ((V1' * V1) \ (V1' * V3)) ≈ V3
    @test V3 * ((V3' * V3) \ (V3' * V1)) ≈ V1
end

@testset "eig_trunc! specify truncation algorithm T = $T" for T in eltypes
    rng = StableRNG(123)
    m = 4
    atol = sqrt(eps(real(T)))
    V = randn(rng, T, m, m)
    D = Diagonal(real(T)[0.9, 0.3, 0.1, 0.01])
    A = V * D * inv(V)
    alg = TruncatedAlgorithm(GS_eig_Francis(), truncrank(2))
    D2, V2, ϵ2 = @constinferred eig_trunc(A; alg)
    @test diagview(D2) ≈ diagview(D)[1:2]
    @test ϵ2 ≈ norm(diagview(D)[3:4]) atol = atol
    @test_throws ArgumentError eig_trunc(A; alg, trunc = (; maxrank = 2))

    alg = TruncatedAlgorithm(GS_eig_Francis(), truncerror(; atol = 0.2, p = 1))
    D3, V3, ϵ3 = @constinferred eig_trunc(A; alg)
    @test diagview(D3) ≈ diagview(D)[1:2]
    @test ϵ3 ≈ norm(diagview(D)[3:4]) atol = atol
end

@testset "eig for Diagonal{$T}" for T in eltypes
    rng = StableRNG(123)
    m = 24
    Ad = randn(rng, T, m)
    A = Diagonal(Ad)
    atol = sqrt(eps(real(T)))

    D, V = @constinferred eig_full(A)
    @test D isa Diagonal{T} && size(D) == size(A)
    @test V isa Diagonal{T} && size(V) == size(A)
    @test A * V ≈ V * D

    D2 = @constinferred eig_vals(A)
    @test D2 isa AbstractVector{T} && length(D2) == m
    @test diagview(D) ≈ D2

    A2 = Diagonal(T[0.9, 0.3, 0.1, 0.01])
    alg = TruncatedAlgorithm(DiagonalAlgorithm(), truncrank(2))
    D2, V2, ϵ2 = @constinferred eig_trunc(A2; alg)
    @test diagview(D2) ≈ diagview(A2)[1:2]
    @test ϵ2 ≈ norm(diagview(A2)[3:4]) atol = atol
end
