using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: Diagonal
using MatrixAlgebraKit: TruncatedAlgorithm, diagview, norm

const BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)

@testset "eig_full! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54
    for alg in (LAPACK_Simple(), LAPACK_Expert(), :LAPACK_Simple, LAPACK_Simple)
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
end

@testset "eig_trunc! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54
    for alg in (LAPACK_Simple(), LAPACK_Expert())
        A = randn(rng, T, m, m)
        A *= A' # TODO: deal with eigenvalue ordering etc
        # eigenvalues are sorted by ascending real component...
        D₀ = sort!(eig_vals(A); by = abs, rev = true)
        rmin = findfirst(i -> abs(D₀[end - i]) != abs(D₀[end - i - 1]), 1:(m - 2))
        r = length(D₀) - rmin

        D1, V1 = @constinferred eig_trunc(A; alg, trunc = truncrank(r))
        @test length(D1.diag) == r
        @test A * V1 ≈ V1 * D1

        s = 1 + sqrt(eps(real(T)))
        trunc = trunctol(; atol = s * abs(D₀[r + 1]))
        D2, V2 = @constinferred eig_trunc(A; alg, trunc)
        @test length(diagview(D2)) == r
        @test A * V2 ≈ V2 * D2

        s = 1 - sqrt(eps(real(T)))
        trunc = truncerror(; atol = s * norm(@view(D₀[r:end]), 1), p = 1)
        D3, V3 = @constinferred eig_trunc(A; alg, trunc)
        @test length(diagview(D3)) == r
        @test A * V3 ≈ V3 * D3

        # trunctol keeps order, truncrank might not
        # test for same subspace
        @test V1 * ((V1' * V1) \ (V1' * V2)) ≈ V2
        @test V2 * ((V2' * V2) \ (V2' * V1)) ≈ V1
        @test V1 * ((V1' * V1) \ (V1' * V3)) ≈ V3
        @test V3 * ((V3' * V3) \ (V3' * V1)) ≈ V1
    end
end

@testset "eig_trunc! specify truncation algorithm T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 4
    V = randn(rng, T, m, m)
    D = Diagonal([0.9, 0.3, 0.1, 0.01])
    A = V * D * inv(V)
    alg = TruncatedAlgorithm(LAPACK_Simple(), truncrank(2))
    D2, V2 = @constinferred eig_trunc(A; alg)
    @test diagview(D2) ≈ diagview(D)[1:2] rtol = sqrt(eps(real(T)))
    @test_throws ArgumentError eig_trunc(A; alg, trunc = (; maxrank = 2))

    alg = TruncatedAlgorithm(LAPACK_Simple(), truncerror(; atol = 0.2, p = 1))
    D3, V3 = @constinferred eig_trunc(A; alg)
    @test diagview(D3) ≈ diagview(D)[1:2] rtol = sqrt(eps(real(T)))
end

@testset "eig for Diagonal{$T}" for T in BLASFloats
    rng = StableRNG(123)
    m = 54
    Ad = randn(rng, T, m)
    A = Diagonal(Ad)

    D, V = @constinferred eig_full(A)
    @test D isa Diagonal{T} && size(D) == size(A)
    @test V isa Diagonal{T} && size(V) == size(A)
    @test A * V ≈ V * D

    D2 = @constinferred eig_vals(A)
    @test D2 isa AbstractVector{T} && length(D2) == m
    @test diagview(D) ≈ D2

    A2 = Diagonal(T[0.9, 0.3, 0.1, 0.01])
    alg = TruncatedAlgorithm(DiagonalAlgorithm(), truncrank(2))
    D2, V2 = @constinferred eig_trunc(A2; alg)
    @test diagview(D2) ≈ diagview(A2)[1:2]
end
