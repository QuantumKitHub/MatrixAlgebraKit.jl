using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: Diagonal
using MatrixAlgebraKit: diagview

@testset "eig_full! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    for alg in (LAPACK_Simple(), LAPACK_Expert())
        A = randn(rng, T, m, m)
        Tc = complex(T)

        D, V = @constinferred eig_full(A; alg)
        @test eltype(D) == eltype(V) == Tc
        @test A * V ≈ V * D

        Ac = similar(A)
        D2, V2 = @constinferred eig_full!(copy!(Ac, A), (D, V), alg)
        @test D2 === D
        @test V2 === V
        @test A * V ≈ V * D

        Dc = @constinferred eig_vals(A, alg)
        @test eltype(Dc) == Tc
        @test D ≈ Diagonal(Dc)
    end
end

@testset "eig_trunc! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    for alg in (LAPACK_Simple(), LAPACK_Expert())
        A = randn(rng, T, m, m)
        A *= A' # TODO: deal with eigenvalue ordering etc
        # eigenvalues are sorted by ascending real component...
        D₀ = sort!(eig_vals(A); by=abs, rev=true)
        rmin = findfirst(i -> abs(D₀[end - i]) != abs(D₀[end - i - 1]), 1:(m - 2))
        r = length(D₀) - rmin

        D1, V1 = @constinferred eig_trunc(A; alg, trunc=truncrank(r))
        @test length(D1.diag) == r
        @test A * V1 ≈ V1 * D1

        s = 1 + sqrt(eps(real(T)))
        trunc = trunctol(s * abs(D₀[r + 1]))
        D2, V2 = @constinferred eig_trunc(A; alg, trunc)
        @test length(diagview(D2)) == r
        @test A * V2 ≈ V2 * D2

        # trunctol keeps order, truncrank might not
        # test for same subspace
        @test V1 * ((V1' * V1) \ (V1' * V2)) ≈ V2
        @test V2 * ((V2' * V2) \ (V2' * V1)) ≈ V1
    end
end
