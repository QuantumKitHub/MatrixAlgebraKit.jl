using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: Diagonal

@testset "gen_eig_full! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    for alg in (LAPACK_Simple(), :LAPACK_Simple, LAPACK_Simple)
        A  = randn(rng, T, m, m)
        B  = randn(rng, T, m, m)
        A_init = copy(A) 
        B_init = copy(B)
        Tc = complex(T)

        W, V = @constinferred gen_eig_full(A, B; alg=($alg))
        @test eltype(W) == eltype(V) == Tc
        @test A_init == A
        @test B_init == B
        @test A * V ≈ B * V * Diagonal(W)

        alg′ = @constinferred MatrixAlgebraKit.select_algorithm(gen_eig_full!, (A, B), $alg)

        Ac = similar(A)
        Bc = similar(B)
        W2, V2 = @constinferred gen_eig_full!(copy!(Ac, A), copy!(Bc, B), (W, V), alg′)
        @test W2 === W
        @test V2 === V
        @test A * V ≈ B * V * Diagonal(W)

        Wc = @constinferred gen_eig_vals(A, B, alg′)
        @test eltype(Wc) == Tc
        @test W ≈ Diagonal(Wc)
    end
end
