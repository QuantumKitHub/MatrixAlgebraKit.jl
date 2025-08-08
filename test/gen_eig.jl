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

        W, V = @constinferred gen_eig(A, B; alg=($alg))
        @test eltype(W) == eltype(V) == Tc
        @test A_init == A
        @test B_init == B
        @test A * V ≈ B * V * Diagonal(W)

        alg′ = @constinferred MatrixAlgebraKit.select_algorithm(gen_eig_full!, (A, B), $alg)

        Ac = similar(A)
        Bc = similar(B)
        W2, V2 = @constinferred gen_eig!(copy!(Ac, A), copy!(Bc, B), (W, V), alg′)
        @test W2 === W
        @test V2 === V
        @test A * V ≈ B * V * Diagonal(W)

        Ac = similar(A)
        Bc = similar(B)
        W2, V2 = @constinferred gen_eig_full!(copy!(Ac, A), copy!(Bc, B), (W, V))
        @test W2 === W
        @test V2 === V
        @test A * V ≈ B * V * Diagonal(W)

        Wc = @constinferred gen_eig_vals(A, B, alg′)
        @test eltype(Wc) == Tc
        @test W ≈ Diagonal(Wc)

    end
    A  = randn(rng, T, m, m)
    B  = randn(rng, T, m, m)
    @test_throws ArgumentError("LAPACK_Expert is not supported for ggev") gen_eig_full(A, B; alg=LAPACK_Expert())
    @test_throws ArgumentError("LAPACK_Simple (ggev) does not accept any keyword arguments") gen_eig_full(A, B; alg=LAPACK_Simple(bad="sad"))
    @test_throws ArgumentError("LAPACK_Expert is not supported for ggev") gen_eig_vals(A, B; alg=LAPACK_Expert())
    @test_throws ArgumentError("LAPACK_Simple (ggev) does not accept any keyword arguments") gen_eig_vals(A, B; alg=LAPACK_Simple(bad="sad"))

    # a tuple of the input types is passed to `default_algorithm`
    @test_throws MethodError MatrixAlgebraKit.default_algorithm(gen_eig_full, A, B)
    @test_throws MethodError MatrixAlgebraKit.default_algorithm(gen_eig_vals, A, B)
    if T <: Real
        # Float16 isn't supported
        Afp16 = Float16.(A)
        Bfp16 = Float16.(B)
        @test_throws MethodError MatrixAlgebraKit.default_algorithm(gen_eig_full, (Afp16, Bfp16))
        @test_throws MethodError MatrixAlgebraKit.default_gen_eig_algorithm(Afp16, Bfp16)
    end
end
    
