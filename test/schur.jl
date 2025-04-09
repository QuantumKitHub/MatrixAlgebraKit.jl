using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: I

@testset "schur_full! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    for alg in (LAPACK_Simple(), LAPACK_Expert())
        A = randn(rng, T, m, m)
        Tc = complex(T)

        TA, Z, vals = @constinferred schur_full(A; alg)
        @test eltype(TA) == eltype(Z) == T
        @test eltype(vals) == Tc
        @test Z' * Z ≈ I
        @test A * Z ≈ Z * TA

        Ac = similar(A)
        TA2, Z2, vals2 = @constinferred schur_full!(copy!(Ac, A), (TA, Z, vals), alg)
        @test TA2 === TA
        @test Z2 === Z
        @test vals2 === vals
        @test A * Z ≈ Z * TA

        valsc = @constinferred schur_vals(A, alg)
        @test eltype(valsc) == Tc
        @test valsc ≈ eig_vals(A, alg)
    end
end
