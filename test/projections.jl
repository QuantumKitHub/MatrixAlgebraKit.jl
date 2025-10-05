using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: Diagonal

const BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)

@testset "(anti)hermitianpart! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54
    for alg in (NativeBlocked(blocksize = 16), NativeBlocked(blocksize = 32), NativeBlocked(blocksize = 64))
        A = randn(rng, T, m, m)
        Ah = (A + A') / 2
        Aa = (A - A') / 2
        Ac = copy(A)

        Bh = hermitianpart(A, alg)
        @test ishermitian(Bh)
        @test Bh ≈ Ah
        @test A == Ac

        Ba = antihermitianpart(A, alg)
        @test isantihermitian(Ba)
        @test Ba ≈ Aa
        @test A == Ac

        Bh = hermitianpart!(Ac, alg)
        @test Bh === Ac
        @test ishermitian(Bh)
        @test Bh ≈ Ah

        copy!(Ac, A)
        Ba = antihermitianpart!(Ac, alg)
        @test Ba === Ac
        @test isantihermitian(Ba)
        @test Ba ≈ Aa
    end
end
