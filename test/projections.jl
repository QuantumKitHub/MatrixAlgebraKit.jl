using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: Diagonal

const BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)

@testset "project_(anti)hermitian! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54
    noisefactor = eps(real(T))^(3 / 4)
    for alg in (NativeBlocked(blocksize = 16), NativeBlocked(blocksize = 32), NativeBlocked(blocksize = 64))
        A = randn(rng, T, m, m)
        Ah = (A + A') / 2
        Aa = (A - A') / 2
        Ac = copy(A)

        Bh = project_hermitian(A, alg)
        @test ishermitian(Bh)
        @test Bh ≈ Ah
        @test A == Ac
        Bh_approx = Bh + noisefactor * Aa
        @test !ishermitian(Bh_approx)
        @test ishermitian(Bh_approx; rtol = 10 * noisefactor)

        Ba = project_antihermitian(A, alg)
        @test isantihermitian(Ba)
        @test Ba ≈ Aa
        @test A == Ac
        Ba_approx = Ba + noisefactor * Ah
        @test !isantihermitian(Ba_approx)
        @test isantihermitian(Ba_approx; rtol = 10 * noisefactor)

        Bh = project_hermitian!(Ac, alg)
        @test Bh === Ac
        @test ishermitian(Bh)
        @test Bh ≈ Ah

        copy!(Ac, A)
        Ba = project_antihermitian!(Ac, alg)
        @test Ba === Ac
        @test isantihermitian(Ba)
        @test Ba ≈ Aa
    end
end
