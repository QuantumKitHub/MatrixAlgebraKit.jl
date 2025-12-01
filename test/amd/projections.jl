using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: LinearAlgebra, Diagonal, norm
using AMDGPU

const BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)

@testset "project_(anti)hermitian! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54
    noisefactor = eps(real(T))^(3 / 4)
    for alg in (NativeBlocked(blocksize = 16), NativeBlocked(blocksize = 32), NativeBlocked(blocksize = 64))
        for A in (ROCArray(randn(rng, T, m, m)), Diagonal(ROCArray(randn(rng, T, m))))
            Ah = (A + A') / 2
            Aa = (A - A') / 2
            Ac = copy(A)

            Bh = project_hermitian(A, alg)
            @test ishermitian(Bh)
            @test Bh ≈ Ah
            @test A == Ac
            # this is still hermitian for real Diagonals!
            Bh_approx = Bh + noisefactor * Aa
            if !isa(A, Diagonal) && !(T <: Real)
                @test !ishermitian(Bh_approx)
            end
            @test ishermitian(Bh_approx; rtol = 10 * noisefactor)

            Ba = project_antihermitian(A, alg)
            @test isantihermitian(Ba)
            @test Ba ≈ Aa
            @test A == Ac
            # this is still hermitian for real Diagonals!
            Ba_approx = Ba + noisefactor * Ah
            @test !isantihermitian(Ba_approx)
            if !isa(A, Diagonal) && !(T <: Real)
                @test isantihermitian(Ba_approx; rtol = 10 * noisefactor)
            end

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
end

@testset "project_isometric! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54
    @testset "size ($m, $n)" for n in (37, m)
        k = min(m, n)
        svdalgs = (ROCSOLVER_QRIteration(), ROCSOLVER_Jacobi())
        algs = (PolarViaSVD.(svdalgs)...,) # PolarNewton()) # TODO
        @testset "algorithm $alg" for alg in algs
            for A in (ROCArray(randn(rng, T, m, n)), Diagonal(ROCArray(randn(rng, T, m))))
                W = project_isometric(A, alg)
                @test isisometric(W)
                W2 = project_isometric(W, alg)
                @test W2 ≈ W # stability of the projection
                @test W * (W' * A) ≈ A

                Ac = similar(A)
                W2 = @constinferred project_isometric!(copy!(Ac, A), W, alg)
                @test W2 === W
                @test isisometric(W)

                # test that W is closer to A then any other isometry
                for k in 1:10
                    δA = ROCArray(randn(rng, T, size(A)...))
                    W = project_isometric(A, alg)
                    W2 = project_isometric(A + δA / 100, alg)
                    @test norm(A - W2) > norm(A - W)
                end
            end
        end
    end
end
