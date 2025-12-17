using MatrixAlgebraKit
using MatrixAlgebraKit: check_hermitian, default_hermitian_tol
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: LinearAlgebra, Diagonal, norm, normalize!

const BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)

@testset "project_(anti)hermitian! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54
    noisefactor = eps(real(T))^(3 / 4)

    mat0 = zeros(T, (1, 1))
    @test ishermitian(mat0)
    @test ishermitian(mat0; atol = default_hermitian_tol(mat0))
    @test isnothing(check_hermitian(mat0))

    for alg in (NativeBlocked(blocksize = 16), NativeBlocked(blocksize = 32), NativeBlocked(blocksize = 64))
        for A in (randn(rng, T, m, m), Diagonal(randn(rng, T, m)))
            Ah = (A + A') / 2
            Aa = (A - A') / 2
            Ac = copy(A)

            Bh = project_hermitian(A, alg)
            @test ishermitian(Bh)
            @test Bh ≈ Ah
            @test A == Ac
            Bh_approx = Bh + noisefactor * Aa
            # this is still hermitian for real Diagonal: |A - A'| == 0
            @test !ishermitian(Bh_approx) || norm(Aa) == 0
            @test ishermitian(Bh_approx; rtol = 10 * noisefactor)

            Ba = project_antihermitian(A, alg)
            @test isantihermitian(Ba)
            @test Ba ≈ Aa
            @test A == Ac
            Ba_approx = Ba + noisefactor * Ah
            @test !isantihermitian(Ba_approx)
            # this is never anti-hermitian for real Diagonal: |A - A'| == 0
            @test isantihermitian(Ba_approx; rtol = 10 * noisefactor) || norm(Aa) == 0

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

    # test approximate error calculation
    A = normalize!(randn(rng, T, m, m))
    Ah = project_hermitian(A)
    Aa = project_antihermitian(A)

    Ah_approx = Ah + noisefactor * Aa
    ϵ = norm(project_antihermitian(Ah_approx))
    @test !ishermitian(Ah_approx; atol = (999 // 1000) * ϵ)
    @test ishermitian(Ah_approx; atol = (1001 // 1000) * ϵ)

    Aa_approx = Aa + noisefactor * Ah
    ϵ = norm(project_hermitian(Aa_approx))
    @test !isantihermitian(Aa_approx; atol = (999 // 1000) * ϵ)
    @test isantihermitian(Aa_approx; atol = (1001 // 1000) * ϵ)
end

@testset "project_isometric! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54
    @testset "size ($m, $n)" for n in (37, m)
        k = min(m, n)
        if LinearAlgebra.LAPACK.version() < v"3.12.0"
            svdalgs = (LAPACK_DivideAndConquer(), LAPACK_QRIteration(), LAPACK_Bisection())
        else
            svdalgs = (LAPACK_DivideAndConquer(), LAPACK_QRIteration(), LAPACK_Bisection(), LAPACK_Jacobi())
        end
        algs = (PolarViaSVD.(svdalgs)..., PolarNewton())
        @testset "algorithm $alg" for alg in algs
            A = randn(rng, T, m, n)
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
                δA = randn(rng, T, m, n)
                W = project_isometric(A, alg)
                W2 = project_isometric(A + δA / 100, alg)
                @test norm(A - W2) > norm(A - W)
            end
        end
    end
end
