using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: LinearAlgebra, Diagonal, I
using MatrixAlgebraKit: TruncatedAlgorithm, diagview
using AMDGPU

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)

@testset "eigh_full! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54
    for alg in (
            ROCSOLVER_DivideAndConquer(),
            ROCSOLVER_Jacobi(),
            ROCSOLVER_Bisection(),
            ROCSOLVER_QRIteration(),
        )
        A = ROCArray(randn(rng, T, m, m))
        A = (A + A') / 2

        D, V = @constinferred eigh_full(A; alg)
        @test A * V ≈ V * D
        @test isunitary(V)
        @test all(isreal, D)

        D2, V2 = eigh_full!(copy(A), (D, V), alg)
        @test D2 === D
        @test V2 === V

        D3 = @constinferred eigh_vals(A, alg)
        @test parent(D) ≈ D3
    end
end

#=@testset "eigh_trunc! for T = $T" for T in BLASFloats 
    rng = StableRNG(123)
    m = 54
    for alg in (ROCSOLVER_QRIteration(),
                ROCSOLVER_DivideAndConquer(),
                )
        A = ROCArray(randn(rng, T, m, m))
        A = A * A'
        A = (A + A') / 2
        Ac = similar(A)
        D₀ = reverse(eigh_vals(A))
        r = m - 2
        s = 1 + sqrt(eps(real(T)))

        D1, V1, ϵ1 = @constinferred eigh_trunc(A; alg, trunc=truncrank(r))
        @test length(diagview(D1)) == r
        @test isisometric(V1)
        @test A * V1 ≈ V1 * D1
        @test LinearAlgebra.opnorm(A - V1 * D1 * V1') ≈ D₀[r + 1]

        trunc = trunctol(; atol=s * D₀[r + 1])
        D2, V2, ϵ2 = @constinferred eigh_trunc(A; alg, trunc)
        @test length(diagview(D2)) == r
        @test isisometric(V2)
        @test A * V2 ≈ V2 * D2

        # test for same subspace
        @test V1 * (V1' * V2) ≈ V2
        @test V2 * (V2' * V1) ≈ V1
    end
end

@testset "eigh_trunc! specify truncation algorithm T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 4
    V = qr_compact(ROCArray(randn(rng, T, m, m)))[1]
    D = Diagonal([0.9, 0.3, 0.1, 0.01])
    A = V * D * V'
    A = (A + A') / 2
    alg = TruncatedAlgorithm(ROCSOLVER_QRIteration(), truncrank(2))
    D2, V2, ϵ2 = @constinferred eigh_trunc(A; alg)
    @test diagview(D2) ≈ diagview(D)[1:2] rtol = sqrt(eps(real(T)))
    @test_throws ArgumentError eigh_trunc(A; alg, trunc=(; maxrank=2))
end=#

@testset "eigh for Diagonal{$T}" for T in BLASFloats
    rng = StableRNG(123)
    m = 54
    Ad = randn(rng, T, m)
    Ad .+= conj.(Ad)
    A = Diagonal(ROCArray(Ad))
    atol = sqrt(eps(real(T)))

    D, V = @constinferred eigh_full(A)
    @test D isa Diagonal{real(T)} && size(D) == size(A)
    @test V isa Diagonal{T} && size(V) == size(A)
    @test A * V ≈ V * D

    D2 = @constinferred eigh_vals(A)
    @test D2 isa AbstractVector{real(T)} && length(D2) == m
    @test diagview(D) ≈ D2

    # TODO partialsortperm
    #=A2 = Diagonal(ROCArray(T[0.9, 0.3, 0.1, 0.01]))
    alg = TruncatedAlgorithm(DiagonalAlgorithm(), truncrank(2))
    D2, V2, ϵ2 = @constinferred eigh_trunc(A2; alg)
    @test diagview(D2) ≈ diagview(A2)[1:2]
    @test ϵ2 ≈ norm(diagview(A2)[3:4]) atol = atol=#
end
