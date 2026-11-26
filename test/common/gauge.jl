using MatrixAlgebraKit
using MatrixAlgebraKit: gauge_index
using Test
using StableRNGs

@testset "gauge_index tie-breaking ($T)" for T in (Float64, ComplexF64)
    rng = StableRNG(123)
    # the first two entries of every column have (nearly) equal, largest magnitude
    V = T[1 -2 3 -4; -1 2 -3 4; 0 0 0 0; 0 0 0 0] .+ randn(rng, T, 4, 4) .* [0, 0, 1e-1, 1e-1]
    V[2, :] .*= 1 + 1.0e-14
    I = gauge_index(V; dims = 1)
    @test vec(I) == [CartesianIndex(1, j) for j in 1:4]
    # rounding-level perturbations of the magnitudes do not change the selected entries
    V′ = copy(V)
    V′[1, :] .*= 1 - 4eps()
    @test gauge_index(V′; dims = 1) == I
    @test vec(gauge_index(permutedims(V); dims = 2)) == [CartesianIndex(j, 1) for j in 1:4]
end

@testset "svd gauge stable for exactly tied magnitudes ($T)" for T in (Float64, ComplexF64)
    rng = StableRNG(1234)
    m, n = 10, 4
    B = randn(rng, T, m ÷ 2, n)
    # rows of A come in pairs related by a sign flip, so every left singular vector has pairs
    # of entries with exactly equal magnitude (in exact arithmetic)
    A = vcat(B, -B)
    U, S, Vᴴ = svd_compact(A)
    for A′ in (A * (1 + eps()), A * (1 - eps()), A .* (1 .+ eps() .* rand(rng, m, n)))
        U′, S′, Vᴴ′ = svd_compact(A′)
        @test U′ ≈ U
        @test Vᴴ′ ≈ Vᴴ
    end
end
