using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using MatrixAlgebraKit: diagview
using LinearAlgebra
using LinearAlgebra: exp
using CUDA, AMDGPU

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
GenericFloats = (Float16, ComplexF16, BigFloat, Complex{BigFloat})

@testset "exponential! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54

    A = LinearAlgebra.normalize!(randn(rng, T, m, m))
    Ac = copy(A)
    expA = LinearAlgebra.exp(A)

    expA2 = @constinferred exponential(A)
    @test expA ≈ expA2
    @test A == Ac

    algs = (MatrixFunctionViaLA(), MatrixFunctionViaEig(LAPACK_Simple()), MatrixFunctionViaTaylor())
    @testset "algorithm $alg" for alg in algs
        expA2 = @constinferred exponential(A, alg)
        @test expA ≈ expA2
        @test A == Ac
    end

    @test_throws DomainError exponential(A; alg = MatrixFunctionViaEigh(LAPACK_QRIteration()))
end

@testset "exponential! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 54

    A = randn(rng, T, m, m)
    τ = randn(rng, T)
    Ac = copy(A)

    Aτ = A * τ
    expAτ = LinearAlgebra.exp(Aτ)

    expAτ2 = @constinferred exponential((τ, A))
    @test expAτ ≈ expAτ2
    @test A == Ac

    algs = (MatrixFunctionViaLA(), MatrixFunctionViaEig(LAPACK_Simple()), MatrixFunctionViaTaylor())
    @testset "algorithm $alg" for alg in algs
        expAτ2 = @constinferred exponential((τ, A), alg)
        @test expAτ ≈ expAτ2
        @test A == Ac
    end

    @test_throws DomainError exponential((τ, A); alg = MatrixFunctionViaEigh(LAPACK_QRIteration()))
end

@testset "exponential! for non-Matrix input $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 12
    A = LinearAlgebra.normalize!(randn(rng, T, m, m))
    expA = LinearAlgebra.exp(A)

    wrappers = (
        ("view", B -> view(B, :, :)),
        ("PermutedDimsArray", B -> PermutedDimsArray(permutedims(B), (2, 1))),
        ("ReshapedArray", B -> reshape(view(vec(B), 1:(m * m)), m, m)),
    )
    @testset "$name" for (name, wrap) in wrappers
        W = wrap(copy(A))
        @test !(W isa Matrix)
        @test exponential!(W) ≈ expA
    end
end

@testset "exponential! for Diagonal{$T}" for T in (BLASFloats..., GenericFloats...)
    rng = StableRNG(123)
    m = 54

    A = Diagonal(randn(rng, T, m))
    τ = randn(rng, T)
    Ac = copy(A)

    expA = LinearAlgebra.exp(A)

    expA2 = @constinferred exponential(A)
    @test expA ≈ expA2
    @test A == Ac
end

@testset "exponential! for Diagonal{$T}" for T in (BLASFloats..., GenericFloats...)
    rng = StableRNG(123)
    m = 1

    A = Diagonal(randn(rng, T, m))
    τ = randn(rng, T)
    Ac = copy(A)

    Aτ = A * τ
    expAτ = LinearAlgebra.exp(Aτ)

    expAτ2 = @constinferred exponential((τ, A))
    @test expAτ ≈ expAτ2
    @test A == Ac
end

# GPU tests
# ---------
# The Taylor exponential is backend-generic, so the same code runs on GPU. Compare device
# results against the CPU reference, exercising both `balance` settings (fix 1 & 3) and the
# scaled `(τ, A)` entrypoint. A badly-scaled matrix exercises the balancing path. If any step
# fell back to scalar indexing these would error under GPUArrays' scalar-indexing guard.
function test_exponential_gpu(ArrayT, T)
    rng = StableRNG(123)
    m = 54
    A = randn(rng, T, m, m) ./ (2 * m)
    τ = randn(rng, T)

    # badly-scaled similarity transform Aᵢⱼ ← Aᵢⱼ sᵢ / sⱼ, to give balancing work to do
    s = exp10.(range(-real(T)(3), real(T)(3), length = m))
    Abad = A .* s ./ transpose(s)

    for M in (A, Abad)
        M_gpu = ArrayT(M)
        for alg in (MatrixFunctionViaTaylor(), MatrixFunctionViaTaylor(; balance = false))
            @test Array(exponential(M_gpu, alg)) ≈ exponential(M, alg)
            @test Array(exponential((τ, M_gpu), alg)) ≈ exponential((τ, M), alg)
        end
    end
    return nothing
end

if CUDA.functional()
    @testset "exponential on CUDA for T = $T" for T in BLASFloats
        test_exponential_gpu(CuArray, T)
    end
end

if AMDGPU.functional()
    @testset "exponential on AMDGPU for T = $T" for T in BLASFloats
        test_exponential_gpu(ROCArray, T)
    end
end
