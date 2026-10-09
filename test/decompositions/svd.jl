using MatrixAlgebraKit
using Test
using LinearAlgebra: Diagonal, isposdef
using CUDA, AMDGPU

if @isdefined(fast_tests) && fast_tests
    BLASFloats = (Float64, ComplexF64)
    GenericFloats = (BigFloat, Complex{BigFloat})
else
    BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
    GenericFloats = (BigFloat, Complex{BigFloat})
end

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite
using .TestSuite: testargs_summary

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

# CPU tests
# ---------
if !is_buildkite
    # LAPACK algorithms:
    for T in BLASFloats, m in (0, 54), n in (0, 37, m, 63)
        TestSuite.seed_rng!(123)
        LAPACK_SVD_ALGS = (QRIteration(), DivideAndConquer(), SafeDivideAndConquer(; fixgauge = true), Bisection())
        TestSuite.test_svd(T, (m, n))
        TestSuite.test_svd_algs(T, (m, n), LAPACK_SVD_ALGS)
        @static if VERSION > v"1.11-" # Jacobi broken on 1.10
            TestSuite.test_svd_algs(T, (m, n), (LAPACK_Jacobi(),); test_full = false, test_vals = false)
        end
    end

    # Generic floats:
    for T in GenericFloats, m in (0, 54), n in (0, 37, m, 63)
        TestSuite.seed_rng!(123)
        TestSuite.test_svd(T, (m, n))
        TestSuite.test_svd_algs(T, (m, n), (GLA_QRIteration(),))
    end

    # Diagonal:
    for T in (BLASFloats..., GenericFloats...), m in (0, 54)
        TestSuite.seed_rng!(123)
        AT = Diagonal{T, Vector{T}}
        TestSuite.test_svd(AT, m)
        TestSuite.test_svd_algs(AT, m, (DiagonalAlgorithm(),))
    end
end

batch_size = 16

# CUDA tests
# ------------
if CUDA.functional()
    # CUSOLVER algorithms:
    for T in BLASFloats, m in (0, 23), n in (0, 17, m, 27)
        TestSuite.seed_rng!(123)
        TestSuite.test_svd(CuMatrix{T}, (m, n))
        CUDA_SVD_ALGS = (QRIteration(), SVDViaPolar(), Jacobi())
        TestSuite.test_svd_algs(CuMatrix{T}, (m, n), CUDA_SVD_ALGS)

        TestSuite.test_svd_batched(CuMatrix{T}, (m, n), batch_size)
        CUDA_SVD_ALGS = (Jacobi(),)
        TestSuite.test_svd_batched_algs(CuMatrix{T}, (m, n), batch_size, CUDA_SVD_ALGS)
    end
    for T in BLASFloats
        @test MatrixAlgebraKit.default_svd_algorithm(Vector{CuMatrix{T}}) isa Jacobi
        @test MatrixAlgebraKit.default_svd_algorithm(CuArray{T, 3}) isa Jacobi
    end
    for T in BLASFloats
        TestSuite.seed_rng!(123)
        TestSuite.test_svd_algs_batched_oversized(CuMatrix{T}, (Jacobi(),), batch_size)
        TestSuite.test_svd_algs_batched_ragged_support(CuMatrix{T}, (Jacobi(),), batch_size)
        TestSuite.test_svd_algs_batched_ragged_support(CuMatrix{T}, (QRIteration(),), batch_size; supported = false)
    end

    # Randomized SVD:
    for T in BLASFloats, m in (0, 23), n in (0, 17, m, 27)
        TestSuite.seed_rng!(123)
        k = 5
        p = min(m, n) - k - 2
        p > 0 || continue
        TestSuite.test_randomized_svd(CuMatrix{T}, (m, n), (MatrixAlgebraKit.TruncatedAlgorithm(CUSOLVER_Randomized(; k, p, niters = 20), truncrank(k)),))
    end

    # Diagonal:
    for T in BLASFloats, m in (0, 23)
        TestSuite.seed_rng!(123)
        AT = Diagonal{T, CuVector{T}}
        TestSuite.test_svd(AT, m)
        TestSuite.test_svd_algs(AT, m, (DiagonalAlgorithm(),))
    end
end

# AMDGPU tests
# ------------
if AMDGPU.functional()
    # ROCSOLVER algorithms:
    for T in BLASFloats, m in (0, 1, 23), n in (0, 1, 17, m, 27)
        TestSuite.seed_rng!(123)
        TestSuite.test_svd(ROCMatrix{T}, (m, n))
        AMD_SVD_ALGS = (QRIteration(), Jacobi(), DivideAndConquer(), Bisection())
        TestSuite.test_svd_algs(ROCMatrix{T}, (m, n), AMD_SVD_ALGS)
        TestSuite.test_svd_batched(ROCMatrix{T}, (m, n), batch_size)
        TestSuite.test_svd_batched_algs(ROCMatrix{T}, (m, n), batch_size, AMD_SVD_ALGS)
    end
    for T in BLASFloats
        @test MatrixAlgebraKit.default_svd_algorithm(Vector{ROCMatrix{T}}) ==
            MatrixAlgebraKit.default_svd_algorithm(ROCMatrix{T})
    end
    for T in BLASFloats
        TestSuite.seed_rng!(123)
        AMD_SVD_ALGS = (QRIteration(), Jacobi(), DivideAndConquer(), Bisection())
        TestSuite.test_svd_algs_batched_ragged_support(ROCMatrix{T}, AMD_SVD_ALGS, batch_size)
    end

    @testset "Bisection with min(m, n) == 1 $(testargs_summary(T, sz))" for T in BLASFloats,
            sz in ((1, 1), (2, 1), (5, 1), (1, 2), (1, 5))

        TestSuite.seed_rng!(123)
        for _ in 1:16
            A = TestSuite.instantiate_matrix(ROCMatrix{T}, sz)
            U, S, Vᴴ = svd_compact(A; alg = Bisection())
            @test U * S * Vᴴ ≈ A
            @test isisometric(U)
            @test isisometric(Vᴴ; side = :right)
            @test isposdef(S)
            @test Array(S) ≈ Diagonal(svd_vals(Array(A)))
        end
    end

    # Diagonal:
    for T in BLASFloats, m in (0, 23)
        TestSuite.seed_rng!(123)
        AT = Diagonal{T, ROCVector{T}}
        TestSuite.test_svd(AT, m)
        TestSuite.test_svd_algs(AT, m, (DiagonalAlgorithm(),))
    end
end
