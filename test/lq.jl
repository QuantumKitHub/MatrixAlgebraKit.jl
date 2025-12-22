using MatrixAlgebraKit
using Test
using StableRNGs
using LinearAlgebra: diag, I, Diagonal
using MatrixAlgebraKit: LQViaTransposedQR, LAPACK_HouseholderLQ
using CUDA, AMDGPU, GenericLinearAlgebra

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
GenericFloats = (Float16, BigFloat, Complex{BigFloat})

@isdefined(TestSuite) || include("testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

m = 54
for T in (BLASFloats..., GenericFloats...), n in (37, m, 63)
    TestSuite.seed_rng!(123)
    if T ∈ BLASFloats
        if CUDA.functional()
            CUDA_LQ_ALGS = LQViaTransposedQR.((CUSOLVER_HouseholderQR(; positive = false), CUSOLVER_HouseholderQR(; positive = true)))
            TestSuite.test_lq(CuMatrix{T}, (m, n); test_pivoted = false, test_blocksize = false)
            TestSuite.test_lq_algs(CuMatrix{T}, (m, n), CUDA_LQ_ALGS)
            if n == m
                TestSuite.test_lq(Diagonal{T, CuVector{T}}, m; test_pivoted = false, test_blocksize = false)
                TestSuite.test_lq_algs(Diagonal{T, CuVector{T}}, m, (DiagonalAlgorithm(),))
            end
        end
        if AMDGPU.functional()
            ROC_LQ_ALGS = LQViaTransposedQR.((ROCSOLVER_HouseholderQR(; positive = false), ROCSOLVER_HouseholderQR(; positive = true)))
            TestSuite.test_lq(ROCMatrix{T}, (m, n); test_pivoted = false, test_blocksize = false)
            TestSuite.test_lq_algs(ROCMatrix{T}, (m, n), ROC_LQ_ALGS)
            if n == m
                TestSuite.test_lq(Diagonal{T, ROCVector{T}}, m; test_pivoted = false, test_blocksize = false)
                TestSuite.test_lq_algs(Diagonal{T, ROCVector{T}}, m, (DiagonalAlgorithm(),))
            end
        end
    end
    if !is_buildkite
        if T ∈ BLASFloats
            TestSuite.test_lq(T, (m, n))
            LAPACK_LQ_ALGS = (
                LAPACK_HouseholderLQ(; positive = false, pivoted = false, blocksize = 1),
                LAPACK_HouseholderLQ(; positive = false, pivoted = false, blocksize = 8),
                LAPACK_HouseholderLQ(; positive = false, pivoted = true, blocksize = 1),
                #LAPACK_HouseholderLQ(; positive=false, pivoted=true, blocksize=8), # not supported
                LAPACK_HouseholderLQ(; positive = true, pivoted = false, blocksize = 1),
                LAPACK_HouseholderLQ(; positive = true, pivoted = false, blocksize = 8),
                LAPACK_HouseholderLQ(; positive = true, pivoted = true, blocksize = 1),
                #LAPACK_HouseholderLQ(; positive=true, pivoted=true, blocksize=8), # not supported
            )
            TestSuite.test_lq_algs(T, (m, n), LAPACK_LQ_ALGS)
        elseif T ∈ GenericFloats
            TestSuite.test_lq(T, (m, n); test_null = false, test_pivoted = false, test_blocksize = false)
            GLA_LQ_ALGS = (LQViaTransposedQR(GLA_HouseholderQR()),)
            TestSuite.test_lq_algs(T, (m, n), GLA_LQ_ALGS; test_null = false)
        end
        if m == n
            AT = Diagonal{T, Vector{T}}
            TestSuite.test_lq(AT, m; test_pivoted = false, test_blocksize = false)
            TestSuite.test_lq_algs(AT, m, (DiagonalAlgorithm(),))
        end
    end
end
