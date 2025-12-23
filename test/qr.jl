using MatrixAlgebraKit
using Test
using StableRNGs
using LinearAlgebra: diag, I, Diagonal
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
            CUDA_QR_ALGS = (CUSOLVER_HouseholderQR(; positive = false), CUSOLVER_HouseholderQR(; positive = true))
            TestSuite.test_qr(CuMatrix{T}, (m, n); test_pivoted = false, test_blocksize = false)
            TestSuite.test_qr_algs(CuMatrix{T}, (m, n), CUDA_QR_ALGS)
            if n == m
                TestSuite.test_qr(Diagonal{T, CuVector{T}}, m; test_pivoted = false, test_blocksize = false)
                TestSuite.test_qr_algs(Diagonal{T, CuVector{T}}, m, (DiagonalAlgorithm(),))
            end
        end
        if AMDGPU.functional()
            ROC_QR_ALGS = (ROCSOLVER_HouseholderQR(; positive = false), ROCSOLVER_HouseholderQR(; positive = true))
            TestSuite.test_qr(ROCMatrix{T}, (m, n); test_pivoted = false, test_blocksize = false)
            TestSuite.test_qr_algs(ROCMatrix{T}, (m, n), ROC_QR_ALGS)
            if n == m
                TestSuite.test_qr(Diagonal{T, ROCVector{T}}, m; test_pivoted = false, test_blocksize = false)
                TestSuite.test_qr_algs(Diagonal{T, ROCVector{T}}, m, (DiagonalAlgorithm(),))
            end
        end
    end
    if !is_buildkite
        if T ∈ BLASFloats
            TestSuite.test_qr(T, (m, n))
            LAPACK_QR_ALGS = (
                LAPACK_HouseholderQR(; positive = false, pivoted = false, blocksize = 1),
                LAPACK_HouseholderQR(; positive = false, pivoted = false, blocksize = 8),
                LAPACK_HouseholderQR(; positive = false, pivoted = true, blocksize = 1),
                #LAPACK_HouseholderQR(; positive=false, pivoted=true, blocksize=8), # not supported
                LAPACK_HouseholderQR(; positive = true, pivoted = false, blocksize = 1),
                LAPACK_HouseholderQR(; positive = true, pivoted = false, blocksize = 8),
                LAPACK_HouseholderQR(; positive = true, pivoted = true, blocksize = 1),
                #LAPACK_HouseholderQR(; positive=true, pivoted=true, blocksize=8), # not supported
            )
            TestSuite.test_qr_algs(T, (m, n), LAPACK_QR_ALGS)
        elseif T ∈ GenericFloats
            TestSuite.test_qr(T, (m, n); test_null = true, test_pivoted = false, test_blocksize = false)
            GLA_QR_ALGS = (GLA_HouseholderQR(),)
            TestSuite.test_qr_algs(T, (m, n), GLA_QR_ALGS; test_null = false)
        end
        if m == n
            AT = Diagonal{T, Vector{T}}
            TestSuite.test_qr(AT, m; test_pivoted = false, test_blocksize = false)
            TestSuite.test_qr_algs(AT, m, (DiagonalAlgorithm(),))
        end
    end
end
