using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: Diagonal
using CUDA, AMDGPU

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
GenericFloats = (BigFloat, Complex{BigFloat})

@isdefined(TestSuite) || include("testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

for T in (BLASFloats..., GenericFloats...), m in (0, 54), n in (0, 37, m, 63)
    TestSuite.seed_rng!(123)
    if T ∈ BLASFloats
        if CUDA.functional()
            TestSuite.test_svd(CuMatrix{T}, (m, n); test_trunc = false)
            CUDA_SVD_ALGS = (
                CUSOLVER_QRIteration(),
                CUSOLVER_SVDPolar(),
                CUSOLVER_Jacobi(),
            )
            TestSuite.test_svd_algs(CuMatrix{T}, (m, n), CUDA_SVD_ALGS; test_trunc = false)
            if n == m
                TestSuite.test_svd(Diagonal{T, CuVector{T}}, m; test_trunc = false)
                TestSuite.test_svd_algs(Diagonal{T, CuVector{T}}, m, (DiagonalAlgorithm(),); test_trunc = false)
            end
        end
        if AMDGPU.functional()
            TestSuite.test_svd(ROCMatrix{T}, (m, n); test_trunc = false)
            AMD_SVD_ALGS = (
                ROCSOLVER_QRIteration(),
                ROCSOLVER_SVDPolar(),
                ROCSOLVER_Jacobi(),
            )
            TestSuite.test_svd_algs(ROCMatrix{T}, (m, n), AMD_SVD_ALGS; test_trunc = false)
            if n == m
                TestSuite.test_svd(Diagonal{T, ROCVector{T}}, m; test_trunc = false)
                TestSuite.test_svd_algs(Diagonal{T, ROCVector{T}}, m, (DiagonalAlgorithm(),); test_trunc = false)
            end
        end
    end
    if !is_buildkite
        if T ∈ BLASFloats
            LAPACK_SVD_ALGS = (
                LAPACK_QRIteration(),
                LAPACK_DivideAndConquer(),
            )
            TestSuite.test_svd(T, (m, n))
            TestSuite.test_svd_algs(T, (m, n), LAPACK_SVD_ALGS)
        elseif T ∈ GenericFloats
            TestSuite.test_svd(T, (m, n); test_trunc = false)
            TestSuite.test_svd_algs(T, (m, n), (GLA_QRIteration(),); test_trunc = false)
        end
        if m == n
            AT = Diagonal{T, Vector{T}}
            TestSuite.test_svd(AT, m; test_trunc = !(T ∈ GenericFloats))
            TestSuite.test_svd_algs(AT, m, (DiagonalAlgorithm(),); test_trunc = !(T ∈ GenericFloats))
        end
    end
end
