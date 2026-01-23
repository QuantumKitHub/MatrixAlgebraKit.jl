using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: Diagonal
using CUDA, AMDGPU
using CUDA.CUSOLVER # pull in opnorm binding

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
GenericFloats = (BigFloat, Complex{BigFloat})

@isdefined(TestSuite) || include("testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

for T in (BLASFloats..., GenericFloats...), m in (0, 54), n in (0, 37, m, 63)
    TestSuite.seed_rng!(123)
    if T ∈ BLASFloats
        if CUDA.functional()
            TestSuite.test_svd(CuMatrix{T}, (m, n))
            CUDA_SVD_ALGS = (
                CUSOLVER_QRIteration(),
                CUSOLVER_SVDPolar(),
                CUSOLVER_Jacobi(),
            )
            TestSuite.test_svd_algs(CuMatrix{T}, (m, n), CUDA_SVD_ALGS)
            k = 5
            p = min(m, n) - 2
            min(m, n) > k && TestSuite.test_randomized_svd(CuMatrix{T}, (m, n), (CUSOLVER_Randomized(; k, p, niters = 20),))
            if n == m
                TestSuite.test_svd(Diagonal{T, CuVector{T}}, m)
                TestSuite.test_svd_algs(Diagonal{T, CuVector{T}}, m, (DiagonalAlgorithm(),))
            end
        end
        if AMDGPU.functional()
            TestSuite.test_svd(ROCMatrix{T}, (m, n))
            AMD_SVD_ALGS = (
                ROCSOLVER_QRIteration(),
                ROCSOLVER_Jacobi(),
            )
            TestSuite.test_svd_algs(ROCMatrix{T}, (m, n), AMD_SVD_ALGS)
            if n == m
                TestSuite.test_svd(Diagonal{T, ROCVector{T}}, m)
                TestSuite.test_svd_algs(Diagonal{T, ROCVector{T}}, m, (DiagonalAlgorithm(),))
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
            TestSuite.test_svd(T, (m, n))
            TestSuite.test_svd_algs(T, (m, n), (GLA_QRIteration(),))
        end
        if m == n
            AT = Diagonal{T, Vector{T}}
            TestSuite.test_svd(AT, m)
            TestSuite.test_svd_algs(AT, m, (DiagonalAlgorithm(),))
        end
    end
end
