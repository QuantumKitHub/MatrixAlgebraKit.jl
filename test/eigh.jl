using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: LinearAlgebra, Diagonal, I
using CUDA, AMDGPU

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
GenericFloats = (BigFloat, Complex{BigFloat})

@isdefined(TestSuite) || include("testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

m = 54
for T in (BLASFloats..., GenericFloats...)
    TestSuite.seed_rng!(123)
    if T ∈ BLASFloats
        if CUDA.functional()
            CUSOLVER_EIGH_ALGS = (
                CUSOLVER_Jacobi(),
                CUSOLVER_DivideAndConquer(),
                CUSOLVER_QRIteration(),
                CUSOLVER_Bisection(),
            )
            TestSuite.test_eigh(CuMatrix{T}, (m, m); test_trunc = false)
            TestSuite.test_eigh_algs(CuMatrix{T}, (m, m), CUSOLVER_EIGH_ALGS; test_trunc = false)
            TestSuite.test_eigh(Diagonal{T, CuVector{T}}, m)
            TestSuite.test_eigh_algs(Diagonal{T, CuVector{T}}, m, (DiagonalAlgorithm(),))
        end
        if AMDGPU.functional()
            ROCSOLVER_EIGH_ALGS = (
                ROCSOLVER_Jacobi(),
                ROCSOLVER_DivideAndConquer(),
                ROCSOLVER_QRIteration(),
                ROCSOLVER_Bisection(),
            )
            TestSuite.test_eigh(ROCMatrix{T}, (m, m); test_trunc = false)
            TestSuite.test_eigh(ROCMatrix{T}, (m, m), ROCSOLVER_EIGH_ALGS; test_trunc = false)
            TestSuite.test_eigh(Diagonal{T, ROCVector{T}}, m)
            TestSuite.test_eigh_algs(Diagonal{T, ROCVector{T}}, m, (DiagonalAlgorithm(),))
        end
    end
    if !is_buildkite
        TestSuite.test_eigh(T, (m, m))
        if T ∈ BLASFloats
            LAPACK_EIGH_ALGS = (
                LAPACK_MultipleRelativelyRobustRepresentations(),
                LAPACK_DivideAndConquer(),
                LAPACK_QRIteration(),
                LAPACK_Bisection(),
            )
            TestSuite.test_eigh_algs(T, (m, m), LAPACK_EIGH_ALGS)
        elseif T ∈ GenericFloats
            GLA_EIGH_ALGS = (GLA_QRIteration(),)
            TestSuite.test_eigh_algs(T, (m, m), GLA_EIGH_ALGS)
        end
        AT = Diagonal{T, Vector{T}}
        TestSuite.test_eigh(AT, m)
        TestSuite.test_eigh_algs(AT, m, (DiagonalAlgorithm(),))
    end
end
