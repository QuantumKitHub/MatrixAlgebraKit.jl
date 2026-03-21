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
                Jacobi(),
                DivideAndConquer(),
            )
            TestSuite.test_eigh(CuMatrix{T}, (m, m))
            TestSuite.test_eigh_algs(CuMatrix{T}, (m, m), CUSOLVER_EIGH_ALGS)
            TestSuite.test_eigh(Diagonal{T, CuVector{T}}, m)
            TestSuite.test_eigh_algs(Diagonal{T, CuVector{T}}, m, (DiagonalAlgorithm(),))
        end
        if AMDGPU.functional()
            ROCSOLVER_EIGH_ALGS = (
                Jacobi(),
                DivideAndConquer(),
                QRIteration(),
                Bisection(),
            )
            # see https://github.com/JuliaGPU/AMDGPU.jl/issues/837
            TestSuite.test_eigh(ROCMatrix{T}, (m, m); test_trunc = false)
            TestSuite.test_eigh_algs(ROCMatrix{T}, (m, m), ROCSOLVER_EIGH_ALGS; test_trunc = false)
            TestSuite.test_eigh(Diagonal{T, ROCVector{T}}, m; test_trunc = false)
            TestSuite.test_eigh_algs(Diagonal{T, ROCVector{T}}, m, (DiagonalAlgorithm(),); test_trunc = false)
        end
    end
    if !is_buildkite
        TestSuite.test_eigh(T, (m, m))
        if T ∈ BLASFloats
            LAPACK_EIGH_ALGS = (
                RobustRepresentations(),
                DivideAndConquer(),
                QRIteration(),
                Bisection(),
            )
            TestSuite.test_eigh_algs(T, (m, m), LAPACK_EIGH_ALGS)
        elseif T ∈ GenericFloats
            GLA_EIGH_ALGS = (QRIteration(),)
            TestSuite.test_eigh_algs(T, (m, m), GLA_EIGH_ALGS)
        end
        AT = Diagonal{T, Vector{T}}
        TestSuite.test_eigh(AT, m)
        TestSuite.test_eigh_algs(AT, m, (DiagonalAlgorithm(),))
    end
end
