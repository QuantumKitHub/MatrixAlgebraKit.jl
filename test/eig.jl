using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: Diagonal
using MatrixAlgebraKit: TruncatedAlgorithm, diagview, norm
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
            TestSuite.test_eig(CuMatrix{T}, (m, m); test_trunc = false)
            TestSuite.test_eig_algs(CuMatrix{T}, (m, m), (CUSOLVER_Simple(),))
            TestSuite.test_eig(Diagonal{T, CuVector{T}}, m)
            TestSuite.test_eig_algs(Diagonal{T, CuVector{T}}, m, (DiagonalAlgorithm(),))
        end
        #= not yet supported
        if AMDGPU.functional()
            TestSuite.test_eig(ROCMatrix{T}, (m, m); test_blocksize = false)
            TestSuite.test_eig_algs(ROCMatrix{T}, (m, m), (ROCSOLVER_Simple(),))
            TestSuite.test_eig(Diagonal{T, ROCVector{T}}, m; test_blocksize = false)
            TestSuite.test_eig_algs(Diagonal{T, ROCVector{T}}, m, (DiagonalAlgorithm(),))
        end=#
    end
    if !is_buildkite
        TestSuite.test_eig(T, (m, m))
        if T ∈ BLASFloats
            LAPACK_EIG_ALGS = (LAPACK_Simple(), LAPACK_Expert())
            TestSuite.test_eig_algs(T, (m, m), LAPACK_EIG_ALGS)
        elseif T ∈ GenericFloats
            GS_EIG_ALGS = (GS_QRIteration(),)
            TestSuite.test_eig_algs(T, (m, m), GS_EIG_ALGS)
        end
        AT = Diagonal{T, Vector{T}}
        TestSuite.test_eig(AT, m)
        TestSuite.test_eig_algs(AT, m, (DiagonalAlgorithm(),))
    end
end
