using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: I, Diagonal
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

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

m = 54
for T in (BLASFloats..., GenericFloats...)
    TestSuite.seed_rng!(123)
    if T ∈ BLASFloats
        if CUDA.functional()
            # dense GPU schur is not yet supported: there is no `gees!` for CUSOLVER
            TestSuite.test_schur(Diagonal{T, CuVector{T}}, m)
            TestSuite.test_schur_algs(Diagonal{T, CuVector{T}}, m, (DiagonalAlgorithm(),))
        end
        #= not yet supported
        if AMDGPU.functional()
            TestSuite.test_schur(ROCMatrix{T}, (m, m); test_blocksize = false)
            TestSuite.test_schur(Diagonal{T, ROCVector{T}}, m; test_blocksize = false)
        end=#
    end
    if !is_buildkite
        TestSuite.test_schur(T, (m, m))
        if T ∈ BLASFloats
            LAPACK_SCHUR_ALGS = (QRIteration(), QRIteration(expert = true))
            TestSuite.test_schur_algs(T, (m, m), LAPACK_SCHUR_ALGS)
        end
        AT = Diagonal{T, Vector{T}}
        TestSuite.test_schur(AT, m)
        TestSuite.test_schur_algs(AT, m, (DiagonalAlgorithm(),))
    end
end
