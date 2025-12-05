using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: LinearAlgebra, Diagonal, norm, normalize!
using CUDA, AMDGPU

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
GenericFloats = (Float16, BigFloat, Complex{BigFloat})

@isdefined(TestSuite) || include("testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"


m = 54
for T in BLASFloats
    TestSuite.seed_rng!(123)
    if is_buildkite
        if CUDA.functional()
            TestSuite.test_projections(CuMatrix{T}, (m, m); test_pivoted = false, test_blocksize = false)
            TestSuite.test_projections(Diagonal{T, CuVector{T}}, m; test_pivoted = false, test_blocksize = false)
        end
        if AMDGPU.functional()
            TestSuite.test_projections(ROCMatrix{T}, (m, m); test_pivoted = false, test_blocksize = false)
            TestSuite.test_projections(Diagonal{T, ROCVector{T}}, m; test_pivoted = false, test_blocksize = false)
        end
    else
        TestSuite.test_projections(T, (m, m))
    end
end
if !is_buildkite
    for T in (BLASFloats..., GenericFloats...)
        AT = Diagonal{T, Vector{T}}
        TestSuite.test_projections(AT, m; test_pivoted = false, test_blocksize = false)
    end
end
