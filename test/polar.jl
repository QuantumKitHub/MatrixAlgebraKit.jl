using MatrixAlgebraKit
using Test
using StableRNGs
using LinearAlgebra: LinearAlgebra, I, isposdef, Diagonal
using CUDA, AMDGPU

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
GenericFloats = (Float16, BigFloat, Complex{BigFloat})

@isdefined(TestSuite) || include("testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

m = 54
for T in BLASFloats, n in (37, m, 63)
    TestSuite.seed_rng!(123)
    if is_buildkite
        if CUDA.functional()
            TestSuite.test_polar(CuMatrix{T}, (m, n); test_pivoted = false, test_blocksize = false)
            # not supported
            #n == m && TestSuite.test_polar(Diagonal{T, CuVector{T}}, m; test_pivoted = false, test_blocksize = false)
        end
        if AMDGPU.functional()
            TestSuite.test_polar(ROCMatrix{T}, (m, n); test_pivoted = false, test_blocksize = false)
            # not supported
            #n == m && TestSuite.test_polar(Diagonal{T, ROCVector{T}}, m; test_pivoted = false, test_blocksize = false)
        end
    else
        TestSuite.test_polar(T, (m, n))
    end
end
if !is_buildkite
    for T in (BLASFloats..., GenericFloats...)
        AT = Diagonal{T, Vector{T}}
        TestSuite.test_polar(AT, m; test_pivoted = false, test_blocksize = false)
    end
end
