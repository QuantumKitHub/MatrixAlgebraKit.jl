using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: LinearAlgebra, I, isposdef

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
GenericFloats = (Float16, BigFloat, Complex{BigFloat})

@isdefined(TestSuite) || include("testsuite/TestSuite.jl")
using .TestSuite

m = 54
for T in BLASFloats, n in (37, m, 63)
    TestSuite.seed_rng!(123)
    TestSuite.test_polar(T, (m, n))
    if CUDA.functional()
        TestSuite.test_polar(CuMatrix{T}, (m, n); test_pivoted = false, test_blocksize = false)
        TestSuite.test_polar(Diagonal{T, CuVector{T}}, m; test_pivoted = false, test_blocksize = false)
    end
    if AMDGPU.functional()
        TestSuite.test_polar(ROCMatrix{T}, (m, n); test_pivoted = false, test_blocksize = false)
        TestSuite.test_polar(Diagonal{T, ROCVector{T}}, m; test_pivoted = false, test_blocksize = false)
    end
end
for T in (BLASFloats..., GenericFloats...)
    AT = Diagonal{T, Vector{T}}
    TestSuite.test_polar(AT, m; test_pivoted = false, test_blocksize = false)
end

