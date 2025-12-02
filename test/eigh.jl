using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: LinearAlgebra, Diagonal, I
using CUDA, AMDGPU

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
GenericFloats = (Float16, BigFloat, Complex{BigFloat})

@isdefined(TestSuite) || include("testsuite/TestSuite.jl")
using .TestSuite

m = 54
for T in BLASFloats
    TestSuite.seed_rng!(123)
    TestSuite.test_eigh(T, (m, m))
    if CUDA.functional()
        TestSuite.test_eigh(CuMatrix{T}, (m, m); test_blocksize = false)
        TestSuite.test_eigh(Diagonal{T, CuVector{T}}, m; test_blocksize = false)
    end
    if AMDGPU.functional()
        TestSuite.test_eigh(ROCMatrix{T}, (m, m); test_blocksize = false)
        TestSuite.test_eigh(Diagonal{T, ROCVector{T}}, m; test_blocksize = false)
    end
end
for T in (BLASFloats..., GenericFloats...)
    AT = Diagonal{T, Vector{T}}
    TestSuite.test_eigh(AT, m; test_blocksize = false)
end
