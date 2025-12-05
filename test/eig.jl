using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: Diagonal
using MatrixAlgebraKit: TruncatedAlgorithm, diagview, norm

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
GenericFloats = (Float16, BigFloat, Complex{BigFloat})

using CUDA, AMDGPU

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
GenericFloats = (Float16,) #BigFloat, Complex{BigFloat})

@isdefined(TestSuite) || include("testsuite/TestSuite.jl")
using .TestSuite

m = 54
for T in BLASFloats
    TestSuite.seed_rng!(123)
    TestSuite.test_eig(T, (m, m))
    if CUDA.functional()
        TestSuite.test_eig(CuMatrix{T}, (m, m); test_blocksize = false)
        TestSuite.test_eig(Diagonal{T, CuVector{T}}, m; test_blocksize = false)
    end
    #= not yet supported
    if AMDGPU.functional()
        TestSuite.test_eig(ROCMatrix{T}, (m, m); test_blocksize = false)
        TestSuite.test_eig(Diagonal{T, ROCVector{T}}, m; test_blocksize = false)
    end=#
end
for T in (BLASFloats..., GenericFloats...)
    AT = Diagonal{T, Vector{T}}
    TestSuite.test_eig(AT, m; test_blocksize = false)
end
