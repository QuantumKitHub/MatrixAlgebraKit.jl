using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: I, Diagonal

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
GenericFloats = (BigFloat, Complex{BigFloat})

@isdefined(TestSuite) || include("testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

m = 54
for T in (BLASFloats..., GenericFloats...)
    TestSuite.seed_rng!(123)
    if T ∈ BLASFloats
        #=if CUDA.functional()
            TestSuite.test_schur(CuMatrix{T}, (m, m); test_blocksize = false)
            TestSuite.test_schur(Diagonal{T, CuVector{T}}, m; test_blocksize = false)
        end
        if AMDGPU.functional()
            TestSuite.test_schur(ROCMatrix{T}, (m, m); test_blocksize = false)
            TestSuite.test_schur(Diagonal{T, ROCVector{T}}, m; test_blocksize = false)
        end=# # not yet supported
    end
    if !is_buildkite
        TestSuite.test_schur(T, (m, m))
        #AT = Diagonal{T, Vector{T}}
        #TestSuite.test_schur(AT, m) # not supported yet
    end
end
