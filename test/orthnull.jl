using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: LinearAlgebra, I, Diagonal
using CUDA, AMDGPU

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
GenericFloats = (BigFloat, Complex{BigFloat})

@isdefined(TestSuite) || include("testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

m = 54
for T in (BLASFloats..., GenericFloats...), n in (37, m, 63)
    TestSuite.seed_rng!(123)
    if T ∈ BLASFloats
        if CUDA.functional()
            TestSuite.test_orthnull(CuMatrix{T}, (m, n); test_nullity = false)
            n == m && TestSuite.test_orthnull(Diagonal{T, CuVector{T}}, m)
        end
        if AMDGPU.functional()
            TestSuite.test_orthnull(ROCMatrix{T}, (m, n); test_nullity = false)
            n == m && TestSuite.test_orthnull(Diagonal{T, ROCVector{T}}, m)
        end
    end
    if !is_buildkite
        TestSuite.test_orthnull(T, (m, n))
        AT = Diagonal{T, Vector{T}}
        TestSuite.test_orthnull(AT, m)
    end
end
