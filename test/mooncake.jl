using MatrixAlgebraKit
using Test
using LinearAlgebra: Diagonal
using CUDA, AMDGPU

#BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
BLASFloats = (Float32, ComplexF64) # full suite is too expensive on CI
GenericFloats = ()
@isdefined(TestSuite) || include("testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

m = 19
for T in (BLASFloats..., GenericFloats...), n in (17, m, 23)
    TestSuite.seed_rng!(123)
    #=if CUDA.functional()
        TestSuite.test_mooncake(CuMatrix{T}, (m, n); atol = m * n * TestSuite.precision(T), rtol = m * n * TestSuite.precision(T))
        #n == m && TestSuite.test_mooncake(Diagonal{T, CuVector{T}}, m; atol = m * TestSuite.precision(T), rtol = m * TestSuite.precision(T))
    end
    if AMDGPU.functional()
        TestSuite.test_mooncake(ROCMatrix{T}, (m, n); atol = m * n * TestSuite.precision(T), rtol = m * n * TestSuite.precision(T))
        TestSuite.test_mooncake(Diagonal{T, ROCVector{T}}, m; atol = m * TestSuite.precision(T), rtol = m * TestSuite.precision(T))
    end=# # not yet supported
    if !is_buildkite
        TestSuite.test_mooncake(T, (m, n); atol = m * n * TestSuite.precision(T), rtol = m * n * TestSuite.precision(T))
        #n == m && TestSuite.test_mooncake(Diagonal{T, Vector{T}}, m; atol = m * TestSuite.precision(T), rtol = m * TestSuite.precision(T))
    end
end
