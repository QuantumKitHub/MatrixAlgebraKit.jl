using MatrixAlgebraKit
using Test
using LinearAlgebra: Diagonal
using CUDA, AMDGPU

BLASFloats = (ComplexF64,) # full suite is too expensive on CI
GenericFloats = (BigFloat,)
@isdefined(TestSuite) || include("testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

m = 19
for T in (BLASFloats..., GenericFloats...), n in (17, m, 23)
    TestSuite.seed_rng!(123)
    if T <: BLASFloats
        if CUDA.functional()
            TestSuite.test_enzyme(CuMatrix{T}, (m, n); atol = m * n * TestSuite.precision(T), rtol = m * n * TestSuite.precision(T))
            #n == m && TestSuite.test_enzyme(Diagonal{T, CuVector{T}}, m; atol = m * TestSuite.precision(T), rtol = m * TestSuite.precision(T))
        end
        if AMDGPU.functional()
            TestSuite.test_enzyme(ROCMatrix{T}, (m, n); atol = m * n * TestSuite.precision(T), rtol = m * n * TestSuite.precision(T))
            #TestSuite.test_enzyme(Diagonal{T, ROCVector{T}}, m; atol = m * TestSuite.precision(T), rtol = m * TestSuite.precision(T))
        end
    end
    if !is_buildkite
        TestSuite.test_enzyme(T, (m, n); atol = m * n * TestSuite.precision(T), rtol = m * n * TestSuite.precision(T))
        #n == m && TestSuite.test_enzyme(Diagonal{T, Vector{T}}, m; atol = m * TestSuite.precision(T), rtol = m * TestSuite.precision(T))
    end
end
