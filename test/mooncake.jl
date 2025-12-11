using MatrixAlgebraKit
using Test
using CUDA, AMDGPU

#BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
BLASFloats = (Float32, ComplexF64) # full suite is too expensive on CI

@isdefined(TestSuite) || include("testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

m = 19
for T in BLASFloats, n in (17, m, 23)
    TestSuite.seed_rng!(123)
    if is_buildkite
        if CUDA.functional()
            TestSuite.test_mooncake(CuMatrix{T}, (m, n); atol = m * n * TestSuite.precision(T), rtol = m * n * TestSuite.precision(T))
        end
        #=if AMDGPU.functional()
            TestSuite.test_mooncake(ROCMatrix{T}, (m, n); atol = m * n * TestSuite.precision(T), rtol = m * n * TestSuite.precision(T))
        end=# # not yet supported
    else
        TestSuite.test_mooncake(T, (m, n); atol = m * n * TestSuite.precision(T), rtol = m * n * TestSuite.precision(T))
    end
end
