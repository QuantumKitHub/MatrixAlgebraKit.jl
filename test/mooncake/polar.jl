using MatrixAlgebraKit
using Test
using LinearAlgebra: Diagonal
using CUDA, AMDGPU

BLASFloats = (Float32, ComplexF64) # full suite is too expensive on CI
GenericFloats = ()
@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

m = 19
for T in (BLASFloats..., GenericFloats...), n in (17, m, 23)
    TestSuite.seed_rng!(123)
    if !is_buildkite
        atol = rtol = m * n * TestSuite.precision(T)
        m >= n && TestSuite.test_mooncake_left_polar(T, (m, n); atol, rtol)
        n >= m && TestSuite.test_mooncake_right_polar(T, (m, n); atol, rtol)
    end
end
