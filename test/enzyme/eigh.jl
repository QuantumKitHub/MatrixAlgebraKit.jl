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
for T in (BLASFloats..., GenericFloats...)
    TestSuite.seed_rng!(123)
    if !is_buildkite
        TestSuite.test_enzyme_eigh(T, (m, m); atol = m * m * TestSuite.precision(T), rtol = m * m * TestSuite.precision(T))
    end
end
