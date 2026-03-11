using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: mul!

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)

@isdefined(TestSuite) || include("testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

m, p, n, batch = 7, 5, 4, 10

for T in BLASFloats
    TestSuite.seed_rng!(123)
    if !is_buildkite
        TestSuite.test_strided_batched_mul(T, (m, p, n, batch))
        TestSuite.test_strided_batched_mul_algs(T, (m, p, n, batch), (LoopGEMM(),))
        TestSuite.test_batched_mul(T, (m, p, n, batch))
        TestSuite.test_batched_mul_algs(T, (m, p, n, batch), (LoopGEMM(),))
    end
end
