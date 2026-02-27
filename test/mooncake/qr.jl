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
        TestSuite.test_mooncake_qr(T, (m, n); atol = m * n * TestSuite.precision(T), rtol = m * n * TestSuite.precision(T))
        if m == n
            AT = Diagonal{T, Vector{T}}
            TestSuite.test_mooncake_qr(AT, m; atol = m * n * TestSuite.precision(T), rtol = m * n * TestSuite.precision(T))
        end
    end
end
