using MatrixAlgebraKit
using Test
using LinearAlgebra: Diagonal
using CUDA, AMDGPU

BLASFloats = (Float64, ComplexF64) # full suite is too expensive on CI
GenericFloats = ()
@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

m = 19
for T in (BLASFloats..., GenericFloats...)
    TestSuite.seed_rng!(1234)
    atol = rtol = m * m * TestSuite.precision(T)
    if !is_buildkite
        TestSuite.test_enzyme_projections(T, (m, m); atol, rtol)
        TestSuite.test_enzyme_projections(Diagonal{T, Vector{T}}, (m, m); atol, rtol)
    end
    if CUDA.functional()
        TestSuite.test_enzyme_projections(CuMatrix{T}, (m, n); atol, rtol)
        TestSuite.test_enzyme_projections(Diagonal{T, CuVector{T}}, (m, m); atol, rtol)
    end
end
