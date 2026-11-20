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
    TestSuite.seed_rng!(1234)
    if !is_buildkite
        TestSuite.test_mooncake_svd(T, (m, n); atol = m * n * TestSuite.precision(T), rtol = m * n * TestSuite.precision(T))
        if m == n
            AT = Diagonal{T, Vector{T}}
            TestSuite.test_mooncake_svd(AT, m; atol = m * n * TestSuite.precision(T), rtol = m * n * TestSuite.precision(T))
        end
    end
    if T ∈ BLASFloats && CUDA.functional()
        TestSuite.test_mooncake_svd(CuMatrix{T}, (m, n); atol = m * n * TestSuite.precision(T), rtol = m * n * TestSuite.precision(T))
        if m == n
            AT = Diagonal{T, CuVector{T}}
            TestSuite.test_mooncake_svd(AT, m; atol = m * n * TestSuite.precision(T), rtol = m * n * TestSuite.precision(T))
        end
        k = 5
        p = min(m, n) - k - 2
        alg = MatrixAlgebraKit.TruncatedAlgorithm(CUSOLVER_Randomized(; k, p, niters = 20), truncrank(k))
        TestSuite.test_mooncake_svd_trunc_randomized(CuMatrix{T}, (m, n), alg; atol = m * n * TestSuite.precision(T), rtol = m * n * TestSuite.precision(T))
    end
end
