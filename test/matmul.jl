using MKL
using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: BLAS, mul!

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)

@isdefined(TestSuite) || include("testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"
is_mkl = any(lib -> contains(lib.libname, "mkl"), BLAS.get_config().loaded_libs)

m, p, n, batch = 7, 5, 4, 10
gemm_algs = is_mkl ? (LoopGEMM(), GEMM()) : (LoopGEMM(),)

for T in BLASFloats
    TestSuite.seed_rng!(123)
    if !is_buildkite
        TestSuite.test_strided_batched_mul(T, (m, p, n, batch))
        TestSuite.test_strided_batched_mul_algs(T, (m, p, n, batch), gemm_algs)
        TestSuite.test_batched_mul(T, (m, p, n, batch))
        TestSuite.test_batched_mul_algs(T, (m, p, n, batch), gemm_algs)
    end
end
