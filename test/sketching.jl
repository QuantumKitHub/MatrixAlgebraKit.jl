using MatrixAlgebraKit

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)

@isdefined(TestSuite) || include("testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

# CPU tests
# ---------
if !is_buildkite
    @testset "Sketching ($T, $m, $n)" for T in BLASFloats, (m, n) in ((100, 40), (40, 100), (60, 60))
        TestSuite.seed_rng!(123)
        TestSuite.test_sketching(T, (m, n))
    end
end
