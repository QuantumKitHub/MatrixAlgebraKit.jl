using MatrixAlgebraKit
using MatrixAlgebraKit: Native
using LinearAlgebra: Diagonal

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

qr_algs = Tuple(
    alg for positive in (false, true) for alg in (
            Householder(; blocksize = 1, positive),
            Householder(; blocksize = 2, positive),
            Householder(; blocksize = 1, pivoted = true, positive),
            Householder(; driver = Native(), positive),
        )
)
lq_algs = Tuple(
    Householder(; positive, blocksize) for positive in (false, true) for blocksize in (1, 2)
)

for T in (Float64, ComplexF64), sz in ((7, 4), (4, 4), (4, 7))
    TestSuite.seed_rng!(123)
    TestSuite.test_orth_without_basis(T, sz, qr_algs, lq_algs)
    TestSuite.test_orth_basis_only(T, sz)
end
for T in (Float64, ComplexF64)
    TestSuite.seed_rng!(123)
    TestSuite.test_orth_without_basis(Diagonal{T, Vector{T}}, 4)
    TestSuite.test_orth_without_basis_derivatives(T, (7, 4))
end
TestSuite.seed_rng!(123)
TestSuite.test_lq_full_without_basis(Float64, (4, 7))
