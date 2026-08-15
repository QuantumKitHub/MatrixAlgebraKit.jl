using MatrixAlgebraKit
using LinearAlgebra: Diagonal
using MatrixAlgebraKit: GLA, GS
using CUDA, AMDGPU
using GenericSchur, GenericLinearAlgebra

if @isdefined(fast_tests) && fast_tests
    BLASFloats = (Float64, ComplexF64)
    GenericFloats = (BigFloat, Complex{BigFloat})
else
    BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
    GenericFloats = (BigFloat, Complex{BigFloat})
end
# only the `Diagonal` fast path applies to these, as they have no `eig`/`eigh` support
DiagonalOnlyFloats = (Float16, ComplexF16)

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

m = 54
md = 4   # the domain tests prescribe the full spectrum, so keep them small

# CPU tests
# ---------
if !is_buildkite
    for T in BLASFloats
        TestSuite.seed_rng!(123)
        LAPACK_EIG_ALGS = (MatrixFunctionViaLA(), MatrixFunctionViaEig(QRIteration()))
        LAPACK_EIGH_ALGS = (
            MatrixFunctionViaEigh(QRIteration()),
            MatrixFunctionViaEigh(DivideAndConquer()),
        )
        TestSuite.test_squareroot(T, (m, m))
        TestSuite.test_squareroot_algs(T, (m, m), LAPACK_EIG_ALGS)
        TestSuite.test_squareroot_hermitian(T, (m, m), LAPACK_EIG_ALGS; exact_hermiticity = false)
        TestSuite.test_squareroot_hermitian(T, (m, m), LAPACK_EIGH_ALGS)
        TestSuite.test_squareroot_reference(T, (m, m))
        TestSuite.test_squareroot_domain(T, (md, md), LAPACK_EIG_ALGS)
        TestSuite.test_squareroot_domain(T, (md, md), LAPACK_EIGH_ALGS; hermitian_output = true)
    end

    # `eig` comes from GenericSchur, `eigh` from GenericLinearAlgebra. Both are loaded here, so name
    # the driver explicitly instead of relying on `default_driver`.
    for T in GenericFloats
        TestSuite.seed_rng!(123)
        GS_ALGS = (MatrixFunctionViaEig(QRIteration(; driver = GS())),)
        GLA_ALGS = (MatrixFunctionViaEigh(QRIteration(; driver = GLA())),)
        TestSuite.test_squareroot_algs(T, (24, 24), GS_ALGS)
        TestSuite.test_squareroot_hermitian(T, (24, 24), GS_ALGS; exact_hermiticity = false)
        TestSuite.test_squareroot_hermitian(T, (24, 24), GLA_ALGS)
        TestSuite.test_squareroot_domain(T, (md, md), GS_ALGS)
        TestSuite.test_squareroot_domain(T, (md, md), GLA_ALGS; hermitian_output = true)
    end

    for T in (BLASFloats..., GenericFloats..., DiagonalOnlyFloats...)
        TestSuite.seed_rng!(123)
        AT = Diagonal{T, Vector{T}}
        test_spectrum = !(T in DiagonalOnlyFloats)
        TestSuite.test_squareroot(AT, m)
        TestSuite.test_squareroot_algs(AT, m, (DiagonalAlgorithm(),))
        TestSuite.test_squareroot_hermitian(AT, m, (DiagonalAlgorithm(),); test_spectrum)
        TestSuite.test_squareroot_reference(AT, m; test_hermitian = !(T in GenericFloats))
        TestSuite.test_squareroot_domain(AT, md, (DiagonalAlgorithm(),))
    end
end

# CUDA tests
# ----------
# general dense matrices are not supported on device: `MatrixFunctionViaLA` would call LAPACK on
# device memory, and `MatrixFunctionViaEig` scalar-indexes in its `lu!`-based solve
if CUDA.functional()
    for T in BLASFloats
        TestSuite.seed_rng!(123)
        CUDA_EIGH_ALGS = (
            MatrixFunctionViaEigh(Jacobi()),
            MatrixFunctionViaEigh(DivideAndConquer()),
        )
        TestSuite.test_squareroot_hermitian(CuMatrix{T}, (m, m), CUDA_EIGH_ALGS)
        TestSuite.test_squareroot_domain(CuMatrix{T}, (md, md), CUDA_EIGH_ALGS; hermitian_output = true)

        AT = Diagonal{T, CuVector{T}}
        TestSuite.test_squareroot(AT, m)
        TestSuite.test_squareroot_algs(AT, m, (DiagonalAlgorithm(),))
        TestSuite.test_squareroot_hermitian(AT, m, (DiagonalAlgorithm(),))
        TestSuite.test_squareroot_domain(AT, md, (DiagonalAlgorithm(),))
    end
end

# AMDGPU tests
# ------------
if AMDGPU.functional()
    for T in BLASFloats
        TestSuite.seed_rng!(123)
        ROC_EIGH_ALGS = (
            MatrixFunctionViaEigh(Jacobi()),
            MatrixFunctionViaEigh(DivideAndConquer()),
        )
        TestSuite.test_squareroot_hermitian(ROCMatrix{T}, (m, m), ROC_EIGH_ALGS)
        TestSuite.test_squareroot_domain(ROCMatrix{T}, (md, md), ROC_EIGH_ALGS; hermitian_output = true)

        AT = Diagonal{T, ROCVector{T}}
        TestSuite.test_squareroot(AT, m)
        TestSuite.test_squareroot_algs(AT, m, (DiagonalAlgorithm(),))
        TestSuite.test_squareroot_hermitian(AT, m, (DiagonalAlgorithm(),))
        TestSuite.test_squareroot_domain(AT, md, (DiagonalAlgorithm(),))
    end
end
