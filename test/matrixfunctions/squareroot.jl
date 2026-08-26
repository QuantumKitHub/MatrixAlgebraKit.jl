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
HalfFloats = (Float16, ComplexF16)

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

m = 54
md = 4   # the domain tests prescribe the full spectrum, so keep them small
mh = 8   # half precision only holds up on a small matrix

# What half precision can be asked of `squareroot` is set by `schur_full`, not by the kernel, whose
# own residual stays below `eps(Float16)` here
rtol16 = 0.1

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
        SCHUR_ALGS = (
            MatrixFunctionViaSchur(),
            MatrixFunctionViaSchur(; schur_alg = QRIteration(; expert = true)),
        )
        TestSuite.test_squareroot(T, (m, m))
        TestSuite.test_squareroot_algs(T, (m, m), LAPACK_EIG_ALGS)
        TestSuite.test_squareroot_algs(T, (m, m), SCHUR_ALGS)
        TestSuite.test_squareroot_hermitian(T, (m, m), LAPACK_EIG_ALGS; exact_hermiticity = false)
        TestSuite.test_squareroot_hermitian(T, (m, m), SCHUR_ALGS; exact_hermiticity = false)
        TestSuite.test_squareroot_hermitian(T, (m, m), LAPACK_EIGH_ALGS)
        TestSuite.test_squareroot_reference(T, (m, m))
        TestSuite.test_squareroot_blocked(T, (m, m), (MatrixFunctionViaSchur(),))
        TestSuite.test_squareroot_defective(T, (6, 6), (MatrixFunctionViaSchur(),))
        # `MatrixFunctionViaLA` has no access to the spectrum, and thus no `domain_atol`
        TestSuite.test_squareroot_domain(T, (md, md), (MatrixFunctionViaLA(),); test_domain_atol = false)
        TestSuite.test_squareroot_domain(T, (md, md), (MatrixFunctionViaEig(QRIteration()),))
        TestSuite.test_squareroot_domain(T, (md, md), (MatrixFunctionViaSchur(),))
        TestSuite.test_squareroot_domain(T, (md, md), LAPACK_EIGH_ALGS; hermitian_output = true)
    end

    # `eig` comes from GenericSchur, `eigh` from GenericLinearAlgebra. Both are loaded here, so name
    # the driver explicitly instead of relying on `default_driver`.
    for T in GenericFloats
        TestSuite.seed_rng!(123)
        GS_ALGS = (MatrixFunctionViaEig(QRIteration(; driver = GS())),)
        GLA_ALGS = (MatrixFunctionViaEigh(QRIteration(; driver = GLA())),)
        GS_SCHUR_ALGS = (MatrixFunctionViaSchur(; schur_alg = QRIteration(; driver = GS())),)
        # `LinearAlgebra.sqrt` promotes a real generic matrix with complex eigenvalues to a complex
        # Schur form, so the Schur route is the only one here that is both real and backward stable
        TestSuite.test_squareroot(T, (24, 24))
        TestSuite.test_squareroot_algs(T, (24, 24), GS_ALGS)
        TestSuite.test_squareroot_algs(T, (24, 24), GS_SCHUR_ALGS)
        TestSuite.test_squareroot_hermitian(T, (24, 24), GS_ALGS; exact_hermiticity = false)
        TestSuite.test_squareroot_hermitian(T, (24, 24), GS_SCHUR_ALGS; exact_hermiticity = false)
        TestSuite.test_squareroot_hermitian(T, (24, 24), GLA_ALGS)
        TestSuite.test_squareroot_blocked(T, (12, 12), GS_SCHUR_ALGS)
        TestSuite.test_squareroot_domain(T, (md, md), GS_ALGS)
        TestSuite.test_squareroot_domain(T, (md, md), GS_SCHUR_ALGS)
        TestSuite.test_squareroot_domain(T, (md, md), GLA_ALGS; hermitian_output = true)
    end

    # `eigh` is unavailable in half precision, so the Schur route is what covers these
    for T in HalfFloats
        TestSuite.seed_rng!(123)
        HALF_SCHUR_ALGS = (MatrixFunctionViaSchur(; schur_alg = QRIteration(; driver = GS())),)
        TestSuite.test_squareroot(T, (mh, mh); rtol = rtol16)
        TestSuite.test_squareroot_algs(T, (mh, mh), HALF_SCHUR_ALGS; rtol = rtol16)
        TestSuite.test_squareroot_domain(T, (md, md), HALF_SCHUR_ALGS; atol = rtol16)
        # the hermitian generator `A * A' + I` is out of reach: on it `GenericSchur` returns a
        # half-precision decomposition wrong by `0.4` relative (`6e-7` in `Float32`), so nothing
        # downstream of it can be asserted
    end

    for T in (BLASFloats..., GenericFloats..., HalfFloats...)
        TestSuite.seed_rng!(123)
        AT = Diagonal{T, Vector{T}}
        test_spectrum = !(T in HalfFloats)
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
