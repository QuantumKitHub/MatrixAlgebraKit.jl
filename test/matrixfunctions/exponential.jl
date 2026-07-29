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

# CPU tests
# ---------
if !is_buildkite
    # LAPACK algorithms:
    for T in BLASFloats
        TestSuite.seed_rng!(123)
        LAPACK_EIG_ALGS = (
            MatrixFunctionViaLA(),
            MatrixFunctionViaEig(QRIteration()),
            MatrixFunctionViaTaylor(),
        )
        LAPACK_EIGH_ALGS = (MatrixFunctionViaEigh(QRIteration()), MatrixFunctionViaEigh(DivideAndConquer()))
        TestSuite.test_exponential(T, (m, m))
        TestSuite.test_exponential_algs(T, (m, m), LAPACK_EIG_ALGS)
        TestSuite.test_exponential_scaled(T, (m, m), LAPACK_EIG_ALGS)
        TestSuite.test_exponential_hermitian(T, (m, m), LAPACK_EIGH_ALGS)
        TestSuite.test_exponential_taylor(T, (m, m))
        TestSuite.test_exponential_reference(T, (m, m))
        TestSuite.test_exponential_wrappers(T, (12, 12))
        # `eigh` rejects non-hermitian input rather than projecting it
        TestSuite.test_exponential_domain(T, (m, m), LAPACK_EIGH_ALGS)
    end

    # Generic floats: `eig` comes from GenericSchur, `eigh` from GenericLinearAlgebra. Both are
    # loaded here, so name the driver explicitly instead of relying on `default_driver`. The
    # native Taylor algorithm needs no LAPACK support and applies at arbitrary precision.
    for T in GenericFloats
        TestSuite.seed_rng!(123)
        GS_ALGS = (MatrixFunctionViaEig(QRIteration(; driver = GS())), MatrixFunctionViaTaylor())
        GLA_ALGS = (MatrixFunctionViaEigh(QRIteration(; driver = GLA())),)
        TestSuite.test_exponential_algs(T, (24, 24), GS_ALGS)
        TestSuite.test_exponential_scaled(T, (24, 24), GS_ALGS)
        TestSuite.test_exponential_hermitian(T, (24, 24), GLA_ALGS)
        TestSuite.test_exponential_taylor(T, (24, 24))
    end

    # Diagonal:
    for T in (BLASFloats..., GenericFloats..., DiagonalOnlyFloats...)
        TestSuite.seed_rng!(123)
        AT = Diagonal{T, Vector{T}}
        test_spectrum = !(T in DiagonalOnlyFloats)
        TestSuite.test_exponential(AT, m)
        TestSuite.test_exponential_algs(AT, m, (DiagonalAlgorithm(),))
        TestSuite.test_exponential_scaled(AT, m, (DiagonalAlgorithm(),))
        TestSuite.test_exponential_hermitian(AT, m, (DiagonalAlgorithm(),); test_spectrum)
        TestSuite.test_exponential_reference(AT, m; test_hermitian = !(T in GenericFloats))
    end
end

# CUDA tests
# ----------
# Unlike the other matrix functions, a general dense matrix *is* supported on device, through the
# native `MatrixFunctionViaTaylor`. The `logarithm` roundtrip is unavailable there, so those calls
# rely on the `exp(A) * exp(-A) ≈ I` invariant instead.
if CUDA.functional()
    for T in BLASFloats
        TestSuite.seed_rng!(123)
        CUDA_EIGH_ALGS = (MatrixFunctionViaEigh(Jacobi()), MatrixFunctionViaEigh(DivideAndConquer()))
        TestSuite.test_exponential_algs(CuMatrix{T}, (m, m), (MatrixFunctionViaTaylor(),); test_roundtrip = false)
        TestSuite.test_exponential_scaled(CuMatrix{T}, (m, m), (MatrixFunctionViaTaylor(),))
        TestSuite.test_exponential_taylor(CuMatrix{T}, (m, m))
        TestSuite.test_exponential_hermitian(CuMatrix{T}, (m, m), CUDA_EIGH_ALGS)
        TestSuite.test_exponential_domain(CuMatrix{T}, (m, m), CUDA_EIGH_ALGS)

        AT = Diagonal{T, CuVector{T}}
        TestSuite.test_exponential(AT, m)
        TestSuite.test_exponential_algs(AT, m, (DiagonalAlgorithm(),))
        TestSuite.test_exponential_scaled(AT, m, (DiagonalAlgorithm(),))
        TestSuite.test_exponential_hermitian(AT, m, (DiagonalAlgorithm(),))
    end
end

# AMDGPU tests
# ------------
if AMDGPU.functional()
    for T in BLASFloats
        TestSuite.seed_rng!(123)
        ROC_EIGH_ALGS = (MatrixFunctionViaEigh(Jacobi()), MatrixFunctionViaEigh(DivideAndConquer()))
        TestSuite.test_exponential_algs(ROCMatrix{T}, (m, m), (MatrixFunctionViaTaylor(),); test_roundtrip = false)
        TestSuite.test_exponential_scaled(ROCMatrix{T}, (m, m), (MatrixFunctionViaTaylor(),))
        TestSuite.test_exponential_taylor(ROCMatrix{T}, (m, m))
        TestSuite.test_exponential_hermitian(ROCMatrix{T}, (m, m), ROC_EIGH_ALGS)
        TestSuite.test_exponential_domain(ROCMatrix{T}, (m, m), ROC_EIGH_ALGS)

        AT = Diagonal{T, ROCVector{T}}
        TestSuite.test_exponential(AT, m)
        TestSuite.test_exponential_algs(AT, m, (DiagonalAlgorithm(),))
        TestSuite.test_exponential_scaled(AT, m, (DiagonalAlgorithm(),))
        TestSuite.test_exponential_hermitian(AT, m, (DiagonalAlgorithm(),))
    end
end
