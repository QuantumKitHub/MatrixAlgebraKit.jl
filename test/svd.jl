using MatrixAlgebraKit
using LinearAlgebra: Diagonal
using CUDA, AMDGPU

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
GenericFloats = (BigFloat, Complex{BigFloat})

@isdefined(TestSuite) || include("testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

# CPU tests
# ---------
if !is_buildkite
    # LAPACK algorithms:
    for T in BLASFloats, m in (0, 54), n in (0, 37, m, 63)
        TestSuite.seed_rng!(123)
        LAPACK_SVD_ALGS = (QRIteration(), DivideAndConquer(), SafeDivideAndConquer(; fixgauge = true))
        TestSuite.test_svd(T, (m, n))
        TestSuite.test_svd_algs(T, (m, n), LAPACK_SVD_ALGS)
        @static if VERSION > v"1.11-" # Jacobi broken on 1.10
            m ≥ n && TestSuite.test_svd_algs(T, (m, n), (LAPACK_Jacobi(),); test_full = false, test_vals = false)
        end
    end

    # Generic floats:
    for T in GenericFloats, m in (0, 54), n in (0, 37, m, 63)
        TestSuite.seed_rng!(123)
        TestSuite.test_svd(T, (m, n))
        TestSuite.test_svd_algs(T, (m, n), (GLA_QRIteration(),))
    end

    # Diagonal:
    for T in (BLASFloats..., GenericFloats...), m in (0, 54)
        TestSuite.seed_rng!(123)
        AT = Diagonal{T, Vector{T}}
        TestSuite.test_svd(AT, m)
        TestSuite.test_svd_algs(AT, m, (DiagonalAlgorithm(),))
    end
end

# CUDA tests
# ------------
if CUDA.functional()
    # LAPACK algorithms:
    for T in BLASFloats, m in (0, 23), n in (0, 17, m, 27)
        TestSuite.seed_rng!(123)
        TestSuite.test_svd(CuMatrix{T}, (m, n))
        CUDA_SVD_ALGS = (QRIteration(), SVDViaPolar(), Jacobi())
        TestSuite.test_svd_algs(CuMatrix{T}, (m, n), CUDA_SVD_ALGS)
    end

    # Randomized SVD:
    for T in BLASFloats, m in (0, 23), n in (0, 17, m, 27)
        TestSuite.seed_rng!(123)
        k = 5
        p = min(m, n) - k - 2
        p > 0 || continue
        TestSuite.test_randomized_svd(CuMatrix{T}, (m, n), (MatrixAlgebraKit.TruncatedAlgorithm(CUSOLVER_Randomized(; k, p, niters = 20), truncrank(k)),))
    end

    # Diagonal:
    for T in BLASFloats, m in (0, 23)
        TestSuite.seed_rng!(123)
        AT = Diagonal{T, CuVector{T}}
        TestSuite.test_svd(AT, m)
        TestSuite.test_svd_algs(AT, m, (DiagonalAlgorithm(),))
    end
end

# AMDGPU tests
# ------------
if AMDGPU.functional()
    # LAPACK algorithms:
    for T in BLASFloats, m in (0, 23), n in (0, 17, m, 27)
        TestSuite.seed_rng!(123)
        TestSuite.test_svd(ROCMatrix{T}, (m, n))
        AMD_SVD_ALGS = (QRIteration(), Jacobi())
        TestSuite.test_svd_algs(ROCMatrix{T}, (m, n), AMD_SVD_ALGS)
    end

    # Diagonal:
    for T in BLASFloats, m in (0, 23)
        TestSuite.seed_rng!(123)
        AT = Diagonal{T, ROCVector{T}}
        TestSuite.test_svd(AT, m)
        TestSuite.test_svd_algs(AT, m, (DiagonalAlgorithm(),))
    end
end
