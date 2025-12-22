using MatrixAlgebraKit
using Test
using StableRNGs
using LinearAlgebra: Diagonal
using CUDA, AMDGPU

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
GenericFloats = (BigFloat, Complex{BigFloat})

@isdefined(TestSuite) || include("testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

m = 54
for T in (BLASFloats..., GenericFloats...), n in (37, m, 63)
    TestSuite.seed_rng!(123)
    if T ∈ BLASFloats
        if CUDA.functional()
            # PolarNewton does not work yet on GPU
            CUDA_POLAR_ALGS = (PolarViaSVD.((CUSOLVER_QRIteration(), CUSOLVER_SVDPolar(), CUSOLVER_Jacobi()))...,) # PolarNewton())
            TestSuite.test_polar(CuMatrix{T}, (m, n), CUDA_POLAR_ALGS)
            #n == m && TestSuite.test_polar(Diagonal{T, CuVector{T}}, m, (PolarNewton(),))
        end
        if AMDGPU.functional()
            # PolarNewton does not work yet on GPU
            ROC_POLAR_ALGS = (PolarViaSVD.((ROCSOLVER_QRIteration(), ROCSOLVER_Jacobi()))...,) # PolarNewton())
            TestSuite.test_polar(ROCMatrix{T}, (m, n), ROC_POLAR_ALGS)
            #n == m && TestSuite.test_polar(Diagonal{T, ROCVector{T}}, m, (PolarNewton(),))
        end
    end
    if !is_buildkite
        if T ∈ BLASFloats
            LAPACK_POLAR_ALGS = (PolarViaSVD.((LAPACK_QRIteration(), LAPACK_Bisection(), LAPACK_DivideAndConquer()))..., PolarNewton())
            TestSuite.test_polar(T, (m, n), LAPACK_POLAR_ALGS)
            if LinearAlgebra.LAPACK.version() ≥ v"3.12.0"
                LAPACK_JACOBI = (PolarViaSVD(LAPACK_Jacobi()),)
                TestSuite.test_polar(T, (m, n), LAPACK_JACOBI; test_right = false)
            end
        elseif T ∈ GenericFloats
            GLA_POLAR_ALGS = (PolarViaSVD.((GLA_QRIteration(),))..., PolarNewton())
            TestSuite.test_polar(T, (m, n), GLA_POLAR_ALGS)
        end
        if m == n
            AT = Diagonal{T, Vector{T}}
            TestSuite.test_polar(AT, m, (PolarNewton(),))
        end
    end
end
