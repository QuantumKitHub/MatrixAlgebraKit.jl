using SafeTestsets

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"
# don't run all tests on GPU, only the GPU
# specific ones
if !is_buildkite
    @safetestset "Algorithms" begin
        include("algorithms.jl")
    end
    @safetestset "Truncate" begin
        include("truncate.jl")
    end
    @safetestset "Singular Value Decomposition" begin
        include("svd.jl")
    end
    @safetestset "Generalized Eigenvalue Decomposition" begin
        include("gen_eig.jl")
    end
    @safetestset "Mooncake" begin
        include("mooncake.jl")
    end
    @safetestset "ChainRules" begin
        include("chainrules.jl")
    end
    @safetestset "MatrixAlgebraKit.jl" begin
        @safetestset "Code quality (Aqua.jl)" begin
            using MatrixAlgebraKit
            using Aqua
            Aqua.test_all(MatrixAlgebraKit)
        end
        @safetestset "Code linting (JET.jl)" begin
            using MatrixAlgebraKit
            using JET
            JET.test_package(MatrixAlgebraKit; target_defined_modules = true)
        end
    end

    using GenericLinearAlgebra
    @safetestset "Singular Value Decomposition" begin
        include("genericlinearalgebra/svd.jl")
    end
end

@safetestset "QR / LQ Decomposition" begin
    include("qr.jl")
    include("lq.jl")
end
@safetestset "Polar Decomposition" begin
    include("polar.jl")
end
@safetestset "Projections" begin
    include("projections.jl")
end
@safetestset "Schur Decomposition" begin
    include("schur.jl")
end
@safetestset "General Eigenvalue Decomposition" begin
    include("eig.jl")
end
@safetestset "Hermitian Eigenvalue Decomposition" begin
    include("eigh.jl")
end
@safetestset "Image and Null Space" begin
    include("orthnull.jl")
end

using CUDA
if CUDA.functional()
    @safetestset "CUDA SVD" begin
        include("cuda/svd.jl")
    end
end

using AMDGPU
if AMDGPU.functional()
    @safetestset "AMDGPU SVD" begin
        include("amd/svd.jl")
    end
end
