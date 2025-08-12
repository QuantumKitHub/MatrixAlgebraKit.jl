using SafeTestsets

# don't run all tests on GPU, only the GPU
# specific ones
is_buildkite = get(ENV, "BUILDKITE", "false") == "true"
if !is_buildkite
    @safetestset "Algorithms" begin
        include("algorithms.jl")
    end
    @safetestset "Truncate" begin
        include("truncate.jl")
    end
    @safetestset "QR / LQ Decomposition" begin
        include("qr.jl")
        include("lq.jl")
    end
    @safetestset "Singular Value Decomposition" begin
        include("svd.jl")
    end
    @safetestset "Hermitian Eigenvalue Decomposition" begin
        include("eigh.jl")
    end
    @safetestset "General Eigenvalue Decomposition" begin
        include("eig.jl")
    end
    @safetestset "Schur Decomposition" begin
        include("schur.jl")
    end
    @safetestset "Polar Decomposition" begin
        include("polar.jl")
    end
    @safetestset "Image and Null Space" begin
        include("orthnull.jl")
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
            JET.test_package(MatrixAlgebraKit; target_defined_modules=true)
        end
    end
end

using CUDA
if CUDA.functional()
    @safetestset "CUDA QR" begin
        include("cuda/qr.jl")
    end
    @safetestset "CUDA LQ" begin
        include("cuda/lq.jl")
    end
    @safetestset "CUDA SVD" begin
        include("cuda/svd.jl")
    end
end

