using SafeTestsets

const pat = r"(?:--group=)(\w+)"
arg_id = findfirst(contains(pat), ARGS)
const GROUP = if isnothing(arg_id)
    uppercase(get(ENV, "GROUP", "ALL"))
else
    uppercase(only(match(pat, ARGS[arg_id]).captures))
end

@time begin
    if GROUP == "ALL" || GROUP == "DECOMPOSITIONS"
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
    end

    if GROUP == "ALL" || GROUP == "ChainRules"
        @safetestset "ChainRules" begin
            include("chainrules.jl")
        end
    end

    if GROUP == "ALL" || GROUP == "CUDA"
        @safetestset "CUDA" begin
            include("cuda.jl")
        end
    end

    if GROUP == "ALL" || GROUP == "UTILITY"
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
