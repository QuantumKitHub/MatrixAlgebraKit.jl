## using MatrixAlgebraKit
## using Test
## using TestExtras
## using ChainRulesTestUtils
## using StableRNGs
## using Aqua
## using JET
## using LinearAlgebra: LinearAlgebra, diag, Diagonal, I, isposdef, diagind, mul!
## using MatrixAlgebraKit: diagview

using SafeTestsets

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
@safetestset "ChainRules" verbose = true begin
    include("chainrules.jl")
end

@safetestset "MatrixAlgebraKit.jl" begin
    @safetestset "Code quality (Aqua.jl)" begin
        Aqua.test_all(MatrixAlgebraKit)
    end
    @safetestset "Code linting (JET.jl)" begin
        JET.test_package(MatrixAlgebraKit; target_defined_modules=true)
    end
end
