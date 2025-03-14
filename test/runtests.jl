using MatrixAlgebraKit
using Test
using TestExtras
using ChainRulesTestUtils
using StableRNGs
using Aqua
using JET
using LinearAlgebra: LinearAlgebra, diag, Diagonal, I, isposdef, diagind, mul!
using MatrixAlgebraKit: diagview

@testset "QR / LQ Decomposition" begin
    include("qr.jl")
    include("lq.jl")
end
@testset "Singular Value Decomposition" begin
    include("svd.jl")
end
@testset "Hermitian Eigenvalue Decomposition" begin
    include("eigh.jl")
end
@testset "General Eigenvalue Decomposition" begin
    include("eig.jl")
end
@testset "Schur Decomposition" begin
    include("schur.jl")
end
@testset "Polar Decomposition" begin
    include("polar.jl")
end
@testset "Image and Null Space" begin
    include("orthnull.jl")
end
@testset "ChainRules" verbose = true begin
    include("chainrules.jl")
end

@testset "MatrixAlgebraKit.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(MatrixAlgebraKit)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(MatrixAlgebraKit; target_defined_modules=true)
    end
end
