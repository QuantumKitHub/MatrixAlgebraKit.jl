using TestExtras
using LinearAlgebra: isposdef

function test_polar(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "polar $summary_str" begin
        (length(sz) == 1 || sz[1] ≥ sz[2]) && test_left_polar(T, sz; kwargs...)
        (length(sz) == 1 || sz[2] ≥ sz[1]) && test_right_polar(T, sz; kwargs...)
    end
end

function test_left_polar(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "left_polar! $summary_str" begin
        A = instantiate_matrix(T, sz)
        algs = if T <: Diagonal
            (PolarNewton(),)
        elseif T <: Number
            (PolarViaSVD(MatrixAlgebraKit.default_svd_algorithm(A)), PolarNewton())
        else
            (PolarViaSVD(MatrixAlgebraKit.default_svd_algorithm(A)),)
        end
        @testset "algorithm $alg" for alg in algs
            A = instantiate_matrix(T, sz)
            Ac = deepcopy(A)
            W, P = left_polar(A; alg)
            @test eltype(W) == eltype(A) && size(W) == (size(A, 1), size(A, 2))
            @test eltype(P) == eltype(A) && size(P) == (size(A, 2), size(A, 2))
            @test W * P ≈ A
            @test isisometric(W)
            @test ishermitian(P; rtol = MatrixAlgebraKit.defaulttol(P))
            @test isposdef(project_hermitian!(P))

            W2, P2 = @testinferred left_polar!(Ac, (W, P), alg)
            @test W2 === W
            @test P2 === P
            @test W * P ≈ A
            @test isisometric(W)
            @test ishermitian(P; rtol = MatrixAlgebraKit.defaulttol(P))
            @test isposdef(project_hermitian!(P))

            noP = similar(P, (0, 0))
            W2, P2 = @testinferred left_polar!(copy!(Ac, A), (W, noP), alg)
            @test P2 === noP
            @test W2 === W
            @test isisometric(W)
            P = W' * A # compute P explicitly to verify W correctness
            @test ishermitian(P; rtol = MatrixAlgebraKit.defaulttol(P))
            @test isposdef(project_hermitian!(P))
        end
    end
end

function test_right_polar(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "right_polar! $summary_str" begin
        A = instantiate_matrix(T, sz)
        algs = if T <: Diagonal
            (PolarNewton(),)
        elseif T <: Number
            (PolarViaSVD(MatrixAlgebraKit.default_svd_algorithm(A)), PolarNewton())
        else
            (PolarViaSVD(MatrixAlgebraKit.default_svd_algorithm(A)),)
        end
        @testset "algorithm $alg" for alg in algs
            A = instantiate_matrix(T, sz)
            Ac = deepcopy(A)
            P, Wᴴ = right_polar(A; alg)
            @test eltype(Wᴴ) == eltype(A) && size(Wᴴ) == (size(A, 1), size(A, 2))
            @test eltype(P) == eltype(A) && size(P) == (size(A, 1), size(A, 1))
            @test P * Wᴴ ≈ A
            @test isisometric(Wᴴ; side = :right)
            @test ishermitian(P; rtol = MatrixAlgebraKit.defaulttol(P))
            @test isposdef(project_hermitian!(P))

            P2, Wᴴ2 = @testinferred right_polar!(Ac, (P, Wᴴ), alg)
            @test P2 === P
            @test Wᴴ2 === Wᴴ
            @test P * Wᴴ ≈ A
            @test isisometric(Wᴴ; side = :right)
            @test ishermitian(P; rtol = MatrixAlgebraKit.defaulttol(P))
            @test isposdef(project_hermitian!(P))

            noP = similar(P, (0, 0))
            P2, Wᴴ2 = @testinferred right_polar!(copy!(Ac, A), (noP, Wᴴ), alg)
            @test P2 === noP
            @test Wᴴ2 === Wᴴ
            @test isisometric(Wᴴ; side = :right)
            P = A * Wᴴ' # compute P explicitly to verify W correctness
            @test ishermitian(P; rtol = MatrixAlgebraKit.defaulttol(P))
            @test isposdef(project_hermitian!(P))
        end
    end
end
