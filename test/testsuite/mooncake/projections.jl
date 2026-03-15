"""
    test_mooncake_projections(T, sz; kwargs...)

Run all Mooncake AD tests for hermitian and anti-hermitian projections of element type `T`
and size `sz`.
"""
function test_mooncake_projections(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "Mooncake projection $summary_str" begin
        test_mooncake_project_hermitian(T, sz; kwargs...)
        test_mooncake_project_antihermitian(T, sz; kwargs...)
    end
end

"""
    test_mooncake_project_hermitian(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rule for `project_hermitian` and its in-place variant.
"""
function test_mooncake_project_hermitian(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "project_hermitian" begin
        A = instantiate_matrix(T, sz)
        B = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(project_hermitian, A)
        Mooncake.TestUtils.test_rule(
            rng, project_hermitian, A, alg;
            mode = Mooncake.ReverseMode, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, project_hermitian!, A, A, alg;
            mode = Mooncake.ReverseMode, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, project_hermitian!, A, B, alg;
            mode = Mooncake.ReverseMode, atol, rtol
        )
    end
end

"""
    test_mooncake_project_antihermitian(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rule for `project_antihermitian` and its in-place variant.
"""
function test_mooncake_project_antihermitian(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "project_antihermitian" begin
        A = instantiate_matrix(T, sz)
        B = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(project_hermitian, A)
        Mooncake.TestUtils.test_rule(
            rng, project_antihermitian, A, alg;
            mode = Mooncake.ReverseMode, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, project_antihermitian!, A, A, alg;
            mode = Mooncake.ReverseMode, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, project_antihermitian!, A, B, alg;
            mode = Mooncake.ReverseMode, atol, rtol
        )
    end
end
