"""
    test_enzyme_projections(T, sz; kwargs...)

Run all Enzyme AD tests for hermitian and anti-hermitian projections of element type `T`
and size `sz`.
"""
function test_enzyme_projections(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "Enzyme projection $summary_str" begin
        test_enzyme_project_hermitian(T, sz; kwargs...)
        test_enzyme_project_antihermitian(T, sz; kwargs...)
    end
end

"""
    test_enzyme_project_hermitian(T, sz; rng, atol, rtol)

Test the Enzyme reverse-mode AD rule for `project_hermitian` and its in-place variant.
"""
function test_enzyme_project_hermitian(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = enzyme_fdm(T)
    )
    return @testset "project_hermitian reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        B = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(project_hermitian, A)
        test_reverse(project_hermitian, RT, (A, TA), (alg, Const); atol, rtol, fdm)
        #test_reverse(project_hermitian!, RT, (A, TA), (A, TA), (alg, Const); atol, rtol, fdm)
        test_reverse(project_hermitian!, RT, (A, TA), (B, TA), (alg, Const); atol, rtol, fdm)
    end
end

"""
    test_enzyme_project_antihermitian(T, sz; rng, atol, rtol)

Test the Enzyme reverse-mode AD rule for `project_antihermitian` and its in-place variant.
"""
function test_enzyme_project_antihermitian(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = enzyme_fdm(T)
    )
    return @testset "project_antihermitian reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        B = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(project_hermitian, A)
        test_reverse(project_antihermitian, RT, (A, TA), (alg, Const); atol, rtol, fdm)
        #test_reverse(project_antihermitian!, RT, (A, TA), (A, TA), (alg, Const); atol, rtol, fdm)
        test_reverse(project_antihermitian!, RT, (A, TA), (B, TA), (alg, Const); atol, rtol, fdm)
    end
end
