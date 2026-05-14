"""
    test_enzyme_polar(T, sz; kwargs...)

Run all Enzyme AD tests for polar decompositions of element type `T` and size `sz`.
"""
function test_enzyme_polar(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "Enzyme polar $summary_str" begin
        test_enzyme_left_polar(T, sz; kwargs...)
        test_enzyme_right_polar(T, sz; kwargs...)
    end
end

"""
    test_enzyme_left_polar(T, sz; rng, atol, rtol)

Test the Enzyme reverse-mode AD rule for `left_polar` and its in-place variant. Only runs
for tall or square matrices (`m >= n`).
"""
function test_enzyme_left_polar(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "left_polar reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        m, n = size(A)
        if m >= n
            alg = MatrixAlgebraKit.select_algorithm(left_polar, A)
            WP, ΔWP = ad_left_polar_setup(A)
            test_reverse(left_polar, RT, (A, TA), (alg, Const); atol, rtol)
            test_reverse(call_and_zero!, RT, (left_polar!, Const), (A, TA), (alg, Const); atol, rtol)
        end
    end
end

"""
    test_enzyme_right_polar(T, sz; rng, atol, rtol)

Test the Enzyme reverse-mode AD rule for `right_polar` and its in-place variant. Only runs
for wide or square matrices (`m <= n`).
"""
function test_enzyme_right_polar(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "right_polar reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        m, n = size(A)
        if m <= n
            alg = MatrixAlgebraKit.select_algorithm(right_polar, A)
            PWᴴ, ΔPWᴴ = ad_right_polar_setup(A)
            test_reverse(right_polar, RT, (A, TA), (alg, Const); atol, rtol)
            test_reverse(call_and_zero!, RT, (right_polar!, Const), (A, TA), (alg, Const); atol, rtol)
        end
    end
end
