"""
    test_mooncake_polar(T, sz; kwargs...)

Run all Mooncake AD tests for polar decompositions of element type `T` and size `sz`.
"""
function test_mooncake_polar(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "Mooncake polar $summary_str" begin
        test_mooncake_left_polar(T, sz; kwargs...)
        test_mooncake_right_polar(T, sz; kwargs...)
    end
end

"""
    test_mooncake_left_polar(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rule for `left_polar` and its in-place variant. Only runs
for tall or square matrices (`m >= n`).
"""
function test_mooncake_left_polar(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "left_polar" begin
        A = instantiate_matrix(T, sz)
        m, n = size(A)
        if m >= n
            alg = MatrixAlgebraKit.select_algorithm(left_polar, A)
            WP = left_polar(A, alg)
            ΔWP = Mooncake.randn_tangent(rng, WP)

            Mooncake.TestUtils.test_rule(
                rng, left_polar, A, alg;
                mode = Mooncake.ReverseMode, output_tangent = ΔWP, atol, rtol
            )
            Mooncake.TestUtils.test_rule(
                rng, make_input_scratch!, left_polar!, A, alg;
                mode = Mooncake.ReverseMode, output_tangent = ΔWP, atol, rtol, is_primitive = false
            )
        end
    end
end

"""
    test_mooncake_right_polar(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rule for `right_polar` and its in-place variant. Only runs
for wide or square matrices (`m <= n`).
"""
function test_mooncake_right_polar(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "right_polar" begin
        A = instantiate_matrix(T, sz)
        m, n = size(A)
        if m <= n
            alg = MatrixAlgebraKit.select_algorithm(right_polar, A)
            PWᴴ = right_polar(A, alg)
            ΔPWᴴ = Mooncake.randn_tangent(rng, PWᴴ)

            Mooncake.TestUtils.test_rule(
                rng, right_polar, A, alg;
                mode = Mooncake.ReverseMode, output_tangent = ΔPWᴴ, atol, rtol
            )
            Mooncake.TestUtils.test_rule(
                rng, make_input_scratch!, right_polar!, A, alg;
                mode = Mooncake.ReverseMode, output_tangent = ΔPWᴴ, atol, rtol, is_primitive = false
            )
        end
    end
end
