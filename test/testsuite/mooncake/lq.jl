"""
    test_mooncake_lq(T, sz; kwargs...)

Run all Mooncake AD tests for LQ decompositions of element type `T` and size `sz`.
"""
function test_mooncake_lq(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "Mooncake lq $summary_str" begin
        test_mooncake_lq_compact(T, sz; kwargs...)
        test_mooncake_lq_full(T, sz; kwargs...)
        test_mooncake_lq_null(T, sz; kwargs...)
    end
end

"""
    test_mooncake_lq_compact(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rule for `lq_compact` and its in-place variant.
"""
function test_mooncake_lq_compact(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "lq_compact" begin
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(lq_compact, A; positive = true)
        LQ = lq_compact(A, alg)
        ΔLQ = Mooncake.randn_tangent(rng, LQ)
        remove_lq_gauge_dependence!(ΔLQ[2], A, LQ...)

        Mooncake.TestUtils.test_rule(
            rng, lq_compact, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔLQ, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, call_and_zero!, lq_compact!, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔLQ, atol, rtol, is_primitive = false
        )
    end
end

"""
    test_mooncake_lq_full(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rule for `lq_full` and its in-place variant.
"""
function test_mooncake_lq_full(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "lq_full" begin
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(lq_full, A; positive = true)
        LQ = lq_full(A, alg)
        ΔLQ = Mooncake.randn_tangent(rng, LQ)
        remove_lq_gauge_dependence!(ΔLQ[2], A, LQ...)

        Mooncake.TestUtils.test_rule(
            rng, lq_full, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔLQ, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, call_and_zero!, lq_full!, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔLQ, atol, rtol, is_primitive = false
        )
    end
end

"""
    test_mooncake_lq_null(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rule for `lq_null` and its in-place variant.
"""
function test_mooncake_lq_null(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "lq_null" begin
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(lq_null, A; positive = true)
        Nᴴ = lq_null(A, alg)
        ΔNᴴ = Mooncake.randn_tangent(rng, Nᴴ)
        remove_lq_null_gauge_dependence!(ΔNᴴ, A, Nᴴ)

        Mooncake.TestUtils.test_rule(
            rng, lq_null, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔNᴴ, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, call_and_zero!, lq_null!, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔNᴴ, atol, rtol, is_primitive = false
        )
    end
end
