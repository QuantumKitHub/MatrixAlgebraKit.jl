"""
    test_mooncake_qr(T, sz; kwargs...)

Run all Mooncake AD tests for QR decompositions of element type `T` and size `sz`.
"""
function test_mooncake_qr(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "Mooncake qr $summary_str" begin
        test_mooncake_qr_compact(T, sz; kwargs...)
        test_mooncake_qr_full(T, sz; kwargs...)
        test_mooncake_qr_null(T, sz; kwargs...)
    end
end

"""
    test_mooncake_qr_compact(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rule for `qr_compact` and its in-place variant.
"""
function test_mooncake_qr_compact(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "qr_compact" begin
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(qr_compact, A; positive = true)
        QR, ΔQR = ad_qr_compact_setup(A)
        output_tangent = Mooncake.primal_to_tangent!!(Mooncake.zero_tangent(QR), ΔQR)

        Mooncake.TestUtils.test_rule(
            rng, qr_compact, A, alg;
            mode = Mooncake.ReverseMode, output_tangent, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, call_and_zero!, qr_compact!, A, alg;
            mode = Mooncake.ReverseMode, output_tangent, atol, rtol, is_primitive = false
        )

        A = instantiate_rank_deficient_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(qr_compact, A; positive = true)
        QR, ΔQR = ad_qr_compact_setup(A)
        output_tangent = Mooncake.primal_to_tangent!!(Mooncake.zero_tangent(QR), ΔQR)

        Mooncake.TestUtils.test_rule(
            rng, qr_compact, A, alg;
            mode = Mooncake.ReverseMode, output_tangent, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, call_and_zero!, qr_compact!, A, alg;
            mode = Mooncake.ReverseMode, output_tangent, atol, rtol, is_primitive = false
        )
    end
end

"""
    test_mooncake_qr_full(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rule for `qr_full` and its in-place variant.
"""
function test_mooncake_qr_full(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "qr_full" begin
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(qr_full, A; positive = true)
        QR, ΔQR = ad_qr_full_setup(A)
        output_tangent = Mooncake.primal_to_tangent!!(Mooncake.zero_tangent(QR), ΔQR)

        Mooncake.TestUtils.test_rule(
            rng, qr_full, A, alg;
            mode = Mooncake.ReverseMode, output_tangent, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, call_and_zero!, qr_full!, A, alg;
            mode = Mooncake.ReverseMode, output_tangent, atol, rtol, is_primitive = false
        )
    end
end

"""
    test_mooncake_qr_null(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rule for `qr_null` and its in-place variant.
"""
function test_mooncake_qr_null(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "qr_null" begin
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(qr_null, A; positive = true)
        N, ΔN = ad_qr_null_setup(A)
        output_tangent = Mooncake.primal_to_tangent!!(Mooncake.zero_tangent(N), ΔN)

        Mooncake.TestUtils.test_rule(
            rng, qr_null, A, alg;
            mode = Mooncake.ReverseMode, output_tangent, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, call_and_zero!, qr_null!, A, alg;
            mode = Mooncake.ReverseMode, output_tangent, atol, rtol, is_primitive = false
        )
    end
end
