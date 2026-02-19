"""
    remove_qr_gauge_dependence!(ΔQ, A, Q, R)

Remove the gauge-dependent part from the cotangent `ΔQ` of the full-QR orthogonal factor `Q`.
For the full QR decomposition, the extra columns of `Q` beyond `min(m, n)` are not uniquely
determined by `A`, so the corresponding part of `ΔQ` is projected to remove this ambiguity.
"""
function remove_qr_gauge_dependence!(ΔQ, A, Q, R)
    m, n = size(A)
    minmn = min(m, n)
    Q₁ = @view Q[:, 1:minmn]
    ΔQ₂ = @view ΔQ[:, (minmn + 1):end]
    Q₁ᴴΔQ₂ = Q₁' * ΔQ₂
    mul!(ΔQ₂, Q₁, Q₁ᴴΔQ₂)
    MatrixAlgebraKit.check_qr_full_cotangents(Q₁, ΔQ₂, Q₁ᴴΔQ₂)
    return ΔQ
end

"""
    remove_qr_null_gauge_dependence!(ΔN, A, N)

Remove the gauge-dependent part from the cotangent `ΔN` of the QR null space `N`. The null
space is only determined up to a unitary rotation, so `ΔN` is projected onto the column span
of the compact QR factor `Q₁`.
"""
function remove_qr_null_gauge_dependence!(ΔN, A, N)
    Q, _ = qr_compact(A)
    return mul!(ΔN, Q, Q' * ΔN)
end

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
        QR = qr_compact(A, alg)
        ΔQR = Mooncake.randn_tangent(rng, QR)
        remove_qr_gauge_dependence!(ΔQR[1], A, QR...)

        Mooncake.TestUtils.test_rule(
            rng, qr_compact, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔQR, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, call_and_zero!, qr_compact!, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔQR, atol, rtol, is_primitive = false
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
        QR = qr_full(A, alg)
        ΔQR = Mooncake.randn_tangent(rng, QR)
        remove_qr_gauge_dependence!(ΔQR[1], A, QR...)

        Mooncake.TestUtils.test_rule(
            rng, qr_full, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔQR, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, call_and_zero!, qr_full!, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔQR, atol, rtol, is_primitive = false
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
        N = qr_null(A, alg)
        ΔN = Mooncake.randn_tangent(rng, N)
        remove_qr_null_gauge_dependence!(ΔN, A, N)

        Mooncake.TestUtils.test_rule(
            rng, qr_null, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔN, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, call_and_zero!, qr_null!, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔN, atol, rtol, is_primitive = false
        )
    end
end
