function test_mooncake_qr(
        T::Type, sz;
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "Mooncake qr $summary_str" begin
        test_mooncake_qr_compact(T, sz; kwargs...)
        test_mooncake_qr_full(T, sz; kwargs...)
        test_mooncake_qr_null(T, sz; kwargs...)
    end
end

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

function remove_qr_null_gauge_dependence!(ΔN, A, N)
    Q, _ = qr_compact(A)
    return mul!(ΔN, Q, Q' * ΔN)
end

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
            rng, make_input_scratch!, qr_compact!, A, QR, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔQR, atol, rtol, is_primitive = false
        )
    end
end

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
            rng, make_input_scratch!, qr_full!, A, QR, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔQR, atol, rtol, is_primitive = false
        )
    end
end

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
        N, ΔN = ad_qr_null_setup(A)
        Mooncake.TestUtils.test_rule(
            rng, make_input_scratch!, qr_null!, A, N, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔN, atol, rtol, is_primitive = false
        )
    end
end
