function remove_left_null_gauge_dependence!(ΔN, A, N)
    Q, _ = qr_compact(A)
    mul!(ΔN, Q, Q' * ΔN)
    return ΔN
end

function remove_right_null_gauge_dependence!(ΔNᴴ, A, Nᴴ)
    _, Q = lq_compact(A)
    mul!(ΔNᴴ, ΔNᴴ * Q', Q)
    return ΔNᴴ
end

function test_mooncake_orthnull(
        T::Type, sz;
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "Mooncake orthnull $summary_str" begin
        test_mooncake_left_orth(T, sz; kwargs...)
        test_mooncake_right_orth(T, sz; kwargs...)
        test_mooncake_left_null(T, sz; kwargs...)
        test_mooncake_right_null(T, sz; kwargs...)
    end
end

function test_mooncake_left_orth(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "left_orth" begin
        A = instantiate_matrix(T, sz)
        m, n = size(A)

        @testset "qr" begin
            alg = MatrixAlgebraKit.select_algorithm(left_orth!, A, :qr)
            VC = left_orth(A, alg)
            ΔVC = Mooncake.randn_tangent(rng, VC)

            Mooncake.TestUtils.test_rule(
                rng, left_orth, A, alg;
                mode = Mooncake.ReverseMode, output_tangent = ΔVC, is_primitive = false, atol, rtol
            )
            Mooncake.TestUtils.test_rule(
                rng, make_input_scratch!, left_orth!, A, alg;
                mode = Mooncake.ReverseMode, output_tangent = ΔVC, is_primitive = false, atol, rtol
            )
        end

        if m >= n
            @testset "polar" begin
                alg = MatrixAlgebraKit.select_algorithm(left_orth!, A, :polar)
                VC = left_orth(A, alg)
                ΔVC = Mooncake.randn_tangent(rng, VC)

                Mooncake.TestUtils.test_rule(
                    rng, left_orth, A, alg;
                    mode = Mooncake.ReverseMode, output_tangent = ΔVC, is_primitive = false, atol, rtol
                )
                Mooncake.TestUtils.test_rule(
                    rng, make_input_scratch!, left_orth!, A, alg;
                    mode = Mooncake.ReverseMode, output_tangent = ΔVC, is_primitive = false, atol, rtol
                )
            end
        end
    end
end

function test_mooncake_right_orth(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "right_orth" begin
        A = instantiate_matrix(T, sz)
        m, n = size(A)

        @testset "lq" begin
            alg = MatrixAlgebraKit.select_algorithm(right_orth!, A, :lq)
            CVᴴ = right_orth(A, alg)
            ΔCVᴴ = Mooncake.randn_tangent(rng, CVᴴ)

            Mooncake.TestUtils.test_rule(
                rng, right_orth, A, alg;
                mode = Mooncake.ReverseMode, output_tangent = ΔCVᴴ, is_primitive = false, atol, rtol
            )
            Mooncake.TestUtils.test_rule(
                rng, make_input_scratch!, right_orth!, A, alg;
                mode = Mooncake.ReverseMode, output_tangent = ΔCVᴴ, is_primitive = false, atol, rtol
            )
        end

        if m <= n
            @testset "polar" begin
                alg = MatrixAlgebraKit.select_algorithm(right_orth!, A, :polar)
                CVᴴ = right_orth(A, alg)
                ΔCVᴴ = Mooncake.randn_tangent(rng, CVᴴ)

                Mooncake.TestUtils.test_rule(
                    rng, right_orth, A, alg;
                    mode = Mooncake.ReverseMode, output_tangent = ΔCVᴴ, is_primitive = false, atol, rtol
                )
                Mooncake.TestUtils.test_rule(
                    rng, make_input_scratch!, right_orth!, A, alg;
                    mode = Mooncake.ReverseMode, output_tangent = ΔCVᴴ, is_primitive = false, atol, rtol
                )
            end
        end
    end
end

function test_mooncake_left_null(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "left_null" begin
        A = instantiate_matrix(T, sz)

        @testset "qr" begin
            alg = MatrixAlgebraKit.select_algorithm(left_null!, A, :qr)
            N = left_null(A, alg)
            ΔN = Mooncake.randn_tangent(rng, N)
            remove_left_null_gauge_dependence!(ΔN, A, N)

            Mooncake.TestUtils.test_rule(
                rng, left_null, A, alg;
                mode = Mooncake.ReverseMode, output_tangent = ΔN, is_primitive = false, atol, rtol
            )
            Mooncake.TestUtils.test_rule(
                rng, make_input_scratch!, left_null!, A, alg;
                mode = Mooncake.ReverseMode, output_tangent = ΔN, is_primitive = false, atol, rtol
            )
        end
    end
end

function test_mooncake_right_null(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "right_null" begin
        A = instantiate_matrix(T, sz)

        @testset "lq" begin
            alg = MatrixAlgebraKit.select_algorithm(right_null!, A, :lq)
            Nᴴ = right_null(A, alg)
            ΔNᴴ = Mooncake.randn_tangent(rng, Nᴴ)
            remove_right_null_gauge_dependence!(ΔNᴴ, A, Nᴴ)

            Mooncake.TestUtils.test_rule(
                rng, right_null, A, alg;
                mode = Mooncake.ReverseMode, output_tangent = ΔNᴴ, is_primitive = false, atol, rtol
            )
            Mooncake.TestUtils.test_rule(
                rng, make_input_scratch!, right_null!, A, alg;
                mode = Mooncake.ReverseMode, output_tangent = ΔNᴴ, is_primitive = false, atol, rtol
            )
        end
    end
end
