"""
    remove_left_null_gauge_dependence!(ΔN, A, N)

Remove the gauge-dependent part from the cotangent `ΔN` of the left null space `N`. The null
space basis is only determined up to a unitary rotation, so `ΔN` is projected onto the column
span of the compact QR factor `Q₁` of `A`.
"""
function remove_left_null_gauge_dependence!(ΔN, A, N)
    Q, _ = qr_compact(A)
    mul!(ΔN, Q, Q' * ΔN)
    return ΔN
end

"""
    remove_right_null_gauge_dependence!(ΔNᴴ, A, Nᴴ)

Remove the gauge-dependent part from the cotangent `ΔNᴴ` of the right null space `Nᴴ`. The
null space basis is only determined up to a unitary rotation, so `ΔNᴴ` is projected onto the
row span of the compact LQ factor `Q₁` of `A`.
"""
function remove_right_null_gauge_dependence!(ΔNᴴ, A, Nᴴ)
    _, Q = lq_compact(A)
    mul!(ΔNᴴ, ΔNᴴ * Q', Q)
    return ΔNᴴ
end

"""
    test_mooncake_orthnull(T, sz; kwargs...)

Run all Mooncake AD tests for orthogonal basis and null space computations of element type `T`
and size `sz`.
"""
function test_mooncake_orthnull(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "Mooncake orthnull $summary_str" begin
        test_mooncake_left_orth(T, sz; kwargs...)
        test_mooncake_right_orth(T, sz; kwargs...)
        test_mooncake_left_null(T, sz; kwargs...)
        test_mooncake_right_null(T, sz; kwargs...)
    end
end

"""
    test_mooncake_left_orth(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rules for `left_orth` with QR and polar (when `m >= n`)
algorithms, and their in-place variants.
"""
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

"""
    test_mooncake_right_orth(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rules for `right_orth` with LQ and polar (when `m <= n`)
algorithms, and their in-place variants.
"""
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

"""
    test_mooncake_left_null(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rule for `left_null` with the QR algorithm and its
in-place variant.
"""
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

"""
    test_mooncake_right_null(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rule for `right_null` with the LQ algorithm and its
in-place variant.
"""
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
