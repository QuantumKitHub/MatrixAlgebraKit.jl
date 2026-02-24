"""
    test_mooncake_eig(T, sz; kwargs...)

Run all Mooncake AD tests for eigendecompositions of element type `T` and size `sz`.
"""
function test_mooncake_eig(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "Mooncake eig $summary_str" begin
        test_mooncake_eig_full(T, sz; kwargs...)
        test_mooncake_eig_vals(T, sz; kwargs...)
        test_mooncake_eig_trunc(T, sz; kwargs...)
    end
end

"""
    test_mooncake_eig_full(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rule for `eig_full` and its in-place variant.
"""
function test_mooncake_eig_full(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "eig_full" begin
        A = make_eig_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(eig_full, A)
        DV, ΔDV = ad_eig_full_setup(A)
        output_tangent = Mooncake.primal_to_tangent!!(Mooncake.zero_tangent(DV), ΔDV)

        Mooncake.TestUtils.test_rule(
            rng, eig_full, A, alg;
            mode = Mooncake.ReverseMode, output_tangent, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, call_and_zero!, eig_full!, A, alg;
            mode = Mooncake.ReverseMode, output_tangent, atol, rtol, is_primitive = false
        )
    end
end

"""
    test_mooncake_eig_vals(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rule for `eig_vals` and its in-place variant.
"""
function test_mooncake_eig_vals(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "eig_vals" begin
        A = make_eig_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(eig_vals, A)
        D = eig_vals(A)
        output_tangent = Mooncake.randn_tangent(rng, D)

        Mooncake.TestUtils.test_rule(
            rng, eig_vals, A, alg;
            mode = Mooncake.ReverseMode, output_tangent, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, call_and_zero!, eig_vals!, A, alg;
            mode = Mooncake.ReverseMode, output_tangent, atol, rtol, is_primitive = false
        )
    end
end

"""
    test_mooncake_eig_trunc(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rules for `eig_trunc`, `eig_trunc_no_error`, and their
in-place variants, over a range of truncation ranks and a tolerance-based truncation.
"""
function test_mooncake_eig_trunc(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "eig_trunc" begin
        A = make_eig_matrix(T, sz)
        m = size(A, 1)

        alg = MatrixAlgebraKit.select_algorithm(eig_full, A)

        @testset "truncrank($r)" for r in round.(Int, range(1, m + 4, 4))
            trunc = truncrank(r; by = abs)
            alg_trunc = TruncatedAlgorithm(alg, trunc)

            DV, DVtrunc, ΔDV_arrays, ΔDVtrunc_arrays = ad_eig_trunc_setup(A, alg_trunc)
            ΔDVtrunc = Mooncake.primal_to_tangent!!(Mooncake.zero_tangent(DVtrunc), ΔDVtrunc_arrays)

            Mooncake.TestUtils.test_rule(
                rng, eig_trunc_no_error, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔDVtrunc, atol, rtol
            )
            Mooncake.TestUtils.test_rule(
                rng, call_and_zero!, eig_trunc_no_error!, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔDVtrunc, atol, rtol, is_primitive = false
            )

            DVϵ = eig_trunc(A, alg_trunc)
            Δϵ = Mooncake.zero_tangent(DVϵ[end])
            ΔDVϵtrunc = (ΔDVtrunc..., Δϵ)

            Mooncake.TestUtils.test_rule(
                rng, eig_trunc, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔDVϵtrunc, atol, rtol
            )
            Mooncake.TestUtils.test_rule(
                rng, call_and_zero!, eig_trunc!, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔDVϵtrunc, atol, rtol, is_primitive = false
            )
        end

        @testset "trunctol" begin
            D = eig_vals(A)
            trunc = trunctol(atol = maximum(abs, D) / 2; by = abs)
            alg_trunc = TruncatedAlgorithm(alg, trunc)

            DV, DVtrunc, ΔDV_arrays, ΔDVtrunc_arrays = ad_eig_trunc_setup(A, alg_trunc)
            ΔDVtrunc = Mooncake.primal_to_tangent!!(Mooncake.zero_tangent(DVtrunc), ΔDVtrunc_arrays)

            Mooncake.TestUtils.test_rule(
                rng, eig_trunc_no_error, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔDVtrunc, atol, rtol
            )
            Mooncake.TestUtils.test_rule(
                rng, call_and_zero!, eig_trunc_no_error!, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔDVtrunc, atol, rtol, is_primitive = false
            )

            DVϵ = eig_trunc(A, alg_trunc)
            Δϵ = Mooncake.zero_tangent(DVϵ[end])
            ΔDVϵtrunc = (ΔDVtrunc..., Δϵ)

            Mooncake.TestUtils.test_rule(
                rng, eig_trunc, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔDVϵtrunc, atol, rtol
            )
            Mooncake.TestUtils.test_rule(
                rng, call_and_zero!, eig_trunc!, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔDVϵtrunc, atol, rtol, is_primitive = false
            )
        end
    end
end
