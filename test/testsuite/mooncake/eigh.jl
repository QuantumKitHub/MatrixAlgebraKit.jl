"""
    test_mooncake_eigh(T, sz; kwargs...)

Run all Mooncake AD tests for Hermitian eigendecompositions of element type `T` and size `sz`.
"""
function test_mooncake_eigh(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "Mooncake eigh $summary_str" begin
        test_mooncake_eigh_full(T, sz; kwargs...)
        test_mooncake_eigh_vals(T, sz; kwargs...)
        test_mooncake_eigh_trunc(T, sz; kwargs...)
    end
end

"""
    test_mooncake_eigh_full(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rule for `eigh_full` and its in-place variant.
"""
function test_mooncake_eigh_full(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "eigh_full" begin
        A = make_eigh_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(eigh_full, A)
        DV, ΔDV = ad_eigh_full_setup(A)
        output_tangent = Mooncake.primal_to_tangent!!(Mooncake.zero_tangent(DV), ΔDV)

        Mooncake.TestUtils.test_rule(
            rng, eigh_wrapper, eigh_full, A, alg;
            mode = Mooncake.ReverseMode, output_tangent, is_primitive = false, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, eigh!_wrapper, eigh_full!, A, alg;
            mode = Mooncake.ReverseMode, output_tangent, is_primitive = false, atol, rtol
        )
    end
end

"""
    test_mooncake_eigh_vals(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rule for `eigh_vals` and its in-place variant.
"""
function test_mooncake_eigh_vals(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "eigh_vals" begin
        A = make_eigh_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(eigh_vals, A)
        D = eigh_vals(A, alg)
        output_tangent = Mooncake.randn_tangent(rng, D)

        Mooncake.TestUtils.test_rule(
            rng, eigh_wrapper, eigh_vals, A, alg;
            mode = Mooncake.ReverseMode, output_tangent, is_primitive = false, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, eigh!_wrapper, eigh_vals!, A, alg;
            mode = Mooncake.ReverseMode, output_tangent, is_primitive = false, atol, rtol
        )
    end
end

"""
    test_mooncake_eigh_trunc(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rules for `eigh_trunc`, `eigh_trunc_no_error`, and their
in-place variants, over a range of truncation ranks.
"""
function test_mooncake_eigh_trunc(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "eigh_trunc" begin
        A = make_eigh_matrix(T, sz)
        m = size(A, 1)

        alg = MatrixAlgebraKit.select_algorithm(eigh_full, A)
        DV, ΔDV_arrays = ad_eigh_full_setup(A)
        ΔDV = Mooncake.primal_to_tangent!!(Mooncake.zero_tangent(DV), ΔDV_arrays)

        @testset "truncrank($r)" for r in round.(Int, range(1, m + 4, 4))
            trunc = truncrank(r; by = abs)
            alg_trunc = TruncatedAlgorithm(alg, trunc)

            # truncate the gauge-corrected tangents
            DVtrunc, ind = MatrixAlgebraKit.truncate(eigh_trunc!, DV, trunc)
            ΔDVtrunc_arrays = (Diagonal(diagview(ΔDV_arrays[1])[ind]), ΔDV_arrays[2][:, ind])
            ΔDVtrunc = Mooncake.primal_to_tangent!!(Mooncake.zero_tangent(DVtrunc), ΔDVtrunc_arrays)

            Mooncake.TestUtils.test_rule(
                rng, eigh_wrapper, eigh_trunc_no_error, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔDVtrunc, is_primitive = false, atol, rtol
            )
            Mooncake.TestUtils.test_rule(
                rng, eigh!_wrapper, eigh_trunc_no_error!, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔDVtrunc, is_primitive = false, atol, rtol
            )

            DVϵ = eigh_trunc(A, alg_trunc)
            Δϵ = Mooncake.zero_tangent(DVϵ[end])
            ΔDVϵtrunc = (ΔDVtrunc..., Δϵ)

            Mooncake.TestUtils.test_rule(
                rng, eigh_wrapper, eigh_trunc, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔDVϵtrunc, is_primitive = false, atol, rtol
            )
            Mooncake.TestUtils.test_rule(
                rng, eigh!_wrapper, eigh_trunc!, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔDVϵtrunc, is_primitive = false, atol, rtol
            )
        end
    end
end
