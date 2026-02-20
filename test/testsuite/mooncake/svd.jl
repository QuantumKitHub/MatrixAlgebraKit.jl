"""
    test_mooncake_svd(T, sz; kwargs...)

Run all Mooncake AD tests for SVD decompositions of element type `T` and size `sz`.
"""
function test_mooncake_svd(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "Mooncake svd $summary_str" begin
        test_mooncake_svd_compact(T, sz; kwargs...)
        test_mooncake_svd_full(T, sz; kwargs...)
        test_mooncake_svd_vals(T, sz; kwargs...)
        test_mooncake_svd_trunc(T, sz; kwargs...)
    end
end

"""
    test_mooncake_svd_compact(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rule for `svd_compact` and its in-place variant.
"""
function test_mooncake_svd_compact(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "svd_compact" begin
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(svd_compact, A)
        USVᴴ = svd_compact(A, alg)
        ΔUSVᴴ = Mooncake.randn_tangent(rng, USVᴴ)
        remove_svd_gauge_dependence!(ΔUSVᴴ[1], ΔUSVᴴ[3], USVᴴ...)

        Mooncake.TestUtils.test_rule(
            rng, svd_compact, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔUSVᴴ, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, call_and_zero!, svd_compact!, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔUSVᴴ, atol, rtol, is_primitive = false
        )
    end
end

"""
    test_mooncake_svd_full(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rule for `svd_full` and its in-place variant. The
gauge-dependent extra columns of `U` and rows of `Vᴴ` are zeroed out in the cotangent.
"""
function test_mooncake_svd_full(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "svd_full" begin
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(svd_full, A)
        USVᴴ = svd_full(A, alg)
        ΔUSVᴴ = Mooncake.randn_tangent(rng, USVᴴ)
        remove_svd_gauge_dependence!(ΔUSVᴴ[1], ΔUSVᴴ[3], USVᴴ...)

        Mooncake.TestUtils.test_rule(
            rng, svd_full, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔUSVᴴ, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, call_and_zero!, svd_full!, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔUSVᴴ, atol, rtol, is_primitive = false
        )
    end
end

"""
    test_mooncake_svd_vals(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rule for `svd_vals` and its in-place variant.
"""
function test_mooncake_svd_vals(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "svd_vals" begin
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(svd_vals, A)
        S = svd_vals(A, alg)
        ΔS = Mooncake.randn_tangent(rng, S)

        Mooncake.TestUtils.test_rule(
            rng, svd_vals, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔS, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, call_and_zero!, svd_vals!, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔS, atol, rtol, is_primitive = false
        )
    end
end

"""
    test_mooncake_svd_trunc(T, sz; rng, atol, rtol)

Test the Mooncake reverse-mode AD rules for `svd_trunc`, `svd_trunc_no_error`, and their
in-place variants, over a range of truncation ranks and a tolerance-based truncation.
"""
function test_mooncake_svd_trunc(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "svd_trunc" begin
        A = instantiate_matrix(T, sz)
        m, n = size(A)
        minmn = min(m, n)

        alg = MatrixAlgebraKit.select_algorithm(svd_compact, A)
        USVᴴ = svd_compact(A, alg)
        ΔUSVᴴ = Mooncake.randn_tangent(rng, USVᴴ)
        remove_svd_gauge_dependence!(ΔUSVᴴ[1], ΔUSVᴴ[3], USVᴴ...)

        @testset "truncrank($r)" for r in round.(Int, range(1, minmn + 4, 4))
            trunc = truncrank(r)
            alg_trunc = TruncatedAlgorithm(alg, trunc)

            # truncate the gauge-corrected tangents
            USVᴴtrunc, ind = MatrixAlgebraKit.truncate(svd_trunc!, USVᴴ, trunc)
            ΔUSVᴴ_primal = Mooncake.tangent_to_primal!!(copy.(USVᴴ), ΔUSVᴴ)
            ΔUSVᴴtrunc_primal = (ΔUSVᴴ_primal[1][:, ind], Diagonal(diagview(ΔUSVᴴ_primal[2])[ind]), ΔUSVᴴ_primal[3][ind, :])
            ΔUSVᴴtrunc = Mooncake.primal_to_tangent!!(Mooncake.zero_tangent(USVᴴtrunc), ΔUSVᴴtrunc_primal)

            Mooncake.TestUtils.test_rule(
                rng, svd_trunc_no_error, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔUSVᴴtrunc, atol, rtol
            )
            Mooncake.TestUtils.test_rule(
                rng, call_and_zero!, svd_trunc_no_error!, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔUSVᴴtrunc, atol, rtol, is_primitive = false
            )

            USVᴴϵ = svd_trunc(A, alg_trunc)
            Δϵ = Mooncake.zero_tangent(USVᴴϵ[end])
            ΔUSVᴴϵtrunc = (ΔUSVᴴtrunc..., Δϵ)

            Mooncake.TestUtils.test_rule(
                rng, svd_trunc, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔUSVᴴϵtrunc, atol, rtol
            )
            Mooncake.TestUtils.test_rule(
                rng, call_and_zero!, svd_trunc!, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔUSVᴴϵtrunc, atol, rtol, is_primitive = false
            )
        end

        @testset "trunctol" begin
            trunc = trunctol(atol = diagview(USVᴴ[2])[1] / 2)
            alg_trunc = TruncatedAlgorithm(alg, trunc)

            USVᴴtrunc, ind = MatrixAlgebraKit.truncate(svd_trunc!, USVᴴ, trunc)
            ΔUSVᴴ_primal = Mooncake.tangent_to_primal!!(copy.(USVᴴ), ΔUSVᴴ)
            ΔUSVᴴtrunc_primal = (ΔUSVᴴ_primal[1][:, ind], Diagonal(diagview(ΔUSVᴴ_primal[2])[ind]), ΔUSVᴴ_primal[3][ind, :])
            ΔUSVᴴtrunc = Mooncake.primal_to_tangent!!(Mooncake.zero_tangent(USVᴴtrunc), ΔUSVᴴtrunc_primal)

            Mooncake.TestUtils.test_rule(
                rng, svd_trunc_no_error, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔUSVᴴtrunc, atol, rtol
            )
            Mooncake.TestUtils.test_rule(
                rng, call_and_zero!, svd_trunc_no_error!, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔUSVᴴtrunc, atol, rtol, is_primitive = false
            )

            USVᴴϵ = svd_trunc(A, alg_trunc)
            Δϵ = Mooncake.zero_tangent(USVᴴϵ[end])
            ΔUSVᴴϵtrunc = (ΔUSVᴴtrunc..., Δϵ)

            Mooncake.TestUtils.test_rule(
                rng, svd_trunc, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔUSVᴴϵtrunc, atol, rtol
            )
            Mooncake.TestUtils.test_rule(
                rng, call_and_zero!, svd_trunc!, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔUSVᴴϵtrunc, atol, rtol, is_primitive = false
            )
        end
    end
end
