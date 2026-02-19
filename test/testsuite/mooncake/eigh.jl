function mc_copy_eigh_full(A, alg)
    A = (A + A') / 2
    return eigh_full(A, alg)
end

function mc_copy_eigh_full!(A, DV, alg)
    A = (A + A') / 2
    return eigh_full!(A, DV, alg)
end

function mc_copy_eigh_vals(A, alg)
    A = (A + A') / 2
    return eigh_vals(A, alg)
end

function mc_copy_eigh_vals!(A, D, alg)
    A = (A + A') / 2
    return eigh_vals!(A, D, alg)
end

function mc_copy_eigh_trunc(A, alg)
    A = (A + A') / 2
    return eigh_trunc(A, alg)
end

function mc_copy_eigh_trunc!(A, DV, alg)
    A = (A + A') / 2
    return eigh_trunc!(A, DV, alg)
end

function mc_copy_eigh_trunc_no_error(A, alg)
    A = (A + A') / 2
    return eigh_trunc_no_error(A, alg)
end

function mc_copy_eigh_trunc_no_error!(A, DV, alg)
    A = (A + A') / 2
    return eigh_trunc_no_error!(A, DV, alg)
end

MatrixAlgebraKit.copy_input(::typeof(mc_copy_eigh_full), A) = MatrixAlgebraKit.copy_input(eigh_full, A)
MatrixAlgebraKit.copy_input(::typeof(mc_copy_eigh_vals), A) = MatrixAlgebraKit.copy_input(eigh_vals, A)
MatrixAlgebraKit.copy_input(::typeof(mc_copy_eigh_trunc), A) = MatrixAlgebraKit.copy_input(eigh_trunc, A)
MatrixAlgebraKit.copy_input(::typeof(mc_copy_eigh_trunc_no_error), A) = MatrixAlgebraKit.copy_input(eigh_trunc, A)

function remove_eigh_gauge_dependence!(
        ΔV, D, V;
        degeneracy_atol = MatrixAlgebraKit.default_pullback_gauge_atol(D)
    )
    gaugepart = V' * ΔV
    gaugepart = project_antihermitian!(gaugepart)
    gaugepart[abs.(transpose(diagview(D)) .- diagview(D)) .>= degeneracy_atol] .= 0
    mul!(ΔV, V, gaugepart, -1, 1)
    return ΔV
end

eigh_wrapper(f, A, alg) = f(project_hermitian(A), alg)
eigh!_wrapper(f!, A, alg) = (F = f!(project_hermitian!(A), alg); MatrixAlgebraKit.zero!(A); F)

function test_mooncake_eigh(
        T::Type, sz;
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "Mooncake eigh $summary_str" begin
        test_mooncake_eigh_full(T, sz; kwargs...)
        test_mooncake_eigh_vals(T, sz; kwargs...)
        test_mooncake_eigh_trunc(T, sz; kwargs...)
    end
end

function test_mooncake_eigh_full(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "eigh_full" begin
        A = make_eigh_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(eigh_full, A)
        DV = eigh_full(A, alg)
        ΔDV = Mooncake.randn_tangent(rng, DV)
        remove_eigh_gauge_dependence!(ΔDV[2], DV...)

        Mooncake.TestUtils.test_rule(
            rng, eigh_wrapper, eigh_full, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔDV, is_primitive = false, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, eigh!_wrapper, eigh_full!, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔDV, atol, rtol, is_primitive = false
        )
    end
end

function test_mooncake_eigh_vals(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "eigh_vals" begin
        A = make_eigh_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(eigh_vals, A)
        D = eigh_vals(A, alg)
        ΔD = Mooncake.randn_tangent(rng, D)

        Mooncake.TestUtils.test_rule(
            rng, eigh_wrapper, eigh_vals, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔD, is_primitive = false, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, eigh!_wrapper, eigh_vals!, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔD, atol, rtol, is_primitive = false
        )
    end
end

function test_mooncake_eigh_trunc(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "eigh_trunc" begin
        A = make_eigh_matrix(T, sz)
        m = size(A, 1)

        alg = MatrixAlgebraKit.select_algorithm(eigh_full, A)
        DV = eigh_full(A, alg)
        ΔDV = Mooncake.randn_tangent(rng, DV)
        remove_eigh_gauge_dependence!(ΔDV[2], DV...)

        @testset "truncrank($r)" for r in round.(Int, range(1, m + 4, 4))
            trunc = truncrank(r; by = abs)
            alg_trunc = TruncatedAlgorithm(alg, trunc)

            # truncate the gauge-corrected tangents
            DVtrunc, ind = MatrixAlgebraKit.truncate(eigh_trunc!, DV, trunc)
            ΔDV_primal = Mooncake.tangent_to_primal!!(copy.(DV), ΔDV)
            ΔDVtrunc_primal = (Diagonal(diagview(ΔDV_primal[1])[ind]), ΔDV_primal[2][:, ind])
            ΔDVtrunc = Mooncake.primal_to_tangent!!(Mooncake.zero_tangent(DVtrunc), ΔDVtrunc_primal)

            Mooncake.TestUtils.test_rule(
                rng, eigh_wrapper, eigh_trunc_no_error, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔDVtrunc, atol, rtol, is_primitive = false
            )
            Mooncake.TestUtils.test_rule(
                rng, eigh!_wrapper, eigh_trunc_no_error!, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔDVtrunc, atol, rtol, is_primitive = false
            )

            DVϵ = eigh_trunc(A, alg_trunc)
            Δϵ = Mooncake.zero_tangent(DVϵ[end])
            ΔDVϵtrunc = (ΔDVtrunc..., Δϵ)

            Mooncake.TestUtils.test_rule(
                rng, eigh_wrapper, eigh_trunc, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔDVϵtrunc, atol, rtol, is_primitive = false
            )
            Mooncake.TestUtils.test_rule(
                rng, eigh!_wrapper, eigh_trunc!, A, alg_trunc;
                mode = Mooncake.ReverseMode, output_tangent = ΔDVϵtrunc, atol, rtol, is_primitive = false
            )
        end
    end
end
