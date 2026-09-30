"""
    test_enzyme_lq(T, sz; kwargs...)

Run all Enzyme AD tests for LQ decompositions of element type `T` and size `sz`.
"""
function test_enzyme_lq(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "Enzyme lq $summary_str" begin
        test_enzyme_lq_compact(T, sz; kwargs...)
        test_enzyme_lq_compact_rank_deficient(T, sz; kwargs...)
        test_enzyme_lq_full(T, sz; kwargs...)
        test_enzyme_lq_null(T, sz; kwargs...)
    end
end

function test_enzyme_lq_compact(
        T::Type, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = enzyme_fdm(T)
    )
    return @testset "lq_compact: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(lq_compact, A)
        LQ, ΔLQ = ad_lq_compact_setup(A)
        test_reverse(lq_compact, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔLQ, fdm)
        test_reverse(call_and_zero!, RT, (lq_compact!, Const), (copy(A), TA), (alg, Const); atol, rtol, output_tangent = ΔLQ, fdm)
        test_forward(lq_compact, RT, (A, TA), (alg, Const); atol, rtol, fdm)
        test_forward(call_and_zero!, RT, (lq_compact!, Const), (copy(A), TA), (alg, Const); atol, rtol, fdm)
    end
end

function test_enzyme_lq_compact_rank_deficient(
        T::Type, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = enzyme_fdm(T)
    )
    return @testset "lq_compact rank deficient A: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        m, n = size(A)
        r = min(m, n) - 5
        A = instantiate_matrix(T, (m, r)) * instantiate_matrix(T, (r, n))
        alg = MatrixAlgebraKit.select_algorithm(lq_compact, A)
        LQ, ΔLQ = ad_lq_compact_setup(A)
        test_reverse(lq_compact, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔLQ, fdm)
        test_reverse(call_and_zero!, RT, (lq_compact!, Const), (copy(A), TA), (alg, Const); atol, rtol, output_tangent = ΔLQ, fdm)
        # only the first r columns/rows of the isometric factor are differentiable
        r = MatrixAlgebraKit.lq_rank(LQ[1])
        test_forward(lq_gauge_invariant_wrapper, RT, (lq_compact, Const), (A, TA), (alg, Const), (r, Const); atol, rtol, fdm)
        test_forward(lq!_gauge_invariant_wrapper, RT, (lq_compact!, Const), (copy(A), TA), (alg, Const), (r, Const); atol, rtol, fdm)
    end
end

function test_enzyme_lq_full(
        T::Type, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = enzyme_fdm(T)
    )
    return @testset "lq_full reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(lq_full, A)
        LQ, ΔLQ = ad_lq_full_setup(A)
        test_reverse(lq_full, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔLQ, fdm)
        test_reverse(call_and_zero!, RT, (lq_full!, Const), (copy(A), TA), (alg, Const); atol, rtol, output_tangent = ΔLQ, fdm)
        # the extra columns/rows of the isometric factor are only determined up to a unitary rotation
        r = min(size(A)...)
        test_forward(lq_gauge_invariant_wrapper, RT, (lq_full, Const), (A, TA), (alg, Const), (r, Const); atol, rtol, fdm)
        test_forward(lq!_gauge_invariant_wrapper, RT, (lq_full!, Const), (copy(A), TA), (alg, Const), (r, Const); atol, rtol, fdm)
    end
end

function test_enzyme_lq_null(
        T::Type, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = enzyme_fdm(T)
    )
    return @testset "lq_null reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(lq_null, A)
        Nᴴ, ΔNᴴ = ad_lq_null_setup(A)
        test_reverse(lq_null, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔNᴴ)
        test_reverse(call_and_zero!, RT, (lq_null!, Const), (copy(A), TA), (alg, Const); atol, rtol, output_tangent = ΔNᴴ)
        # the nullspace basis is only determined up to a unitary rotation
        test_forward(lq_null_gauge_invariant_wrapper, RT, (lq_null, Const), (A, TA), (alg, Const); atol, rtol)
        test_forward(lq_null!_gauge_invariant_wrapper, RT, (lq_null!, Const), (copy(A), TA), (alg, Const); atol, rtol)
    end
end
