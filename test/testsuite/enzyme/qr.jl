"""
    test_enzyme_qr(T, sz; kwargs...)

Run all Enzyme AD tests for QR decompositions of element type `T` and size `sz`.
"""
function test_enzyme_qr(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "Enzyme qr $summary_str" begin
        test_enzyme_qr_compact(T, sz; kwargs...)
        test_enzyme_qr_compact_rank_deficient(T, sz; kwargs...)
        test_enzyme_qr_full(T, sz; kwargs...)
        test_enzyme_qr_null(T, sz; kwargs...)
    end
end

function test_enzyme_qr_compact(
        T::Type, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = enzyme_fdm(T)
    )
    return @testset "qr_compact reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(qr_compact, A)
        QR, ΔQR = ad_qr_compact_setup(A)
        test_reverse(qr_compact, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔQR, fdm)
        test_reverse(call_and_zero!, RT, (qr_compact!, Const), (copy(A), TA), (alg, Const); atol, rtol, output_tangent = ΔQR, fdm)
        test_forward(qr_compact, RT, (A, TA), (alg, Const); atol, rtol, fdm)
        test_forward(call_and_zero!, RT, (qr_compact!, Const), (copy(A), TA), (alg, Const); atol, rtol, fdm)
    end
end

function test_enzyme_qr_compact_rank_deficient(
        T::Type, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = enzyme_fdm(T)
    )
    return @testset "qr_compact rank deficient A reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        m, n = size(A)
        r = min(m, n) - 5
        A = instantiate_matrix(T, (m, r)) * instantiate_matrix(T, (r, n))
        alg = MatrixAlgebraKit.select_algorithm(qr_compact, A)
        QR, ΔQR = ad_qr_compact_setup(A)
        test_reverse(qr_compact, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔQR, fdm)
        test_reverse(call_and_zero!, RT, (qr_compact!, Const), (copy(A), TA), (alg, Const); atol, rtol, output_tangent = ΔQR, fdm)
        # only the first r columns/rows of the isometric factor are differentiable
        r = MatrixAlgebraKit.qr_rank(QR[2])
        test_forward(qr_gauge_invariant_wrapper, RT, (qr_compact, Const), (A, TA), (alg, Const), (r, Const); atol, rtol, fdm)
        test_forward(qr!_gauge_invariant_wrapper, RT, (qr_compact!, Const), (copy(A), TA), (alg, Const), (r, Const); atol, rtol, fdm)
    end
end

function test_enzyme_qr_full(
        T::Type, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = enzyme_fdm(T)
    )
    return @testset "qr_full reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(qr_full, A)
        QR, ΔQR = ad_qr_full_setup(A)
        test_reverse(qr_full, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔQR, fdm)
        test_reverse(call_and_zero!, RT, (qr_full!, Const), (copy(A), TA), (alg, Const); atol, rtol, output_tangent = ΔQR, fdm)
        # the extra columns/rows of the isometric factor are only determined up to a unitary rotation
        r = min(size(A)...)
        test_forward(qr_gauge_invariant_wrapper, RT, (qr_full, Const), (A, TA), (alg, Const), (r, Const); atol, rtol, fdm)
        test_forward(qr!_gauge_invariant_wrapper, RT, (qr_full!, Const), (copy(A), TA), (alg, Const), (r, Const); atol, rtol, fdm)
    end
end

function test_enzyme_qr_null(
        T::Type, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = enzyme_fdm(T)
    )
    return @testset "qr_null reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(qr_null, A)
        N, ΔN = ad_qr_null_setup(A)
        test_reverse(qr_null, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔN)
        test_reverse(call_and_zero!, RT, (qr_null!, Const), (copy(A), TA), (alg, Const); atol, rtol, output_tangent = ΔN)
        # the nullspace basis is only determined up to a unitary rotation
        test_forward(qr_null_gauge_invariant_wrapper, RT, (qr_null, Const), (A, TA), (alg, Const); atol, rtol)
        test_forward(qr_null!_gauge_invariant_wrapper, RT, (qr_null!, Const), (copy(A), TA), (alg, Const); atol, rtol)
    end
end
