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
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    )
    return @testset "qr_compact reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(qr_compact, A)
        QR, ΔQR = ad_qr_compact_setup(A)
        test_reverse(qr_compact, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔQR, fdm)
        test_reverse(call_and_zero!, RT, (qr_compact!, Const), (A, TA), (alg, Const); atol, rtol, output_tangent = ΔQR, fdm)
    end
end

function test_enzyme_qr_compact_rank_deficient(
        T::Type, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    )
    return @testset "qr_compact rank deficient A reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        m, n = size(A)
        r = min(m, n) - 5
        A = instantiate_matrix(T, (m, r)) * instantiate_matrix(T, (r, n))
        alg = MatrixAlgebraKit.select_algorithm(qr_compact, A)
        QR, ΔQR = ad_qr_rank_deficient_compact_setup(A)
        test_reverse(qr_compact, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔQR, fdm)
        test_reverse(call_and_zero!, RT, (qr_compact!, Const), (A, TA), (alg, Const); atol, rtol, output_tangent = ΔQR, fdm)
    end
end

function test_enzyme_qr_full(
        T::Type, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    )
    return @testset "qr_full reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(qr_full, A)
        QR, ΔQR = ad_qr_full_setup(A)
        test_reverse(qr_full, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔQR, fdm)
        test_reverse(call_and_zero!, RT, (qr_full!, Const), (A, TA), (alg, Const); atol, rtol, output_tangent = ΔQR, fdm)
    end
end

function test_enzyme_qr_null(
        T::Type, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    )
    return @testset "qr_null reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(qr_null, A)
        N, ΔN = ad_qr_null_setup(A)
        test_reverse(qr_null, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔN)
        test_reverse(call_and_zero!, RT, (qr_null!, Const), (A, TA), (alg, Const); atol, rtol, output_tangent = ΔN)
    end
end
