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
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    )
    return @testset "lq_compact reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(lq_compact, A)
        LQ, ΔLQ = ad_lq_compact_setup(A)
        test_reverse(lq_compact, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔLQ, fdm)
        test_reverse(call_and_zero!, RT, (lq_compact!, Const), (A, TA), (alg, Const); atol, rtol, output_tangent = ΔLQ, fdm)
    end
end

function test_enzyme_lq_compact_rank_deficient(
        T::Type, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    )
    return @testset "lq_compact rank deficient A reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        m, n = size(A)
        r = min(m, n) - 5
        A = instantiate_matrix(T, (m, r)) * instantiate_matrix(T, (r, n))
        alg = MatrixAlgebraKit.select_algorithm(lq_compact, A)
        LQ, ΔLQ = ad_lq_rank_deficient_compact_setup(A)
        test_reverse(lq_compact, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔLQ, fdm)
        test_reverse(call_and_zero!, RT, (lq_compact!, Const), (A, TA), (alg, Const); atol, rtol, output_tangent = ΔLQ, fdm)
    end
end

function test_enzyme_lq_full(
        T::Type, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    )
    return @testset "lq_full reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(lq_full, A)
        LQ, ΔLQ = ad_lq_full_setup(A)
        test_reverse(lq_full, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔLQ, fdm)
        test_reverse(call_and_zero!, RT, (lq_full!, Const), (A, TA), (alg, Const); atol, rtol, output_tangent = ΔLQ, fdm)
    end
end

function test_enzyme_lq_null(
        T::Type, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    )
    return @testset "lq_null reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(lq_null, A)
        Nᴴ, ΔNᴴ = ad_lq_null_setup(A)
        test_reverse(lq_null, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔNᴴ)
        test_reverse(call_and_zero!, RT, (lq_null!, Const), (A, TA), (alg, Const); atol, rtol, output_tangent = ΔNᴴ)
    end
end
