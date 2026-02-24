"""
    test_enzyme_eig(T, sz; kwargs...)

Run all Enzyme AD tests for eigendecompositions of element type `T` and size `sz`.
"""
function test_enzyme_eig(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "Enzyme eig $summary_str" begin
        test_enzyme_eig_full(T, sz; kwargs...)
        test_enzyme_eig_vals(T, sz; kwargs...)
        test_enzyme_eig_trunc(T, sz; kwargs...)
    end
end

"""
    test_enzyme_eig_full(T, sz; rng, atol, rtol)

Test the Enzyme reverse-mode AD rule for `eig_full` and its in-place variant.
"""
function test_enzyme_eig_full(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    )
    return @testset "eig_full reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = make_eig_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(eig_full, A)
        DV, ΔDV = ad_eig_full_setup(A)
        test_reverse(eig_full, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔDV, fdm)
        test_reverse(call_and_zero!, RT, (eig_full!, Const), (A, TA), (alg, Const); atol, rtol, output_tangent = ΔDV, fdm)
    end
end

"""
    test_enzyme_eig_vals(T, sz; rng, atol, rtol)

Test the Enzyme reverse-mode AD rule for `eig_vals` and its in-place variant.
"""
function test_enzyme_eig_vals(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    )
    return @testset "eig_vals reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = make_eig_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(eig_vals, A)
        D, ΔD = ad_eig_vals_setup(A)
        test_reverse(eig_vals, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔD, fdm)
        test_reverse(call_and_zero!, RT, (eig_vals!, Const), (A, TA), (alg, Const); atol, rtol, output_tangent = ΔD, fdm)
    end
end

"""
    test_enzyme_eig_trunc(T, sz; rng, atol, rtol)

Test the Enzyme reverse-mode AD rules for `eig_trunc`, `eig_trunc_no_error`, and their
in-place variants, over a range of truncation ranks and a tolerance-based truncation.
"""
function test_enzyme_eig_trunc(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    )
    return @testset "eig_trunc reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = make_eig_matrix(T, sz)
        m = size(A, 1)

        alg = MatrixAlgebraKit.select_algorithm(eig_full, A)
        @testset "truncrank($r)" for r in round.(Int, range(1, m + 4, 4))
            trunc = truncrank(r; by = abs)
            truncalg = TruncatedAlgorithm(alg, trunc)
            A = make_eig_matrix(T, sz)
            DV, _, ΔDV, ΔDVtrunc = ad_eig_trunc_setup(A, truncalg)
            test_reverse(eig_trunc_no_error, RT, (A, TA), (truncalg, Const); atol, rtol, output_tangent = ΔDVtrunc, fdm)
            test_reverse(call_and_zero!, RT, (eig_trunc_no_error!, Const), (A, TA), (truncalg, Const); atol, rtol, output_tangent = ΔDVtrunc, fdm)
        end
        @testset "trunctol" begin
            A = make_eig_matrix(T, sz)
            D = eig_vals(A)
            trunc = trunctol(atol = maximum(abs, D) / 2; by = abs)
            truncalg = TruncatedAlgorithm(alg, trunc)
            DV, _, ΔDV, ΔDVtrunc = ad_eig_trunc_setup(A, truncalg)
            test_reverse(eig_trunc_no_error, RT, (A, TA), (truncalg, Const); atol, rtol, output_tangent = ΔDVtrunc, fdm)
            test_reverse(call_and_zero!, RT, (eig_trunc_no_error!, Const), (A, TA), (truncalg, Const); atol, rtol, output_tangent = ΔDVtrunc, fdm)
        end
    end
end
