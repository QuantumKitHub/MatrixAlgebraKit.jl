"""
    test_enzyme_eigh(T, sz; kwargs...)

Run all Enzyme AD tests for Hermitian eigendecompositions of element type `T` and size `sz`.
"""
function test_enzyme_eigh(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "Enzyme eigh $summary_str" begin
        test_enzyme_eigh_full(T, sz; kwargs...)
        test_enzyme_eigh_vals(T, sz; kwargs...)
        test_enzyme_eigh_trunc(T, sz; kwargs...)
    end
end

"""
    test_enzyme_eigh_full(T, sz; rng, atol, rtol)

Test the Enzyme reverse-mode AD rule for `eigh_full` and its in-place variant.
"""
function test_enzyme_eigh_full(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    )
    return @testset "eigh_full reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = make_eigh_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(eigh_full, A)
        DV, ΔDV = ad_eigh_full_setup(A)
        test_reverse(eigh_wrapper, RT, (eigh_full, Const), (A, TA), (alg, Const); atol, rtol, output_tangent = ΔDV, fdm)
        test_reverse(eigh!_wrapper, RT, (eigh_full!, Const), (A, TA), (alg, Const); atol, rtol, output_tangent = ΔDV, fdm)
    end
end

"""
    test_enzyme_eigh_vals(T, sz; rng, atol, rtol)

Test the Enzyme reverse-mode AD rule for `eigh_vals` and its in-place variant.
"""
function test_enzyme_eigh_vals(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    )
    return @testset "eigh_vals reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = make_eigh_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(eigh_vals, A)
        D, ΔD = ad_eigh_vals_setup(A)
        test_reverse(eigh_wrapper, RT, (eigh_vals, Const), (A, TA), (alg, Const); atol, rtol, output_tangent = ΔD, fdm)
        test_reverse(eigh!_wrapper, RT, (eigh_vals!, Const), (A, TA), (alg, Const); atol, rtol, output_tangent = ΔD, fdm)
    end
end

"""
    test_enzyme_eigh_trunc(T, sz; rng, atol, rtol)

Test the Enzyme reverse-mode AD rules for `eigh_trunc`, `eigh_trunc_no_error`, and their
in-place variants, over a range of truncation ranks.
"""
function test_enzyme_eigh_trunc(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    )
    return @testset "eigh_trunc reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = make_eigh_matrix(T, sz)
        m = size(A, 1)

        alg = MatrixAlgebraKit.select_algorithm(eigh_full, A)
        @testset "truncrank($r)" for r in round.(Int, range(1, m + 4, 4))
            trunc = truncrank(r; by = abs)
            truncalg = TruncatedAlgorithm(alg, trunc)
            A = make_eigh_matrix(T, sz)
            DV, _, ΔDV, ΔDVtrunc = ad_eigh_trunc_setup(A, truncalg)
            test_reverse(eigh_wrapper, RT, (eigh_trunc_no_error, Const), (A, TA), (truncalg, Const); atol, rtol, output_tangent = ΔDVtrunc, fdm)
            test_reverse(eigh!_wrapper, RT, (eigh_trunc_no_error!, Const), (A, TA), (truncalg, Const); atol, rtol, output_tangent = ΔDVtrunc, fdm)
        end
        @testset "trunctol" begin
            A = make_eigh_matrix(T, sz)
            D = eigh_vals(A / 2, alg)
            trunc = trunctol(; atol = maximum(abs, D) / 2)
            truncalg = TruncatedAlgorithm(alg, trunc)
            DV, _, ΔDV, ΔDVtrunc = ad_eigh_trunc_setup(A, truncalg)
            test_reverse(eigh_wrapper, RT, (eigh_trunc_no_error, Const), (A, TA), (truncalg, Const); atol, rtol, output_tangent = ΔDVtrunc, fdm)
            test_reverse(eigh!_wrapper, RT, (eigh_trunc_no_error!, Const), (A, TA), (truncalg, Const); atol, rtol, output_tangent = ΔDVtrunc, fdm)
        end
    end
end
