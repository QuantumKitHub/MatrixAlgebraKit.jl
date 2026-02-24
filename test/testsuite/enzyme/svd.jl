function test_enzyme_svd(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "Enzyme svd $summary_str" begin
        test_enzyme_svd_compact(T, sz; kwargs...)
        test_enzyme_svd_full(T, sz; kwargs...)
        test_enzyme_svd_vals(T, sz; kwargs...)
        test_enzyme_svd_trunc(T, sz; kwargs...)
    end
end

function test_enzyme_svd_compact(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    )
    return @testset "svd_compact reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(svd_compact, A)
        USVᴴ, ΔUSVᴴ = ad_svd_compact_setup(A)
        test_reverse(svd_compact, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔUSVᴴ, fdm)
        test_reverse(call_and_zero!, RT, (svd_compact!, Const), (A, TA), (alg, Const); atol, rtol, output_tangent = ΔUSVᴴ, fdm)
    end
end

function test_enzyme_svd_full(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    )
    return @testset "svd_full reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(svd_full, A)
        USVᴴ, ΔUSVᴴ = ad_svd_full_setup(A)
        test_reverse(svd_full, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔUSVᴴ, fdm)
        test_reverse(call_and_zero!, RT, (svd_full!, Const), (A, TA), (alg, Const); atol, rtol, output_tangent = ΔUSVᴴ, fdm)
    end
end

function test_enzyme_svd_vals(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    )
    return @testset "svd_vals reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(svd_vals, A)
        S, ΔS = ad_svd_vals_setup(A)
        test_reverse(svd_vals, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔS, fdm)
        test_reverse(call_and_zero!, RT, (svd_vals!, Const), (A, TA), (alg, Const); atol, rtol, output_tangent = ΔS, fdm)
    end
end

function test_enzyme_svd_trunc(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    )
    return @testset "svd_trunc reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        m, n = size(A)
        minmn = min(m, n)
        alg = MatrixAlgebraKit.select_algorithm(svd_compact, A)
        @testset "truncrank($r)" for r in round.(Int, range(1, minmn + 4, 4))
            A = instantiate_matrix(T, sz)
            trunc = truncrank(r)
            truncalg = TruncatedAlgorithm(alg, trunc)
            USVᴴ, _, ΔUSVᴴ, ΔUSVᴴtrunc = ad_svd_trunc_setup(A, truncalg)
            test_reverse(svd_trunc_no_error, RT, (A, TA), (truncalg, Const); atol, rtol, output_tangent = ΔUSVᴴtrunc, fdm)
            test_reverse(call_and_zero!, RT, (svd_trunc_no_error!, Const), (A, TA), (truncalg, Const); atol, rtol, output_tangent = ΔUSVᴴtrunc, fdm)
        end
        @testset "trunctol" begin
            A = instantiate_matrix(T, sz)
            S = svd_vals(A, alg)
            trunc = trunctol(atol = S[1] / 2)
            truncalg = TruncatedAlgorithm(alg, trunc)
            USVᴴ, _, ΔUSVᴴ, ΔUSVᴴtrunc = ad_svd_trunc_setup(A, truncalg)
            test_reverse(svd_trunc_no_error, RT, (A, TA), (truncalg, Const); atol, rtol, output_tangent = ΔUSVᴴtrunc, fdm)
            test_reverse(call_and_zero!, RT, (svd_trunc_no_error!, Const), (A, TA), (truncalg, Const); atol, rtol, output_tangent = ΔUSVᴴtrunc, fdm)
        end
    end
end
