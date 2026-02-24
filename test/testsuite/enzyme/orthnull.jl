"""
    test_enzyme_orthnull(T, sz; kwargs...)

Run all Enzyme AD tests for orthogonal basis and null space computations of element type `T`
and size `sz`.
"""
function test_enzyme_orthnull(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "Enzyme orthnull $summary_str" begin
        test_enzyme_left_orth(T, sz; kwargs...)
        test_enzyme_right_orth(T, sz; kwargs...)
        test_enzyme_left_null(T, sz; kwargs...)
        test_enzyme_right_null(T, sz; kwargs...)
    end
end

"""
    test_enzyme_left_orth(T, sz; rng, atol, rtol)

Test the Enzyme reverse-mode AD rules for `left_orth` with QR and polar (when `m >= n`)
algorithms, and their in-place variants.
"""
function test_enzyme_left_orth(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    )
    return @testset "left_orth reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        m, n = size(A)

        @testset "qr" begin
            A = instantiate_matrix(T, sz)
            alg = MatrixAlgebraKit.select_algorithm(left_orth!, A, :qr)
            VC, ΔVC = ad_left_orth_setup(A)
            test_reverse(left_orth, RT, (A, TA), (alg, Const); atol, rtol, fdm, output_tangent = ΔVC)
            test_reverse(call_and_zero!, RT, (left_orth!, Const), (A, TA), (alg, Const); atol, rtol, fdm, output_tangent = ΔVC)
        end

        if m >= n
            @testset "polar" begin
                A = instantiate_matrix(T, sz)
                alg = MatrixAlgebraKit.select_algorithm(left_orth!, A, :polar)
                VC, ΔVC = ad_left_orth_setup(A)
                test_reverse(left_orth, RT, (A, TA), (alg, Const); atol, rtol, fdm, output_tangent = ΔVC)
                test_reverse(call_and_zero!, RT, (left_orth!, Const), (A, TA), (alg, Const); atol, rtol, fdm, output_tangent = ΔVC)
            end
        end
    end
end

"""
    test_enzyme_right_orth(T, sz; rng, atol, rtol)

Test the Enzyme reverse-mode AD rules for `right_orth` with LQ and polar (when `m <= n`)
algorithms, and their in-place variants.
"""
function test_enzyme_right_orth(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    )
    return @testset "right_orth reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        m, n = size(A)
        @testset "lq" begin
            A = instantiate_matrix(T, sz)
            alg = MatrixAlgebraKit.select_algorithm(right_orth!, A, :lq)
            CVᴴ, ΔCVᴴ = ad_right_orth_setup(A)
            test_reverse(right_orth, RT, (A, TA), (alg, Const); atol, rtol, fdm, output_tangent = ΔCVᴴ)
            test_reverse(call_and_zero!, RT, (right_orth!, Const), (A, TA), (alg, Const); atol, rtol, fdm, output_tangent = ΔCVᴴ)
        end

        if m <= n
            @testset "polar" begin
                A = instantiate_matrix(T, sz)
                alg = MatrixAlgebraKit.select_algorithm(right_orth!, A, :polar)
                CVᴴ, ΔCVᴴ = ad_right_orth_setup(A)
                test_reverse(right_orth, RT, (A, TA), (alg, Const); atol, rtol, fdm, output_tangent = ΔCVᴴ)
                test_reverse(call_and_zero!, RT, (right_orth!, Const), (A, TA), (alg, Const); atol, rtol, fdm, output_tangent = ΔCVᴴ)
            end
        end
    end
end

"""
    test_enzyme_left_null(T, sz; rng, atol, rtol)

Test the Enzyme reverse-mode AD rule for `left_null` with the QR algorithm and its
in-place variant.
"""
function test_enzyme_left_null(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    )
    return @testset "left_null reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        @testset "qr" begin
            alg = MatrixAlgebraKit.select_algorithm(left_null!, A, :qr)
            N, ΔN = ad_left_null_setup(A)
            test_reverse(left_null, RT, (A, TA), (alg, Const); output_tangent = ΔN, atol, rtol)
            test_reverse(call_and_zero!, RT, (left_null!, Const), (A, TA), (alg, Const); output_tangent = ΔN, atol, rtol)
        end
    end
end

"""
    test_enzyme_right_null(T, sz; rng, atol, rtol)

Test the Enzyme reverse-mode AD rule for `right_null` with the LQ algorithm and its
in-place variant.
"""
function test_enzyme_right_null(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T),
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    )
    return @testset "right_null reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        A = instantiate_matrix(T, sz)
        @testset "lq" begin
            alg = MatrixAlgebraKit.select_algorithm(right_null!, A, :lq)
            Nᴴ, ΔNᴴ = ad_right_null_setup(A)
            test_reverse(right_null, RT, (A, TA), (alg, Const); output_tangent = ΔNᴴ, atol, rtol)
            test_reverse(call_and_zero!, RT, (right_null!, Const), (A, TA), (alg, Const); output_tangent = ΔNᴴ, atol, rtol)
        end
    end
end
