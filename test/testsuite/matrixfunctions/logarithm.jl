using TestExtras
using LinearAlgebra: LinearAlgebra, I
using MatrixAlgebraKit: ishermitian

# `exponential` is the inverse of `logarithm` on the principal branch, and its default algorithm
# is backend-generic, so it serves as the reference invariant here.

function test_logarithm(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "logarithm $summary_str" begin
        A = instantiate_offaxis_matrix(T, sz)
        Ac = deepcopy(A)

        logA = @testinferred logarithm(A)
        @test eltype(logA) == eltype(A)
        @test exponential(logA) ≈ A
        @test A == Ac

        # the in-place method may not be able to reuse the provided output
        logA2 = @testinferred logarithm!(deepcopy(A), deepcopy(logA))
        @test logA2 ≈ logA
    end
end

function test_logarithm_algs(T::Type, sz, algs; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "logarithm algorithm $alg $summary_str" for alg in algs
        A = instantiate_offaxis_matrix(T, sz)
        Ac = deepcopy(A)

        logA = @testinferred logarithm(A, alg)
        @test eltype(logA) == eltype(A)
        @test exponential(logA) ≈ A
        @test A == Ac
    end
end

# See the corresponding comment in `squareroot.jl` for `exact_hermiticity`.
function test_logarithm_hermitian(
        T::Type, sz, algs;
        exact_hermiticity = true, test_spectrum = true, kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "logarithm hermitian algorithm $alg $summary_str" for alg in algs
        A = instantiate_posdef_matrix(T, sz)
        Ac = deepcopy(A)

        logA = @testinferred logarithm(A, alg)
        @test eltype(logA) == eltype(A)
        @test exponential(logA) ≈ A
        @test A == Ac

        if exact_hermiticity
            @test ishermitian(logA)
            test_spectrum && @test eigh_vals(logA) ≈ log.(eigh_vals(A))
        else
            @test ishermitian(logA; rtol = precision(T))
        end
    end
end

# Domain handling. `logarithm` is undefined both on the negative real axis and at zero, and the
# two interact: an eigenvalue that is negative only by roundoff gets clamped onto the boundary,
# i.e. to zero, where there is still no logarithm. So unlike `squareroot`, the clamp does not
# rescue such a matrix and a `DomainError` is still the correct outcome.
#
# `hermitian_output = true`: see `squareroot.jl`.
# `supports_domain_atol = false`: for `MatrixFunctionViaLA`, which inspects only the realness of
# the result rather than the spectrum, so it silently accepts singular input.
function test_logarithm_domain(
        T::Type, sz, algs;
        hermitian_output = false, supports_domain_atol = true, kwargs...
    )
    R = real(eltype(T))
    n = sz isa Tuple ? first(sz) : sz
    summary_str = testargs_summary(T, sz)
    return @testset "logarithm domain algorithm $alg $summary_str" for alg in algs
        # eigenvalue on the negative real axis
        λ = collect(R, 1:n)
        λ[1] = -one(R)
        A = instantiate_hermitian_spectrum(T, sz, λ)

        if eltype(T) <: Real || hermitian_output
            @test_throws DomainError logarithm(A, alg)
        else
            logA = @testinferred logarithm(A, alg)
            @test exponential(logA) ≈ A
        end

        supports_domain_atol || continue

        # (numerically) zero eigenvalue: no logarithm exists
        λzero = collect(R, 1:n)
        λzero[1] = zero(R)
        @test_throws DomainError logarithm(instantiate_hermitian_spectrum(T, sz, λzero), alg)

        # roundoff-scale negative eigenvalue: clamped onto the boundary, which is still singular
        λtiny = collect(R, 1:n)
        λtiny[1] = -10 * eps(R)
        @test_throws DomainError logarithm(instantiate_hermitian_spectrum(T, sz, λtiny), alg)
    end
end

# Cross-check against `LinearAlgebra`, which only applies to host arrays. Pass
# `test_hermitian = false` for generic eltypes, as `LinearAlgebra` has no matrix functions for a
# `Hermitian` wrapper outside the BLAS floats.
#
# Note this must not be called for *dense* generic eltypes at all:
# `LinearAlgebra.log(::UpperTriangular{BigFloat})` does not terminate.
function test_logarithm_reference(T::Type, sz; test_hermitian = true, kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "logarithm vs LinearAlgebra $summary_str" begin
        A = instantiate_offaxis_matrix(T, sz)
        @test logarithm(A) ≈ LinearAlgebra.log(A)

        if test_hermitian
            H = instantiate_posdef_matrix(T, sz)
            @test logarithm(H) ≈ LinearAlgebra.log(LinearAlgebra.Hermitian(H))
        end
    end
end
