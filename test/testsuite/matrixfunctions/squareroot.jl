using TestExtras
using LinearAlgebra: LinearAlgebra, I
using MatrixAlgebraKit: ishermitian

# The assertions below are invariants rather than comparisons against a reference
# implementation, so that the same bodies apply on GPU and to downstream array types.
# `test_squareroot_reference` is the exception and is host-only by construction.

function test_squareroot(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "squareroot $summary_str" begin
        A = instantiate_offaxis_matrix(T, sz)
        Ac = deepcopy(A)

        sqrtA = @testinferred squareroot(A)
        @test eltype(sqrtA) == eltype(A)
        @test sqrtA * sqrtA ≈ A
        @test A == Ac

        # the in-place method may not be able to reuse the provided output
        sqrtA2 = @testinferred squareroot!(deepcopy(A), deepcopy(sqrtA))
        @test sqrtA2 ≈ sqrtA

        # `squareroot` is the `p = 1/2` case of `power`
        @test sqrtA ≈ power(A, one(real(eltype(A))) / 2)
    end
end

function test_squareroot_algs(T::Type, sz, algs; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "squareroot algorithm $alg $summary_str" for alg in algs
        A = instantiate_offaxis_matrix(T, sz)
        Ac = deepcopy(A)

        sqrtA = @testinferred squareroot(A, alg)
        @test eltype(sqrtA) == eltype(A)
        @test sqrtA * sqrtA ≈ A
        @test A == Ac
    end
end

# Hermitian positive definite input.
#
# The `eigh`-based kernels build the result as a symmetric product (`_mul_herm!`), so it is
# hermitian to the last bit; pass `exact_hermiticity = false` for algorithms that route through
# the general `eig` path instead, whose output is only approximately hermitian. The elementwise
# spectrum check needs exact hermiticity, since `eigh_vals` rejects anything else.
function test_squareroot_hermitian(
        T::Type, sz, algs;
        exact_hermiticity = true, test_spectrum = true, kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "squareroot hermitian algorithm $alg $summary_str" for alg in algs
        A = instantiate_posdef_matrix(T, sz)
        Ac = deepcopy(A)

        sqrtA = @testinferred squareroot(A, alg)
        @test eltype(sqrtA) == eltype(A)
        @test sqrtA * sqrtA ≈ A
        @test A == Ac

        if exact_hermiticity
            @test ishermitian(sqrtA)
            # the square root maps the spectrum elementwise
            test_spectrum && @test eigh_vals(sqrtA) ≈ sqrt.(eigh_vals(A))
        else
            @test ishermitian(sqrtA; rtol = precision(T))
        end
    end
end

# Domain handling: a matrix whose spectrum reaches the negative real axis has a complex
# principal square root, which a type-stable real output cannot represent.
#
# Pass `hermitian_output = true` for algorithms that promise a hermitian result (i.e. the
# `eigh`-based ones). Those must reject a negative eigenvalue whatever the scalar type, since
# the square root of a hermitian matrix with a negative eigenvalue is not hermitian.
function test_squareroot_domain(T::Type, sz, algs; hermitian_output = false, kwargs...)
    R = real(eltype(T))
    n = sz isa Tuple ? first(sz) : sz
    summary_str = testargs_summary(T, sz)
    return @testset "squareroot domain algorithm $alg $summary_str" for alg in algs
        # genuinely negative eigenvalue
        λ = collect(R, 1:n)
        λ[1] = -one(R)
        A = instantiate_hermitian_spectrum(T, sz, λ)

        if eltype(T) <: Real || hermitian_output
            @test_throws DomainError squareroot(A, alg)
        else
            sqrtA = @testinferred squareroot(A, alg)
            @test sqrtA * sqrtA ≈ A
        end

        # roundoff-scale negative eigenvalue: clamped onto the boundary rather than rejected
        λclamp = collect(R, 1:n)
        λclamp[1] = -10 * eps(R)
        Aclamp = instantiate_hermitian_spectrum(T, sz, λclamp)
        sqrtAclamp = @testinferred squareroot(Aclamp, alg)
        @test eltype(sqrtAclamp) == eltype(Aclamp)
        @test sqrtAclamp * sqrtAclamp ≈ Aclamp atol = sqrt(eps(R))
    end
end

# Cross-check against `LinearAlgebra`, which only applies to host arrays. Pass
# `test_hermitian = false` for generic eltypes, as `LinearAlgebra` has no matrix functions for a
# `Hermitian` wrapper outside the BLAS floats.
function test_squareroot_reference(T::Type, sz; test_hermitian = true, kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "squareroot vs LinearAlgebra $summary_str" begin
        A = instantiate_offaxis_matrix(T, sz)
        @test squareroot(A) ≈ LinearAlgebra.sqrt(A)

        if test_hermitian
            H = instantiate_posdef_matrix(T, sz)
            @test squareroot(H) ≈ LinearAlgebra.sqrt(LinearAlgebra.Hermitian(H))
        end
    end
end
