using TestExtras
using LinearAlgebra: LinearAlgebra, I
using MatrixAlgebraKit: ishermitian

# the assertions are invariants rather than comparisons against a reference implementation, so that
# the same bodies apply on GPU and to downstream array types. `test_squareroot_reference` is the
# exception and is host-only by construction.

# `rtol`, and `atol` in `test_squareroot_domain`, is the tolerance on the residual `sqrtA^2 ≈ A`.
# Raise it for a scalar type whose decomposition is resolved to less, as in half precision.

function test_squareroot(T::Type, sz; rtol = precision(T), kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "squareroot $summary_str" begin
        A = instantiate_offaxis_matrix(T, sz)
        Ac = deepcopy(A)

        sqrtA = @testinferred squareroot(A)
        @test eltype(sqrtA) == eltype(A)
        @test isapprox(sqrtA * sqrtA, A; rtol)
        @test A == Ac

        # the in-place method may not be able to reuse the provided output
        sqrtA2 = @testinferred squareroot!(deepcopy(A), deepcopy(sqrtA))
        @test sqrtA2 ≈ sqrtA
    end
end

function test_squareroot_algs(T::Type, sz, algs; rtol = precision(T), kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "squareroot algorithm $alg $summary_str" for alg in algs
        A = instantiate_offaxis_matrix(T, sz)
        Ac = deepcopy(A)

        sqrtA = @testinferred squareroot(A, alg)
        @test eltype(sqrtA) == eltype(A)
        @test isapprox(sqrtA * sqrtA, A; rtol)
        @test A == Ac
    end
end

# the `eigh`-based kernels project the result, so it is hermitian to the last bit; pass
# `exact_hermiticity = false` for algorithms that route through the general `eig` path instead.
# The elementwise spectrum check needs exact hermiticity, since `eigh_vals` rejects anything else.
function test_squareroot_hermitian(
        T::Type, sz, algs;
        exact_hermiticity = true, test_spectrum = true, rtol = precision(T), kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "squareroot hermitian algorithm $alg $summary_str" for alg in algs
        A = instantiate_posdef_matrix(T, sz)
        Ac = deepcopy(A)

        sqrtA = @testinferred squareroot(A, alg)
        @test eltype(sqrtA) == eltype(A)
        @test isapprox(sqrtA * sqrtA, A; rtol)
        @test A == Ac

        if exact_hermiticity
            @test ishermitian(sqrtA)
            test_spectrum && @test isapprox(eigh_vals(sqrtA), sqrt.(eigh_vals(A)); rtol)
        else
            @test ishermitian(sqrtA; rtol)
        end
    end
end

# a matrix whose spectrum reaches the negative real axis has a complex principal square root, which
# a type-stable real output cannot represent. Pass `hermitian_output = true` for the algorithms that
# promise a hermitian result: those must reject a negative eigenvalue whatever the scalar type, and
# `test_domain_atol = false` for the ones without access to the spectrum, which have no tolerance.
function test_squareroot_domain(
        T::Type, sz, algs;
        hermitian_output = false, test_domain_atol = true,
        atol = sqrt(eps(real(eltype(T)))), kwargs...
    )
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
        λclamp[1] = -eps(R)
        Aclamp = instantiate_hermitian_spectrum(T, sz, λclamp)
        sqrtAclamp = @testinferred squareroot(Aclamp, alg)
        @test eltype(sqrtAclamp) == eltype(Aclamp)
        @test isapprox(sqrtAclamp * sqrtAclamp, Aclamp; atol)

        # an eigenvalue beyond the default tolerance is out of domain, while an explicit
        # `domain_atol` admits it after all
        λwide = collect(R, 1:n)
        λwide[1] = -sqrt(eps(R))
        Awide = instantiate_hermitian_spectrum(T, sz, λwide)
        if eltype(T) <: Real || hermitian_output
            @test_throws DomainError squareroot(Awide, alg)
        else
            sqrtAwide = @testinferred squareroot(Awide, alg)
            @test sqrtAwide * sqrtAwide ≈ Awide
        end
        if test_domain_atol
            wide_alg = with_domain_atol(alg, cbrt(eps(R)))
            sqrtAwide = @testinferred squareroot(Awide, wide_alg)
            @test eltype(sqrtAwide) == eltype(Awide)
            # accepting is backward stable, but only to the size of the eigenvalue that was discarded
            @test isapprox(sqrtAwide * sqrtAwide, Awide; atol = sqrt(atol))
        end
    end
end

# cross-check against `LinearAlgebra`, host arrays only. `LinearAlgebra` has no matrix functions
# for a `Hermitian` wrapper outside the BLAS floats, hence `test_hermitian`.
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

# A general matrix carries 2x2 blocks in its real Schur form while a hermitian one carries none, and
# the two structures drive the recursion differently. Every block size shares one decomposition of
# the same input and differs only in the kernel, hence the comparison against each other.
function test_squareroot_blocked(T::Type, sz, algs; rtol = precision(T), kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "squareroot blocking $alg $summary_str" for alg in algs
        for A in (instantiate_offaxis_matrix(T, sz), instantiate_posdef_matrix(T, sz))
            sqrtA = squareroot(A, with_blocksize(alg, 1))
            @test isapprox(sqrtA * sqrtA, A; rtol)
            for blocksize in (2, 3, 8)
                @test squareroot(A, with_blocksize(alg, blocksize)) ≈ sqrtA
            end
        end
    end
end

# A Schur-based algorithm never inverts an eigenvector matrix, so it stays backward stable where
# `MatrixFunctionViaEig` resolves the eigenvalues to no better than `sqrt(eps)`; hence the residual
# is held to roundoff rather than to the default `≈`, which the eigenvector route would still meet.
# A defective eigenvalue *on* the negative real axis is left out: whether the perturbed pair
# surfaces as two real eigenvalues or as a conjugate pair is up to the eigensolver.
function test_squareroot_defective(T::Type, sz, algs; kwargs...)
    R = real(eltype(T))
    n = sz isa Tuple ? first(sz) : sz
    summary_str = testargs_summary(T, sz)
    return @testset "squareroot defective algorithm $alg $summary_str" for alg in algs
        A = instantiate_defective_matrix(T, sz, one(R) + one(R))
        sqrtA = @testinferred squareroot(A, alg)
        @test eltype(sqrtA) == eltype(A)
        @test norm(sqrtA * sqrtA - A) <= 100 * n * eps(R) * norm(A)
    end
end
