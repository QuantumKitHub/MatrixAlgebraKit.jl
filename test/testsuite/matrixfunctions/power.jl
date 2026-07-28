using TestExtras
using LinearAlgebra: LinearAlgebra, I, SingularException
using MatrixAlgebraKit: ishermitian, one!

# `power` takes the exponent as a second positional argument, and splits into an integer branch
# (repeated multiplication, defined for any square matrix) and a fractional branch (the principal
# power, restricted to the domain). Both are exercised throughout.
#
# Integer and fractional exponents are always iterated in separate, uniformly typed testsets:
# a tuple mixing `Int` and floats would infer as a `Union` and defeat `@testinferred`.

function test_power(T::Type, sz; ps = (0, 1, 2, -1, 3), qs = (1 // 2, 3 // 4, -1 // 4), kwargs...)
    R = real(eltype(T))
    summary_str = testargs_summary(T, sz)
    return @testset "power $summary_str" begin
        A = instantiate_offaxis_matrix(T, sz)
        Ac = deepcopy(A)

        @testset "integer p = $p" for p in ps
            powA = @testinferred power(A, p)
            @test eltype(powA) == eltype(A)
            @test A == Ac
            # an integral exponent is recognised at runtime, as `Base.^` does
            @test powA ≈ power(A, convert(R, p))
        end

        @testset "fractional p = $q" for q in qs
            powA = @testinferred power(A, convert(R, q))
            @test eltype(powA) == eltype(A)
            @test A == Ac
        end

        @test power(A, 0) ≈ I
        @test power(A, 1) ≈ A
        @test power(A, 2) ≈ A * A
        @test power(A, 3) ≈ A * A * A
        @test power(A, -1) * A ≈ I

        # `squareroot` is the `p = 1/2` case, and exponents add
        @test power(A, one(R) / 2) ≈ squareroot(A)
        @test power(A, one(R) / 4) * power(A, one(R) / 4) ≈ power(A, one(R) / 2)

        # the in-place method may not be able to reuse the provided output
        powA = power(A, 2)
        @test @testinferred(power!(deepcopy(A), 2, deepcopy(powA))) ≈ powA
    end
end

function test_power_algs(T::Type, sz, algs; ps = (2, -1), qs = (1 // 2, -1 // 4), kwargs...)
    R = real(eltype(T))
    summary_str = testargs_summary(T, sz)
    return @testset "power algorithm $alg $summary_str" for alg in algs
        A = instantiate_offaxis_matrix(T, sz)
        Ac = deepcopy(A)

        @testset "integer p = $p" for p in ps
            powA = @testinferred power(A, p, alg)
            @test eltype(powA) == eltype(A)
            @test A == Ac
        end

        @testset "fractional p = $q" for q in qs
            powA = @testinferred power(A, convert(R, q), alg)
            @test eltype(powA) == eltype(A)
            @test A == Ac
        end

        @test power(A, 2, alg) ≈ A * A
        @test power(A, -1, alg) * A ≈ I
        @test power(A, one(R) / 2, alg) ≈ squareroot(A, alg)
    end
end

# `A^0 = I` and `A^1 = A` hold for every square matrix, and both are short-circuited before any
# decomposition is computed, so the results are exact rather than merely approximate.
#
# `test_rejected_input = true` for algorithms that cannot handle a general matrix at all (the
# `eigh`-based ones): feeding them input they would reject proves the decomposition really is
# skipped, rather than being computed and happening to give the right answer.
function test_power_trivial(T::Type, sz, algs; test_rejected_input = false, kwargs...)
    R = real(eltype(T))
    summary_str = testargs_summary(T, sz)
    return @testset "power trivial exponents algorithm $alg $summary_str" for alg in algs
        A = instantiate_offaxis_matrix(T, sz)
        Id = one!(deepcopy(A))

        @testset "integer p = $p" for p in (0, 1)
            powA = @testinferred power(A, p, alg)
            @test powA == (iszero(p) ? Id : A)
            # in-place, where the output aliases the input
            @test power!(deepcopy(A), p, alg) == (iszero(p) ? Id : A)
        end
        @testset "float p = $p" for p in (zero(R), one(R))
            powA = @testinferred power(A, p, alg)
            @test powA == (iszero(p) ? Id : A)
        end

        if test_rejected_input
            B = instantiate_smallnorm_matrix(T, sz)
            IdB = one!(deepcopy(B))
            # confirm the premise: this algorithm cannot decompose `B`
            @test_throws DomainError power(B, 2, alg)
            # yet the trivial exponents never reach the decomposition
            @test power(B, 0, alg) == IdB
            @test power(B, 1, alg) == B
        end
    end
end

# See the corresponding comment in `squareroot.jl` for `exact_hermiticity`.
function test_power_hermitian(
        T::Type, sz, algs;
        exact_hermiticity = true, test_spectrum = true, kwargs...
    )
    R = real(eltype(T))
    summary_str = testargs_summary(T, sz)
    return @testset "power hermitian algorithm $alg $summary_str" for alg in algs
        A = instantiate_posdef_matrix(T, sz)
        Ac = deepcopy(A)

        check = function (powA, p)
            @test eltype(powA) == eltype(A)
            @test A == Ac
            if exact_hermiticity
                @test ishermitian(powA)
                # `power` maps the spectrum elementwise, up to the reordering a negative
                # exponent induces
                test_spectrum && @test eigh_vals(powA) ≈ sort(eigh_vals(A) .^ p)
            else
                @test ishermitian(powA; rtol = precision(T))
            end
            return nothing
        end

        @testset "integer p = $p" for p in (2, -1)
            check(@testinferred(power(A, p, alg)), p)
        end
        @testset "fractional p = $q" for q in (1 // 2, -1 // 2)
            p = convert(R, q)
            check(@testinferred(power(A, p, alg)), p)
        end
    end
end

# Domain handling. Integer exponents are defined for any square matrix and must ignore the
# domain entirely; fractional exponents are principal powers and must respect it. Unlike
# `logarithm`, clamping a roundoff-negative eigenvalue onto zero is harmless for a positive
# fractional exponent, since `0^p` is well defined there.
#
# `hermitian_output = true` and `supports_domain_atol = false`: see `squareroot.jl` and
# `logarithm.jl`. `test_singular = true` only where the zero eigenvalue is exactly representable
# (the `Diagonal` path); the `eig`-based kernels test `any(iszero, λ)` on *computed* eigenvalues,
# which a numerically singular matrix does not satisfy.
function test_power_domain(
        T::Type, sz, algs;
        hermitian_output = false, supports_domain_atol = true, test_singular = false, kwargs...
    )
    R = real(eltype(T))
    n = sz isa Tuple ? first(sz) : sz
    half = one(R) / 2
    summary_str = testargs_summary(T, sz)
    return @testset "power domain algorithm $alg $summary_str" for alg in algs
        # eigenvalue on the negative real axis
        λ = collect(R, 1:n)
        λ[1] = -one(R)
        A = instantiate_hermitian_spectrum(T, sz, λ)

        # integer exponents are unaffected by the domain
        @test power(A, 2, alg) ≈ A * A
        @test power(A, 3, alg) ≈ A * A * A

        if eltype(T) <: Real || hermitian_output
            @test_throws DomainError power(A, half, alg)
        else
            powA = @testinferred power(A, half, alg)
            @test powA * powA ≈ A
        end

        supports_domain_atol || continue

        # roundoff-scale negative eigenvalue: clamped onto the boundary, which is fine for p > 0
        λtiny = collect(R, 1:n)
        λtiny[1] = -10 * eps(R)
        Atiny = instantiate_hermitian_spectrum(T, sz, λtiny)
        powAtiny = @testinferred power(Atiny, half, alg)
        @test eltype(powAtiny) == eltype(Atiny)
        @test powAtiny * powAtiny ≈ Atiny atol = sqrt(eps(R))

        # a negative fractional exponent additionally requires a nonzero spectrum
        λzero = collect(R, 1:n)
        λzero[1] = zero(R)
        Azero = instantiate_hermitian_spectrum(T, sz, λzero)
        @test_throws DomainError power(Azero, -half, alg)
        test_singular && @test_throws SingularException power(Azero, -1, alg)
    end
end

# Cross-check against `LinearAlgebra`, which only applies to host arrays. Pass
# `test_hermitian = false` for generic eltypes, as `LinearAlgebra` has no matrix functions for a
# `Hermitian` wrapper outside the BLAS floats.
function test_power_reference(T::Type, sz; test_hermitian = true, kwargs...)
    R = real(eltype(T))
    summary_str = testargs_summary(T, sz)
    return @testset "power vs LinearAlgebra $summary_str" begin
        A = instantiate_offaxis_matrix(T, sz)
        @testset "integer p = $p" for p in (2, -1)
            @test power(A, p) ≈ A^p
        end
        @testset "fractional p = $q" for q in (1 // 2, -1 // 4)
            p = convert(R, q)
            @test power(A, p) ≈ A^p
        end

        if test_hermitian
            H = instantiate_posdef_matrix(T, sz)
            @test power(H, 2) ≈ LinearAlgebra.Hermitian(H)^2
            @test power(H, one(R) / 2) ≈ LinearAlgebra.Hermitian(H)^(one(R) / 2)
        end
    end
end
