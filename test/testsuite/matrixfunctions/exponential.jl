using TestExtras
using LinearAlgebra: LinearAlgebra, I
using MatrixAlgebraKit: ishermitian

# `exponential` has no restricted domain. Two invariants are used:
#
#   * `exp(A) * exp(-A) ≈ I`, which holds for every algorithm and every backend, since `A` and
#     `-A` commute. This is the primary check.
#   * `logarithm(exp(A)) ≈ A`, which additionally needs `A` inside the principal branch —
#     guaranteed by `instantiate_smallnorm_matrix`, as a spectral radius of at most one keeps
#     every eigenvalue's imaginary part well inside `(-π, π]`. Gated behind `test_roundtrip`,
#     since `logarithm` of a general matrix is not available on device.

function test_exponential(T::Type, sz; test_roundtrip = true, kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "exponential $summary_str" begin
        A = instantiate_smallnorm_matrix(T, sz)
        Ac = deepcopy(A)

        expA = @testinferred exponential(A)
        @test eltype(expA) == eltype(A)
        @test expA * exponential(-A) ≈ I
        @test A == Ac
        test_roundtrip && @test logarithm(expA) ≈ A

        # the in-place method may not be able to reuse the provided output
        expA2 = @testinferred exponential!(deepcopy(A), deepcopy(expA))
        @test expA2 ≈ expA
    end
end

function test_exponential_algs(T::Type, sz, algs; test_roundtrip = true, kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "exponential algorithm $alg $summary_str" for alg in algs
        A = instantiate_smallnorm_matrix(T, sz)
        Ac = deepcopy(A)

        expA = @testinferred exponential(A, alg)
        @test eltype(expA) == eltype(A)
        @test expA * exponential(-A, alg) ≈ I
        @test A == Ac
        test_roundtrip && @test logarithm(expA) ≈ A
    end
end

# The scaled entrypoint `exponential((τ, A))` computes `exp(τ * A)`.
function test_exponential_scaled(T::Type, sz, algs; kwargs...)
    R = real(eltype(T))
    summary_str = testargs_summary(T, sz)
    return @testset "exponential scaled algorithm $alg $summary_str" for alg in algs
        A = instantiate_smallnorm_matrix(T, sz)
        Ac = deepcopy(A)

        # both a scalar of the matrix eltype and a real one, to exercise promotion
        @testset "τ::$(typeof(τ))" for τ in (randn(rng, eltype(T)), randn(rng, R))
            expτA = @testinferred exponential((τ, A), alg)
            @test eltype(expτA) == eltype(A)
            @test expτA ≈ exponential(τ * A, alg)
            @test expτA * exponential((-τ, A), alg) ≈ I
            @test A == Ac
        end
    end
end

# See the corresponding comment in `squareroot.jl` for `exact_hermiticity`. `alg` is reused for
# the `logarithm` roundtrip, so this must not be called with `MatrixFunctionViaTaylor`, which
# only implements `exponential`.
function test_exponential_hermitian(
        T::Type, sz, algs;
        exact_hermiticity = true, test_spectrum = true, kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "exponential hermitian algorithm $alg $summary_str" for alg in algs
        A = project_hermitian!(instantiate_smallnorm_matrix(T, sz))
        Ac = deepcopy(A)

        expA = @testinferred exponential(A, alg)
        @test eltype(expA) == eltype(A)
        @test expA * exponential(-A, alg) ≈ I
        @test logarithm(expA, alg) ≈ A
        @test A == Ac

        # `exponential` forms the real case as a symmetric product `VexpD * transpose(VexpD)`,
        # which is hermitian to the last bit, but the complex case as a plain `VexpD * V'` with
        # no closing projection, so there it is only approximately hermitian. This differs from
        # `squareroot`, `logarithm` and `power`, which are exact for both scalar types.
        if exact_hermiticity && eltype(T) <: Real
            @test ishermitian(expA)
            test_spectrum && @test eigh_vals(expA) ≈ exp.(eigh_vals(A))
        else
            @test ishermitian(expA; rtol = precision(T))
        end
    end
end

# `MatrixFunctionViaTaylor` is a native scaling-and-squaring evaluation, so it is the only
# algorithm that applies to a general matrix on device and at arbitrary precision. Its balancing
# step is exercised with a badly-scaled similarity transform `Aᵢⱼ ← Aᵢⱼ sᵢ / sⱼ`. Dense input
# only, as the scaling would densify a `Diagonal`.
function test_exponential_taylor(T::Type, sz; kwargs...)
    R = real(eltype(T))
    summary_str = testargs_summary(T, sz)
    return @testset "exponential Taylor $summary_str" begin
        A = instantiate_smallnorm_matrix(T, sz)
        n = size(A, 1)
        s = similar(A, R, n)
        copyto!(s, exp10.(range(-R(3), R(3), length = n)))
        Abad = A .* s ./ transpose(s)

        @testset "balance = $balance" for balance in (true, false)
            alg = MatrixFunctionViaTaylor(; balance)
            # `Abad` is deliberately ill-conditioned, so `exp(A) * exp(-A) ≈ I` loses too many
            # digits to assert there; the balancing invariance below is the check that matters.
            expA = @testinferred exponential(A, alg)
            @test eltype(expA) == eltype(A)
            @test expA * exponential(-A, alg) ≈ I

            for M in (A, Abad)
                expM = @testinferred exponential(M, alg)
                @test eltype(expM) == eltype(M)
                # the balancing similarity must not change the result
                @test expM ≈ exponential(M, MatrixFunctionViaTaylor(; balance = !balance))
            end
        end
    end
end

# `MatrixFunctionViaEigh` requires hermitian input, and says so rather than silently projecting.
function test_exponential_domain(T::Type, sz, algs; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "exponential domain algorithm $alg $summary_str" for alg in algs
        A = instantiate_smallnorm_matrix(T, sz)
        @test_throws DomainError exponential(A, alg)
    end
end

# Input that is not a `Matrix`: the kernels must not assume a strided layout.
function test_exponential_wrappers(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "exponential non-Matrix input $summary_str" begin
        A = instantiate_smallnorm_matrix(T, sz)
        m = size(A, 1)
        expA = exponential(A)

        wrappers = (
            ("view", B -> view(B, :, :)),
            ("PermutedDimsArray", B -> PermutedDimsArray(permutedims(B), (2, 1))),
            ("ReshapedArray", B -> reshape(view(vec(B), 1:(m * m)), m, m)),
        )
        @testset "$name" for (name, wrap) in wrappers
            W = wrap(deepcopy(A))
            @test !(W isa Matrix)
            @test exponential!(W) ≈ expA
        end
    end
end

# Cross-check against `LinearAlgebra`, which only applies to host arrays. Pass
# `test_hermitian = false` for generic eltypes, as `LinearAlgebra` has no matrix functions for a
# `Hermitian` wrapper outside the BLAS floats.
function test_exponential_reference(T::Type, sz; test_hermitian = true, kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "exponential vs LinearAlgebra $summary_str" begin
        A = instantiate_smallnorm_matrix(T, sz)
        @test exponential(A) ≈ LinearAlgebra.exp(A)

        if test_hermitian
            H = project_hermitian!(instantiate_smallnorm_matrix(T, sz))
            @test exponential(H) ≈ LinearAlgebra.exp(LinearAlgebra.Hermitian(H))
        end
    end
end
