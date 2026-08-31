using TestExtras
using LinearAlgebra: LinearAlgebra, I
using MatrixAlgebraKit: ishermitian

# `exp(A) * exp(-A) ≈ I` holds for every algorithm and backend, since `A` and `-A` commute

function test_exponential(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "exponential $summary_str" begin
        A = instantiate_smallnorm_matrix(T, sz)
        Ac = deepcopy(A)

        expA = @testinferred exponential(A)
        @test eltype(expA) == eltype(A)
        @test expA * exponential(-A) ≈ I
        @test A == Ac

        # the in-place method may not be able to reuse the provided output
        expA2 = @testinferred exponential!(deepcopy(A), deepcopy(expA))
        @test expA2 ≈ expA
    end
end

function test_exponential_algs(T::Type, sz, algs; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "exponential algorithm $alg $summary_str" for alg in algs
        A = instantiate_smallnorm_matrix(T, sz)
        Ac = deepcopy(A)

        expA = @testinferred exponential(A, alg)
        @test eltype(expA) == eltype(A)
        @test expA * exponential(-A, alg) ≈ I
        @test A == Ac
    end
end

# the scaled entrypoint `exponential((τ, A))` computes `exp(τ * A)`
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

# `exp(A)` of a real hermitian `A` is built as a symmetric product and is hermitian to the last bit;
# the complex case goes through `V exp(D) V'` and is hermitian only up to roundoff
function test_exponential_hermitian(
        T::Type, sz, algs;
        exact_hermiticity = eltype(T) <: Real, test_spectrum = true, kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "exponential hermitian algorithm $alg $summary_str" for alg in algs
        A = project_hermitian!(instantiate_smallnorm_matrix(T, sz))
        Ac = deepcopy(A)

        expA = @testinferred exponential(A, alg)
        @test eltype(expA) == eltype(A)
        @test expA * exponential(-A, alg) ≈ I
        @test A == Ac

        if exact_hermiticity
            @test ishermitian(expA)
            # `eigh_vals` rejects anything but an exactly hermitian matrix
            test_spectrum && @test eigh_vals(expA) ≈ exp.(eigh_vals(A))
        else
            @test ishermitian(expA; rtol = precision(T))
        end

        τ = randn(rng, real(eltype(T)))
        expτA = @testinferred exponential((τ, A), alg)
        @test expτA ≈ exponential(τ * A, alg)
        exact_hermiticity && @test ishermitian(expτA)
    end
end

# `MatrixFunctionViaTaylor` is the only algorithm that applies to a general matrix on device and at
# arbitrary precision. Its balancing step is exercised with a badly-scaled similarity transform
# `Aᵢⱼ ← Aᵢⱼ sᵢ / sⱼ`; dense input only, as that would densify a `Diagonal`.
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
            expA = @testinferred exponential(A, alg)
            @test eltype(expA) == eltype(A)
            @test expA * exponential(-A, alg) ≈ I

            # `Abad` is too ill-conditioned for the inverse check, but balancing must not
            # change the result
            for M in (A, Abad)
                expM = @testinferred exponential(M, alg)
                @test eltype(expM) == eltype(M)
                @test expM ≈ exponential(M, MatrixFunctionViaTaylor(; balance = !balance))
            end
        end
    end
end

# `MatrixFunctionViaEigh` requires hermitian input, and says so rather than silently projecting
function test_exponential_domain(T::Type, sz, algs; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "exponential domain algorithm $alg $summary_str" for alg in algs
        A = instantiate_smallnorm_matrix(T, sz)
        @test_throws DomainError exponential(A, alg)
    end
end

# the kernels must not assume a strided layout
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

# cross-check against `LinearAlgebra`, host arrays only. `LinearAlgebra` has no matrix functions
# for a `Hermitian` wrapper outside the BLAS floats, hence `test_hermitian`.
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
