using TestExtras

function test_sketching(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "sketching $summary_str" begin
        test_left_sketch(T, sz; kwargs...)
        test_right_sketch(T, sz; kwargs...)
    end
end

function test_left_sketch(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "left_sketch $summary_str" begin
        m, n = sz
        r = min(min(m, n) ÷ 4, 5)
        r > 0 || return

        A = instantiate_almost_rank_deficient_matrix(T, sz; trunc = truncrank(r), atol, rtol)
        Ac = deepcopy(A)
        k = min(r, m, n)

        # Does the elementary functionality work
        Q, B = @testinferred left_sketch(A; howmany = r)
        @test size(Q) == (m, k)
        @test eltype(Q) === float(eltype(T))
        @test isisometric(Q; rtol, atol)
        @test size(B) == (k, n)
        @test eltype(B) === float(eltype(T))
        @test B ≈ Q' * A atol = atol rtol = rtol
        @test A ≈ Q * B atol = atol rtol = rtol
        @test A == Ac

        # Can I pass in outputs
        Q, B = @testinferred left_sketch!(deepcopy(A), (Q, B); howmany = r)
        @test size(Q) == (m, k)
        @test eltype(Q) === float(eltype(T))
        @test isisometric(Q; rtol, atol)
        @test size(B) == (k, n)
        @test eltype(B) === float(eltype(T))
        @test B ≈ Q' * A atol = atol rtol = rtol
        @test A ≈ Q * B atol = atol rtol = rtol

        # Can I pass in keywords
        rng = MersenneTwister(3)
        Q, B = @testinferred left_sketch(A; howmany = r, rng)
        rng = MersenneTwister(3)
        Q′, B′ = @testinferred left_sketch(A; howmany = r, rng)
        @test Q == Q′
        @test B == B′

        # Can I pass in algorithms
        rng = MersenneTwister(3)
        alg = GaussianSketching(r; rng)
        Q′, B′ = @testinferred left_sketch(A, alg)
        @test Q == Q′
        @test B == B′

        # Do power iterations improve accuracy
        Aflat = instantiate_matrix(T, sz)
        Q1, B1 = left_sketch(Aflat, GaussianSketching(r; numiter = 1, rng = MersenneTwister(123)))
        Q5, B5 = left_sketch(Aflat, GaussianSketching(r; numiter = 5, rng = MersenneTwister(123)))
        e1 = norm(Aflat - Q1 * B1) / norm(Aflat)
        e5 = norm(Aflat - Q5 * B5) / norm(Aflat)
        @test e5 ≤ e1 + rtol
    end
end

function test_right_sketch(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "right_sketch $summary_str" begin
        m, n = sz
        r = min(min(m, n) ÷ 4, 5)
        r > 0 || return

        A = instantiate_almost_rank_deficient_matrix(T, sz; trunc = truncrank(r), atol, rtol)
        Ac = deepcopy(A)
        k = min(r, m, n)

        # Does the elementary functionality work
        B, Pᴴ = @testinferred right_sketch(A; howmany = r)
        @test size(B) == (m, k)
        @test eltype(B) === float(eltype(T))
        @test size(Pᴴ) == (k, n)
        @test eltype(Pᴴ) === float(eltype(T))
        @test isisometric(Pᴴ'; rtol, atol)
        @test B ≈ A * Pᴴ' atol = atol rtol = rtol
        @test A ≈ B * Pᴴ atol = atol rtol = rtol
        @test A == Ac

        # Can I pass in outputs
        B, Pᴴ = @testinferred right_sketch!(deepcopy(A), (B, Pᴴ); howmany = r)
        @test size(B) == (m, k)
        @test eltype(B) === float(eltype(T))
        @test size(Pᴴ) == (k, n)
        @test eltype(Pᴴ) === float(eltype(T))
        @test isisometric(Pᴴ'; rtol, atol)
        @test B ≈ A * Pᴴ' atol = atol rtol = rtol
        @test A ≈ B * Pᴴ atol = atol rtol = rtol

        # Can I pass in keywords
        rng = MersenneTwister(3)
        B, Pᴴ = @testinferred right_sketch(A; howmany = r, rng)
        rng = MersenneTwister(3)
        B′, Pᴴ′ = @testinferred right_sketch(A; howmany = r, rng)
        @test B == B′
        @test Pᴴ == Pᴴ′

        # Can I pass in algorithms
        rng = MersenneTwister(3)
        alg = GaussianSketching(r; rng)
        B′, Pᴴ′ = @testinferred right_sketch(A, alg)
        @test B == B′
        @test Pᴴ == Pᴴ′

        # Do power iterations improve accuracy
        Aflat = instantiate_matrix(T, sz)
        B1, P1 = right_sketch(Aflat, GaussianSketching(r; numiter = 1, rng = MersenneTwister(123)))
        B5, P5 = right_sketch(Aflat, GaussianSketching(r; numiter = 5, rng = MersenneTwister(123)))
        e1 = norm(Aflat - B1 * P1) / norm(Aflat)
        e5 = norm(Aflat - B5 * P5) / norm(Aflat)
        @test e5 ≤ e1 + rtol
    end
end
