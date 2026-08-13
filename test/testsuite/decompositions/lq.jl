using TestExtras

function test_lq(T::Type, sz; test_null = true, kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "lq $summary_str" begin
        test_lq_compact(T, sz; kwargs...)
        test_lq_full(T, sz; kwargs...)
        test_null && test_lq_null(T, sz; kwargs...)
        test_lq_inplaceQ(T, sz; kwargs...)
    end
end

function test_lq_algs(T::Type, sz, algs; test_null = true, kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "lq algorithms $summary_str" begin
        test_lq_compact_algs(T, sz, algs; kwargs...)
        test_lq_full_algs(T, sz, algs; kwargs...)
        test_null && test_lq_null_algs(T, sz, algs; kwargs...)
    end
end

function test_lq_compact(
        T::Type, sz;
        test_positive = true, test_pivoted = true, test_blocksize = true,
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "lq_compact! $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)

        # does the elementary functionality work
        L, Q = @testinferred lq_compact(A)
        @test L * Q ≈ A
        @test isisometric(Q; side = :right, atol, rtol)
        @test istril(L)
        @test A == Ac

        # can I pass in outputs?
        L2, Q2 = @testinferred lq_compact!(deepcopy(A), (L, Q))
        @test L2 * Q2 ≈ A
        @test isisometric(Q2; side = :right, atol, rtol)
        @test istril(L2)

        # do we support `positive = true`?
        if test_positive
            Lpos, Qpos = @testinferred lq_compact(A; positive = true)
            @test Lpos * Qpos ≈ A
            @test isisometric(Qpos; side = :right, atol, rtol)
            @test istril(Lpos)
            @test has_positive_diagonal(Lpos)
        else
            @test_throws Exception lq_compact(A; positive = true)
        end

        # do we support `pivoted = true`?
        if test_pivoted
            Lpiv, Qpiv = @testinferred lq_compact(A; pivoted = true)
            @test Lpiv * Qpiv ≈ A
            @test isisometric(Qpiv; side = :right, atol, rtol)
            # pivoted AND blocksize not yet supported
            @test_throws ArgumentError lq_compact(A; pivoted = true, blocksize = 2)
        else
            @test_throws Exception lq_compact(A; pivoted = true)
        end

        # do we support `blocksize = Int`?
        if test_blocksize
            Lblocked, Qblocked = @testinferred lq_compact(A; blocksize = 2)
            @test Lblocked * Qblocked ≈ A
            @test isisometric(Qblocked; side = :right, atol, rtol)
        else
            @test_throws Exception lq_compact(A; blocksize = 2)
        end
    end
end

# test using `A` itself as output for `Q`
function test_lq_inplaceQ(
        T::Type, sz;
        test_inplaceQ = true, test_positive = true,
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    (sz isa Tuple && length(sz) == 2) || return nothing
    m, n = sz
    n >= m || return nothing # inplace Q requires a wide matrix
    summary_str = testargs_summary(T, sz)
    return @testset "lq_compact! inplace Q $summary_str" begin
        A = instantiate_matrix(T, sz)
        L = similar(A, (m, m))

        if !test_inplaceQ
            Ain = deepcopy(A)
            @test_throws Exception lq_compact!(Ain, (L, Ain))
        else
            for positive in (test_positive ? (false, true) : (false,))
                Ain = deepcopy(A)
                L2, Q = lq_compact!(Ain, (L, Ain); positive)
                @test Q === Ain
                @test L2 * Q ≈ A
                @test isisometric(Q; side = :right, atol, rtol)
                @test istril(L2)
                if positive
                    @test has_positive_diagonal(L2)
                end

                # L is not required
                Ain2 = deepcopy(A)
                _, Q2 = lq_compact!(Ain2, (similar(A, (0, 0)), Ain2); positive)
                @test Q2 === Ain2
                @test Q2 ≈ Q
            end

            if m == n
                Ain = deepcopy(A)
                Lf, Q = lq_full!(Ain, (similar(A, (m, n)), Ain))
                @test Q === Ain
                @test Lf * Q ≈ A
                @test isunitary(Q; atol, rtol)
                @test has_positive_diagonal(Lf)
            end

            # blocked algorithm and aliased L are not supported
            Ain = deepcopy(A)
            @test_throws ArgumentError lq_compact!(Ain, (L, Ain); blocksize = 2)
            Ain = deepcopy(A)
            @test_throws ArgumentError lq_compact!(Ain, (view(Ain, :, 1:m), Ain))
        end
    end
end

function test_lq_compact_algs(
        T::Type, sz, algs;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "lq_compact! algorithm $alg $summary_str" for alg in algs
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)

        # does the elementary functionality work
        L, Q = @testinferred lq_compact(A; alg)
        @test L * Q ≈ A
        @test isisometric(Q; side = :right, atol, rtol)
        @test A == Ac
        if !is_pivoted(alg)
            @test istril(L)
            if is_positive(alg)
                @test has_positive_diagonal(L)
            end
        end

        # can I pass in outputs?
        L2, Q2 = @testinferred lq_compact!(Ac, (L, Q); alg)
        @test L2 * Q2 ≈ A
        @test isisometric(Q2; side = :right, atol, rtol)
        if !is_pivoted(alg)
            @test istril(L2)
            if is_positive(alg)
                @test has_positive_diagonal(L2)
            end
        end
    end
end

function test_lq_full(
        T::Type, sz;
        test_positive = true, test_pivoted = true, test_blocksize = true,
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "lq_full! $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)

        # does the elementary functionality work
        L, Q = @testinferred lq_full(A)
        @test L * Q ≈ A
        @test isunitary(Q; atol, rtol)
        @test istril(L)
        @test A == Ac

        # can I pass in outputs?
        L2, Q2 = @testinferred lq_full!(deepcopy(A), (L, Q))
        @test L2 * Q2 ≈ A
        @test isunitary(Q2; atol, rtol)
        @test istril(L2)

        # do we support `positive = true`?
        if test_positive
            Lpos, Qpos = @testinferred lq_full(A; positive = true)
            @test Lpos * Qpos ≈ A
            @test isunitary(Qpos; atol, rtol)
            @test istril(Lpos)
            @test has_positive_diagonal(Lpos)
        else
            @test_throws Exception lq_full(A; positive = true)
        end

        # do we support `pivoted = true`?
        if test_pivoted
            Lpiv, Qpiv = @testinferred lq_full(A; pivoted = true)
            @test Lpiv * Qpiv ≈ A
            @test isunitary(Qpos; atol, rtol)
            # pivoted AND blocksize not yet supported
            @test_throws ArgumentError lq_full(A; pivoted = true, blocksize = 2)
        else
            @test_throws Exception lq_full(A; pivoted = true)
        end

        # do we support `blocksize = Int`?
        if test_blocksize
            Lblocked, Qblocked = @testinferred lq_full(A; blocksize = 2)
            @test Lblocked * Qblocked ≈ A
            @test isunitary(Qblocked; atol, rtol)
        else
            @test_throws Exception lq_full(A; blocksize = 2)
        end
    end
end

function test_lq_full_algs(
        T::Type, sz, algs;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "lq_full! algorithm $alg $summary_str" for alg in algs
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)

        # does the elementary functionality work
        L, Q = @testinferred lq_full(A; alg)
        @test L * Q ≈ A
        @test isisometric(Q; side = :right, atol, rtol)
        @test A == Ac
        if !is_pivoted(alg)
            @test istril(L)
            if is_positive(alg)
                @test has_positive_diagonal(L)
            end
        end

        # can I pass in outputs?
        L2, Q2 = @testinferred lq_full!(Ac, (L, Q); alg)
        @test L2 * Q2 ≈ A
        @test isisometric(Q2; side = :right, atol, rtol)
        if !is_pivoted(alg)
            @test istril(L2)
            if is_positive(alg)
                @test has_positive_diagonal(L2)
            end
        end
    end
end

function test_lq_null(
        T::Type, sz;
        test_pivoted = true, test_blocksize = true,
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "lq_null! $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)

        # does the elementary functionality work
        Nᴴ = @testinferred lq_null(A)
        @test isrightnull(Nᴴ, A; atol, rtol)
        @test isisometric(Nᴴ; side = :right, atol, rtol)
        @test A == Ac

        # can I pass in outputs?
        Nᴴ2 = @testinferred lq_null!(Ac, Nᴴ)
        @test isrightnull(Nᴴ2, A; atol, rtol)
        @test isisometric(Nᴴ2; side = :right, atol, rtol)

        # do we support `pivoted = true`?
        if test_pivoted
            Nᴴpiv = @testinferred lq_null(A; pivoted = true)
            @test isrightnull(Nᴴpiv, A; atol, rtol)
            @test isisometric(Nᴴpiv; side = :right, atol, rtol)
        else
            @test_throws Exception lq_null(A; pivoted = true)
        end

        # do we support `blocksize = Int`?
        if test_blocksize
            Nᴴblocked = @testinferred lq_null(A; blocksize = 2)
            @test isrightnull(Nᴴblocked, A; atol, rtol)
            @test isisometric(Nᴴblocked; side = :right, atol, rtol)
        else
            @test_throws Exception lq_null(A; blocksize = 2)
        end
    end
end

function test_lq_null_algs(
        T::Type, sz, algs;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "lq_null! algorithm $alg $summary_str" for alg in algs
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)

        # does the elementary functionality work
        Nᴴ = @testinferred lq_null(A; alg)
        @test isrightnull(Nᴴ, A; atol, rtol)
        @test isisometric(Nᴴ; side = :right, atol, rtol)
        @test A == Ac

        # can I pass in outputs?
        Nᴴ2 = @testinferred lq_null!(Ac, Nᴴ; alg)
        @test isrightnull(Nᴴ2, A; atol, rtol)
        @test isisometric(Nᴴ2; side = :right, atol, rtol)
    end
end
