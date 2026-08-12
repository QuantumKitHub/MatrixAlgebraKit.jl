using TestExtras

function test_qr(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "qr $summary_str" begin
        test_qr_compact(T, sz; kwargs...)
        test_qr_full(T, sz; kwargs...)
        test_qr_null(T, sz; kwargs...)
        test_qr_inplaceQ(T, sz; kwargs...)
    end
end

function test_qr_algs(T::Type, sz, algs; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "qr algorithms $summary_str" begin
        test_qr_compact_algs(T, sz, algs; kwargs...)
        test_qr_full_algs(T, sz, algs; kwargs...)
        test_qr_null_algs(T, sz, algs; kwargs...)
    end
end

# test correctness and interface for QR regardless of algorithm
function test_qr_compact(
        T::Type, sz;
        test_positive = true, test_pivoted = true, test_blocksize = true,
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "qr_compact! $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)

        # does the elementary functionality work
        Q, R = @testinferred qr_compact(A)
        @test Q * R ≈ A
        @test isisometric(Q; atol, rtol)
        @test istriu(R)
        @test A == Ac

        # can I pass in outputs?
        Q2, R2 = @testinferred qr_compact!(deepcopy(A), (Q, R))
        @test Q2 * R2 ≈ A
        @test isisometric(Q2; atol, rtol)
        @test istriu(R2)

        # do we support `positive = true`?
        if test_positive
            Qpos, Rpos = @testinferred qr_compact(A; positive = true)
            @test Qpos * Rpos ≈ A
            @test isisometric(Qpos; atol, rtol)
            @test istriu(Rpos)
            @test has_positive_diagonal(Rpos)
        else
            @test_throws Exception qr_compact(A; positive = true)
        end

        # do we support `pivoted = true`?
        if test_pivoted
            Qpiv, Rpiv = @testinferred qr_compact(A; pivoted = true)
            @test Qpiv * Rpiv ≈ A
            @test isisometric(Qpos; atol, rtol)
            # pivoted AND blocksize not yet supported
            @test_throws ArgumentError qr_compact(A; pivoted = true, blocksize = 2)
        else
            @test_throws Exception qr_compact(A; pivoted = true)
        end

        # do we support `blocksize = Int`?
        if test_blocksize
            Qblocked, Rblocked = @testinferred qr_compact(A; blocksize = 2)
            @test Qblocked * Rblocked ≈ A
            @test isisometric(Qblocked; atol, rtol)
        else
            @test_throws Exception qr_compact(A; blocksize = 2)
        end
    end
end

# test various algorithms
function test_qr_compact_algs(
        T::Type, sz, algs;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "qr_compact! algorithm $alg $summary_str" for alg in algs
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)

        # does the elementary functionality work
        Q, R = @testinferred qr_compact(A; alg)
        @test Q * R ≈ A
        @test isisometric(Q; atol, rtol)
        @test A == Ac
        if !is_pivoted(alg)
            @test istriu(R)
            if is_positive(alg)
                @test has_positive_diagonal(R)
            end
        end

        # can I pass in outputs?
        Q2, R2 = @testinferred qr_compact!(Ac, (Q, R); alg)
        @test Q2 * R2 ≈ A
        @test isisometric(Q2; atol, rtol)
        if !is_pivoted(alg)
            @test istriu(R2)
            if is_positive(alg)
                @test has_positive_diagonal(R2)
            end
        end
    end
end

# test using `A` itself as output for `Q`
function test_qr_inplaceQ(
        T::Type, sz;
        test_inplaceQ = true, test_positive = true, test_pivoted = true,
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    (sz isa Tuple && length(sz) == 2) || return nothing
    m, n = sz
    m >= n || return nothing # inplace Q requires a tall matrix
    summary_str = testargs_summary(T, sz)
    return @testset "qr_compact! inplace Q $summary_str" begin
        A = instantiate_matrix(T, sz)
        R = similar(A, (n, n))

        if !test_inplaceQ
            Ain = deepcopy(A)
            @test_throws Exception qr_compact!(Ain, (Ain, R))
        else
            for positive in (test_positive ? (false, true) : (false,)),
                    pivoted in (test_pivoted ? (false, true) : (false,))

                Ain = deepcopy(A)
                Q, R2 = qr_compact!(Ain, (Ain, R); positive, pivoted)
                @test Q === Ain
                @test Q * R2 ≈ A
                @test isisometric(Q; atol, rtol)
                if !pivoted
                    @test istriu(R2)
                    if positive
                        @test has_positive_diagonal(R2)
                    end
                end

                # R is not required
                Ain2 = deepcopy(A)
                Q2, = qr_compact!(Ain2, (Ain2, similar(A, (0, 0))); positive, pivoted)
                @test Q2 === Ain2
                @test Q2 ≈ Q
            end

            if m == n
                Ain = deepcopy(A)
                Q, Rf = qr_full!(Ain, (Ain, similar(A, (m, n))))
                @test Q === Ain
                @test Q * Rf ≈ A
                @test isunitary(Q; atol, rtol)
                @test has_positive_diagonal(Rf)
            end

            # blocked algorithm and aliased R are not supported
            Ain = deepcopy(A)
            @test_throws ArgumentError qr_compact!(Ain, (Ain, R); blocksize = 2)
            Ain = deepcopy(A)
            @test_throws ArgumentError qr_compact!(Ain, (Ain, view(Ain, 1:n, :)))
        end
    end
end

function test_qr_full(
        T::Type, sz;
        test_positive = true, test_pivoted = true, test_blocksize = true,
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "qr_full! $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)

        # does the elementary functionality work
        Q, R = @testinferred qr_full(A)
        @test Q * R ≈ A
        @test isunitary(Q; atol, rtol)
        @test istriu(R)
        @test A == Ac

        # can I pass in outputs?
        Q2, R2 = @testinferred qr_full!(deepcopy(A), (Q, R))
        @test Q2 * R2 ≈ A
        @test isunitary(Q2; atol, rtol)
        @test istriu(R2)

        # do we support `positive = true`?
        if test_positive
            Qpos, Rpos = @testinferred qr_full(A; positive = true)
            @test Qpos * Rpos ≈ A
            @test isunitary(Qpos; atol, rtol)
            @test istriu(Rpos)
            @test has_positive_diagonal(Rpos)
        else
            @test_throws Exception qr_full(A; positive = true)
        end

        # do we support `pivoted = true`?
        if test_pivoted
            Qpiv, Rpiv = @testinferred qr_full(A; pivoted = true)
            @test Qpiv * Rpiv ≈ A
            @test isunitary(Qpos; atol, rtol)
            # pivoted AND blocksize not yet supported
            @test_throws ArgumentError qr_full(A; pivoted = true, blocksize = 2)
        else
            @test_throws Exception qr_full(A; pivoted = true)
        end

        # do we support `blocksize = Int`?
        if test_blocksize
            Qblocked, Rblocked = @testinferred qr_full(A; blocksize = 2)
            @test Qblocked * Rblocked ≈ A
            @test isunitary(Qblocked; atol, rtol)
        else
            @test_throws Exception qr_full(A; blocksize = 2)
        end
    end
end

function test_qr_full_algs(
        T::Type, sz, algs;
        test_blocksize = true,
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "qr_full! algorithm $alg $summary_str" for alg in algs
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)

        # does the elementary functionality work
        Q, R = @testinferred qr_full(A; alg)
        @test Q * R ≈ A
        @test isunitary(Q; atol, rtol)
        @test A == Ac
        if !is_pivoted(alg)
            @test istriu(R)
            if is_positive(alg)
                @test has_positive_diagonal(R)
            end
        end

        # can I pass in outputs?
        Q2, R2 = @testinferred qr_full!(Ac, (Q, R); alg)
        @test Q2 * R2 ≈ A
        @test isunitary(Q2; atol, rtol)
        if !is_pivoted(alg)
            @test istriu(R2)
            if is_positive(alg)
                @test has_positive_diagonal(R2)
            end
        end
    end
end

function test_qr_null(
        T::Type, sz;
        test_pivoted = true, test_blocksize = true,
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "qr_null! $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)

        # does the elementary functionality work
        N = @testinferred qr_null(A)
        @test isleftnull(N, A; atol, rtol)
        @test isisometric(N; atol, rtol)
        @test A == Ac

        # can I pass in outputs?
        N2 = @testinferred qr_null!(deepcopy(A), N)
        @test isleftnull(N2, A; atol, rtol)
        @test isisometric(N2; atol, rtol)

        # do we support `pivoted = true`?
        if test_pivoted
            Npiv = @testinferred qr_null(A; pivoted = true)
            @test isleftnull(Npiv, A; atol, rtol)
            @test isisometric(Npiv; atol, rtol)
        else
            @test_throws Exception qr_null(A; pivoted = true)
        end

        # do we support `blocksize = Int`?
        if test_blocksize
            Nblocked = @testinferred qr_null(A; blocksize = 2)
            @test isleftnull(Nblocked, A; atol, rtol)
            @test isisometric(Nblocked; atol, rtol)
        else
            @test_throws Exception qr_null(A; blocksize = 2)
        end
    end
end

function test_qr_null_algs(
        T::Type, sz, algs;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "qr_null! algorithm $alg $summary_str" for alg in algs
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)

        # does the elementary functionality work
        N = @testinferred qr_null(A; alg)
        @test isleftnull(N, A; atol, rtol)
        @test isisometric(N; atol, rtol)
        @test A == Ac

        # can I pass in outputs?
        N2 = @testinferred qr_null!(Ac, N; alg)
        @test isleftnull(N2, A; atol, rtol)
        @test isisometric(N2; atol, rtol)
    end
end
