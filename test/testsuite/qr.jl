function test_qr(::Type{T}, sz; kwargs...) where {T}
    summary_str = testargs_summary(T, sz)
    return @testset "qr $summary_str" begin
        test_qr_compact(T, sz; kwargs...)
        test_qr_full(T, sz; kwargs...)
        test_qr_null(T, sz; kwargs...)
    end
end

function test_qr_compact(
        ::Type{T}, sz;
        test_positive = true, test_pivoted = true, test_blocksize = true, kwargs...
    ) where {T <: Number}
    summary_str = testargs_summary(T, sz)
    return @testset "qr_compact! $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)

        # does the elementary functionality work
        Q, R = @testinferred qr_compact(A)
        @test Q * R ≈ A
        @test isisometric(Q)
        @test istriu(R)
        @test A == Ac

        # can I pass in outputs?
        Q2, R2 = @testinferred qr_compact!(deepcopy(A), (Q, R))
        @test Q2 * R2 ≈ A
        @test isisometric(Q2)
        @test istriu(R2)

        # do we support `positive = true`?
        if test_positive
            Qpos, Rpos = @testinferred qr_compact(A; positive = true)
            @test Qpos * Rpos ≈ A
            @test isisometric(Qpos)
            @test istriu(Rpos)
            @test has_positive_diagonal(Rpos)
        else
            @test_throws ArgumentError qr_compact(A; positive = true)
        end

        # do we support `pivoted = true`?
        if test_pivoted
            Qpiv, Rpiv = @testinferred qr_compact(A; pivoted = true)
            @test Qpiv * Rpiv ≈ A
            @test isisometric(Qpos)
        else
            @test_throws ArgumentError qr_compact(A; pivoted = true)
        end

        # do we support `blocksize = Int`?
        if test_blocksize
            Qblocked, Rblocked = @testinferred qr_compact(A; blocksize = 2)
            @test Qblocked * Rblocked ≈ A
            @test isisometric(Qblocked)
        else
            @test_throws ArgumentError qr_compact(A; blocksize = 2)
        end
    end
end

function test_qr_full(
        ::Type{T}, sz;
        test_positive = true, test_pivoted = true, test_blocksize = true, kwargs...
    ) where {T <: Number}
    summary_str = testargs_summary(T, sz)
    return @testset "qr_full! $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)

        # does the elementary functionality work
        Q, R = @testinferred qr_full(A)
        @test Q * R ≈ A
        @test isunitary(Q)
        @test istriu(R)
        @test A == Ac

        # can I pass in outputs?
        Q2, R2 = @testinferred qr_full!(deepcopy(A), (Q, R))
        @test Q2 * R2 ≈ A
        @test isunitary(Q2)
        @test istriu(R2)

        # do we support `positive = true`?
        if test_positive
            Qpos, Rpos = @testinferred qr_full(A; positive = true)
            @test Qpos * Rpos ≈ A
            @test isunitary(Qpos)
            @test istriu(Rpos)
            @test has_positive_diagonal(Rpos)
        else
            @test_throws ArgumentError qr_full(A; positive = true)
        end

        # do we support `pivoted = true`?
        if test_pivoted
            Qpiv, Rpiv = @testinferred qr_full(A; pivoted = true)
            @test Qpiv * Rpiv ≈ A
            @test isunitary(Qpos)
        else
            @test_throws ArgumentError qr_full(A; pivoted = true)
        end

        # do we support `blocksize = Int`?
        if test_blocksize
            Qblocked, Rblocked = @testinferred qr_full(A; blocksize = 2)
            @test Qblocked * Rblocked ≈ A
            @test isunitary(Qblocked)
        else
            @test_throws ArgumentError qr_full(A; blocksize = 2)
        end
    end
end

function test_qr_null(
        ::Type{T}, sz;
        test_pivoted = true, test_blocksize = true, kwargs...
    ) where {T <: Number}
    summary_str = testargs_summary(T, sz)
    return @testset "qr_null! $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)

        # does the elementary functionality work
        N = @testinferred qr_null(A)
        @test isleftnull(N, A)
        @test isisometric(N)
        @test A == Ac

        # can I pass in outputs?
        N2 = @testinferred qr_null!(deepcopy(A), N)
        @test isleftnull(N2, A)
        @test isisometric(N2)

        # do we support `pivoted = true`?
        if test_pivoted
            Npiv = @testinferred qr_null(A; pivoted = true)
            @test isleftnull(Npiv, A)
            @test isisometric(Npiv)
        else
            @test_throws ArgumentError qr_null(A; pivoted = true)
        end

        # do we support `blocksize = Int`?
        if test_blocksize
            Nblocked = @testinferred qr_null(A; blocksize = 2)
            @test isleftnull(Nblocked, A)
            @test isisometric(Nblocked)
        else
            @test_throws ArgumentError qr_null(A; blocksize = 2)
        end
    end
end
