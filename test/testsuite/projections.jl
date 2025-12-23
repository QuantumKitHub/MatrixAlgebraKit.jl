using TestExtras
using MatrixAlgebraKit: ishermitian
using LinearAlgebra: Diagonal, normalize!
using GenericLinearAlgebra

function test_projections(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "projections $summary_str" begin
        test_project_antihermitian(T, sz; kwargs...)
        test_project_hermitian(T, sz; kwargs...)
        test_project_isometric(T, sz; kwargs...)
    end
end

function test_project_antihermitian(
        T::Type, sz;
        test_blocksize = true,
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "project_antihermitian! $summary_str" begin
        noisefactor = eps(real(eltype(T)))^(3 / 4)
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)
        Ah = (A + A') / 2
        Aa = (A - A') / 2

        # do we support `blocksize = Int`?
        if test_blocksize
            Ba = project_antihermitian(A; blocksize = 16)
            @test isantihermitian(Ba)
            @test Ba ≈ Aa
            @test A == Ac
            Ba_approx = Ba + noisefactor * Ah
            @test !isantihermitian(Ba_approx)
            # this is never anti-hermitian for real Diagonal: |A - A'| == 0
            @test isantihermitian(Ba_approx; rtol = 10 * noisefactor) || norm(Aa) == 0

            copy!(Ac, A)
            Ba = project_antihermitian!(Ac; blocksize = 16)
            @test Ba === Ac
            @test isantihermitian(Ba)
            @test Ba ≈ Aa
        end

        # test approximate error calculation
        A = normalize!(A)
        Ah = project_hermitian(A)
        Aa = project_antihermitian(A)

        Ah_approx = Ah + noisefactor * Aa
        ϵ = norm(project_antihermitian(Ah_approx))
        # this is never off-hermitian for real Diagonal: |A - A'| == 0
        @test !ishermitian(Ah_approx; atol = (999 // 1000) * ϵ) || norm(Aa) == 0
        @test ishermitian(Ah_approx; atol = (1001 // 1000) * ϵ)

        Aa_approx = Aa + noisefactor * Ah
        ϵ = norm(project_hermitian(Aa_approx))
        @test !isantihermitian(Aa_approx; atol = (999 // 1000) * ϵ)
        @test isantihermitian(Aa_approx; atol = (1001 // 1000) * ϵ)
    end
end

function test_project_hermitian(
        T::Type, sz;
        test_blocksize = true,
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "project_hermitian! $summary_str" begin
        noisefactor = eps(real(eltype(T)))^(3 / 4)
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)

        # do we support `blocksize = Int`?
        if test_blocksize
            Ah = (A + A') / 2
            Aa = (A - A') / 2

            Bh = project_hermitian(A; blocksize = 16)
            @test ishermitian(Bh)
            @test Bh ≈ Ah
            @test A == Ac
            Bh_approx = Bh + noisefactor * Aa
            # this is still hermitian for real Diagonal: |A - A'| == 0
            @test !ishermitian(Bh_approx) || norm(Aa) == 0
            @test ishermitian(Bh_approx; rtol = 10 * noisefactor)

            Bh = project_hermitian!(Ac; blocksize = 16)
            @test Bh === Ac
            @test ishermitian(Bh)
            @test Bh ≈ Ah
        end

        # test approximate error calculation
        A = normalize!(A)
        Ah = project_hermitian(A)
        Aa = project_antihermitian(A)

        Ah_approx = Ah + noisefactor * Aa
        ϵ = norm(project_antihermitian(Ah_approx))
        # this is still hermitian for real Diagonal: |A - A'| == 0
        @test !ishermitian(Ah_approx; atol = (999 // 1000) * ϵ) || norm(Aa) == 0
        @test ishermitian(Ah_approx; atol = (1001 // 1000) * ϵ)

        Aa_approx = Aa + noisefactor * Ah
        ϵ = norm(project_hermitian(Aa_approx))
        @test !isantihermitian(Aa_approx; atol = (999 // 1000) * ϵ)
        @test isantihermitian(Aa_approx; atol = (1001 // 1000) * ϵ)
    end
end

function test_project_isometric(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "project_isometric! $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)
        k = min(size(A)...)
        W = project_isometric(A)
        @test isisometric(W)
        W2 = project_isometric(W)
        @test W2 ≈ W # stability of the projection
        @test W * (W' * A) ≈ A

        W2 = @testinferred project_isometric!(Ac, W)
        @test W2 === W
        @test isisometric(W)

        # test that W is closer to A then any other isometry
        for k in 1:10
            δA = instantiate_matrix(T, sz)
            W = project_isometric(A)
            W2 = project_isometric(A + δA / 100)
            # must be ≥ for real Diagonal case
            @test norm(A - W2) ≥ norm(A - W)
        end
    end
end
