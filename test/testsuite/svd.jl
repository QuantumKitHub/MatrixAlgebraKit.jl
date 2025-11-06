using TestExtras

function test_svd(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "svd $summary_str" begin
        test_svd_compact(T, sz; kwargs...)
        test_svd_full(T, sz; kwargs...)
        if min(sz...) > 0 && (T <: Number || T <: Diagonal{<:Number, <:Vector})
            test_svd_trunc(T, sz; kwargs...)
        end
    end
end

function test_svd_compact(
        T::Type, sz;
        test_blocksize = true,
        atol::Real = 0, rtol::Real = precision(eltype(T)),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "svd_compact! $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)
        m, n = size(A)
        minmn = min(m, n)
        if VERSION < v"1.11"
            # This is type unstable on older versions of Julia.
            U, S, Vᴴ = svd_compact(A)
        else
            U, S, Vᴴ = @testinferred svd_compact(A)
        end
        @test size(U) == (m, minmn)
        @test S isa Diagonal{real(eltype(T))} && size(S) == (minmn, minmn)
        @test size(Vᴴ) == (minmn, n)
        @test U * S * Vᴴ ≈ A
        @test isisometric(U)
        @test isisometric(Vᴴ; side = :right)
        @test isposdef(S)

        Sc = similar(A, real(eltype(T)), min(m, n))
        U2, S2, V2ᴴ = @testinferred svd_compact!(Ac, (U, S, Vᴴ))
        @test U2 === U
        @test S2 === S
        @test V2ᴴ === Vᴴ
        @test U * S * Vᴴ ≈ A
        @test isisometric(U)
        @test isisometric(Vᴴ; side = :right)
        @test isposdef(S)

        Sd = @testinferred svd_vals(A)
        @test S ≈ Diagonal(Sd)
    end
end

function test_svd_full(
        T::Type, sz;
        test_blocksize = true,
        atol::Real = 0, rtol::Real = precision(eltype(T)),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "svd_full! $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)
        m, n = size(A)
        minmn = min(m, n)

        U, S, Vᴴ = svd_full(A)
        @test size(U) == (m, m)
        @test eltype(S) == real(eltype(T)) && size(S) == (m, n)
        @test size(Vᴴ) == (n, n)
        @test U * S * Vᴴ ≈ A
        @test isunitary(U)
        @test isunitary(Vᴴ)
        @test all(isposdef, diagview(S))

        U2, S2, V2ᴴ = @testinferred svd_full!(Ac, (U, S, Vᴴ))
        @test U2 === U
        @test S2 === S
        @test V2ᴴ === Vᴴ
        @test U * S * Vᴴ ≈ A
        @test isunitary(U)
        @test isunitary(Vᴴ)
        @test all(isposdef, diagview(S))

        Sc = similar(A, real(eltype(T)), min(m, n))
        Sc2 = svd_vals!(copy!(Ac, A), Sc)
        @test Sc === Sc2
        @test collect(diagview(S)) ≈ collect(Sc)
    end
end

function test_svd_trunc(
        T::Type, sz;
        test_blocksize = true,
        atol::Real = 0, rtol::Real = precision(eltype(T)),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "svd_trunc! $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)
        m, n = size(A)
        minmn = min(m, n)
        S₀ = svd_vals(A)
        r = minmn - 2

        U1, S1, V1ᴴ, ϵ1 = @testinferred svd_trunc(A; trunc = truncrank(r))
        @test length(diagview(S1)) == r
        @test diagview(S1) ≈ S₀[1:r]
        @test LinearAlgebra.opnorm(A - U1 * S1 * V1ᴴ) ≈ S₀[r + 1]
        # Test truncation error
        @test ϵ1 ≈ norm(view(S₀, (r + 1):minmn)) atol = atol

        s = 1 + sqrt(eps(real(eltype(T))))
        trunc = trunctol(; atol = s * S₀[r + 1])

        U2, S2, V2ᴴ, ϵ2 = @testinferred svd_trunc(A; trunc)
        @test length(diagview(S2)) == r
        @test U1 ≈ U2
        @test S1 ≈ S2
        @test V1ᴴ ≈ V2ᴴ
        @test ϵ2 ≈ norm(view(S₀, (r + 1):minmn)) atol = atol

        trunc = truncerror(; atol = s * norm(@view(S₀[(r + 1):end])))
        U3, S3, V3ᴴ, ϵ3 = @testinferred svd_trunc(A; trunc)
        @test length(diagview(S3)) == r
        @test U1 ≈ U3
        @test S1 ≈ S3
        @test V1ᴴ ≈ V3ᴴ
        @test ϵ3 ≈ norm(view(S₀, (r + 1):minmn)) atol = atol

        # TODO
        #=@testset "mix maxrank and tol" begin
            U = qr_compact(randn(rng, T, m, m))[1]
            S = Diagonal(T[0.9, 0.3, 0.1, 0.01])
            Vᴴ = qr_compact(randn(rng, T, m, m))[1]
            A = U * S * Vᴴ

            for trunc_fun in (
                    (rtol, maxrank) -> (; rtol, maxrank),
                    (rtol, maxrank) -> truncrank(maxrank) & trunctol(; rtol),
                )
                U1, S1, V1ᴴ, ϵ1 = svd_trunc(A; trunc = trunc_fun(0.2, 1))
                @test length(diagview(S1)) == 1
                @test diagview(S1) ≈ diagview(S)[1:1]

                U2, S2, V2ᴴ, ϵ2 = svd_trunc(A; trunc = trunc_fun(0.2, 3))
                @test length(diagview(S2)) == 2
                @test diagview(S2) ≈ diagview(S)[1:2]
            end
        end
        @testset "specify truncation algorithm" begin
            atol = sqrt(eps(real(eltype(T))))
            m = 4
            U = qr_compact(randn(rng, T, m, m))[1]
            S = Diagonal(real(eltype(T))[0.9, 0.3, 0.1, 0.01])
            Vᴴ = qr_compact(randn(rng, T, m, m))[1]
            A = U * S * Vᴴ
            alg = TruncatedAlgorithm(trunctol(; atol = 0.2))
            U2, S2, V2ᴴ, ϵ2 = @constinferred svd_trunc(A)
            @test diagview(S2) ≈ diagview(S)[1:2]
            @test ϵ2 ≈ norm(diagview(S)[3:4]) atol = atol
            @test_throws ArgumentError svd_trunc(A; trunc = (; maxrank = 2))
        end=#
    end
end
