using TestExtras
using GenericLinearAlgebra
using MatrixAlgebraKit: TruncatedAlgorithm
using LinearAlgebra: I, opnorm

function test_eigh(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "eigh $summary_str" begin
        test_eigh_full(T, sz; kwargs...)
        test_eigh_trunc(T, sz; kwargs...)
    end
end

function test_eigh_algs(T::Type, sz, algs; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "eigh algorithms $summary_str" begin
        test_eigh_full_algs(T, sz, algs; kwargs...)
        test_eigh_trunc_algs(T, sz, algs; kwargs...)
    end
end

function test_eigh_full(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "eigh_full! $summary_str" begin
        A = project_hermitian!(instantiate_matrix(T, sz))
        Ac = deepcopy(A)

        D, V = @testinferred eigh_full(A)
        @test A * V ≈ V * D
        @test isunitary(V)
        @test all(isreal, D)

        D2, V2 = eigh_full!(Ac, (D, V))
        @test A * V2 ≈ V2 * D2

        D3 = @testinferred eigh_vals(A)
        @test D ≈ Diagonal(D3)
    end
end

function test_eigh_full_algs(
        T::Type, sz, algs;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "eigh_full! algorithm $alg $summary_str" for alg in algs
        A = project_hermitian!(instantiate_matrix(T, sz))
        Ac = deepcopy(A)

        D, V = @testinferred eigh_full(A; alg)
        @test A * V ≈ V * D
        @test isunitary(V)
        @test all(isreal, D)

        D2, V2 = eigh_full!(Ac, (D, V); alg)
        @test A * V2 ≈ V2 * D2

        D3 = @testinferred eigh_vals(A; alg)
        @test D ≈ Diagonal(D3)
    end
end

function test_eigh_trunc(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "eigh_trunc! $summary_str" begin
        A = instantiate_matrix(T, sz)
        A = A * A'
        A = project_hermitian!(A)
        Ac = deepcopy(A)

        m = size(A, 1)
        D₀ = collect(reverse(eigh_vals(A)))
        r = m - 2
        s = 1 + sqrt(eps(real(eltype(T))))
        atol = sqrt(eps(real(eltype(T))))
        # truncrank
        D1, V1, ϵ1 = @testinferred eigh_trunc(A; trunc = truncrank(r))
        @test length(diagview(D1)) == r
        @test isisometric(V1)
        @test A * V1 ≈ V1 * D1
        @test opnorm(A - V1 * D1 * V1') ≈ D₀[r + 1]
        @test ϵ1 ≈ norm(view(D₀, (r + 1):m)) atol = atol

        # trunctol
        trunc = trunctol(; atol = s * D₀[r + 1])
        D2, V2, ϵ2 = @testinferred eigh_trunc(A; trunc)
        @test length(diagview(D2)) == r
        @test isisometric(V2)
        @test A * V2 ≈ V2 * D2
        @test ϵ2 ≈ norm(view(D₀, (r + 1):m)) atol = atol

        #truncerror
        s = 1 - sqrt(eps(real(eltype(T))))
        trunc = truncerror(; atol = s * norm(@view(D₀[r:end]), 1), p = 1)
        D3, V3, ϵ3 = @testinferred eigh_trunc(A; trunc)
        @test length(diagview(D3)) == r
        @test A * V3 ≈ V3 * D3
        @test ϵ3 ≈ norm(view(D₀, (r + 1):m)) atol = atol

        s = 1 - sqrt(eps(real(eltype(T))))
        trunc = truncerror(; atol = s * norm(@view(D₀[r:end]), 1), p = 1)
        D4, V4 = @testinferred eigh_trunc_no_error(A; trunc)
        @test length(diagview(D4)) == r
        @test A * V4 ≈ V4 * D4

        # test for same subspace
        @test V1 * (V1' * V2) ≈ V2
        @test V2 * (V2' * V1) ≈ V1
        @test V1 * (V1' * V3) ≈ V3
        @test V3 * (V3' * V1) ≈ V1
        @test V4 * (V4' * V1) ≈ V1

        @testset "specify truncation algorithm" begin
            atol = sqrt(eps(real(eltype(T))))
            m4 = 4
            smallA = randn!(similar(A, (m4, m4)))
            V = T <: Diagonal ? I : qr_compact(smallA)[1]
            Ddiag = similar(A, real(eltype(T)), m4)
            copyto!(Ddiag, real(eltype(T))[0.9, 0.3, 0.1, 0.01])
            D = Diagonal(Ddiag)
            A = project_hermitian!(V * D * V')
            alg = TruncatedAlgorithm(MatrixAlgebraKit.default_eigh_algorithm(A), truncrank(2))
            D2, V2, ϵ2 = @testinferred eigh_trunc(A; alg)
            @test diagview(D2) ≈ diagview(D)[1:2]
            @test_throws ArgumentError eigh_trunc(A; alg, trunc = (; maxrank = 2))
            @test ϵ2 ≈ norm(diagview(D)[3:4]) atol = atol

            alg = TruncatedAlgorithm(MatrixAlgebraKit.default_eigh_algorithm(A), truncerror(; atol = 0.2))
            D3, V3, ϵ3 = @testinferred eigh_trunc(A; alg)
            @test diagview(D3) ≈ diagview(D)[1:2]
            @test ϵ3 ≈ norm(diagview(D)[3:4]) atol = atol
        end
    end
end

function test_eigh_trunc_algs(
        T::Type, sz, algs;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "eigh_trunc! algorithm $alg $summary_str" for alg in algs
        A = instantiate_matrix(T, sz)
        A = A * A'
        A = project_hermitian!(A)
        Ac = deepcopy(A)

        m = size(A, 1)
        D₀ = collect(reverse(eigh_vals(A)))
        r = m - 2
        s = 1 + sqrt(eps(real(eltype(T))))
        # truncrank
        atol = sqrt(eps(real(eltype(T))))
        m4 = 4
        smallA = randn!(similar(A, (m4, m4)))
        V = T <: Diagonal ? I : qr_compact(smallA)[1]
        Ddiag = similar(A, real(eltype(T)), m4)
        copyto!(Ddiag, real(eltype(T))[0.9, 0.3, 0.1, 0.01])
        D = Diagonal(Ddiag)
        A = project_hermitian!(V * D * V')
        truncalg = TruncatedAlgorithm(alg, truncrank(2))
        D2, V2, ϵ2 = @testinferred eigh_trunc(A; alg = truncalg)
        @test diagview(D2) ≈ diagview(D)[1:2]
        @test_throws ArgumentError eigh_trunc(A; alg = truncalg, trunc = (; maxrank = 2))
        @test ϵ2 ≈ norm(diagview(D)[3:4]) atol = atol

        truncalg = TruncatedAlgorithm(alg, truncerror(; atol = 0.2))
        D3, V3, ϵ3 = @testinferred eigh_trunc(A; alg = truncalg)
        @test diagview(D3) ≈ diagview(D)[1:2]
        @test ϵ3 ≈ norm(diagview(D)[3:4]) atol = atol

        truncalg = TruncatedAlgorithm(alg, truncerror(; atol = 0.2))
        D4, V4 = @testinferred eigh_trunc_no_error(A; alg = truncalg)
        @test diagview(D4) ≈ diagview(D)[1:2]
    end
end
