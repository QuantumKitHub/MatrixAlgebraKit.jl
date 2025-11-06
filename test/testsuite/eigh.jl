using TestExtras

function test_eigh(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "eigh $summary_str" begin
        test_eigh_full(T, sz; kwargs...)
        if T <: Number && eltype(T) <: Union{Float16, ComplexF16, Float32, Float64, ComplexF32, ComplexF64} && !(T <: Diagonal)
            test_eigh_trunc(T, sz; kwargs...)
        end
    end
end

function test_eigh_full(
        T::Type, sz;
        test_blocksize = true,
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "eigh_full! $summary_str" begin
        A = instantiate_matrix(T, sz)
        A = (A + A') / 2
        Ac = deepcopy(A)

        D, V = @testinferred eigh_full(A)
        @test A * V ≈ V * D
        @test isunitary(V)
        @test all(isreal, D)

        D2, V2 = eigh_full!(copy(A), (D, V))
        @test D2 === D
        @test V2 === V

        D3 = @testinferred eigh_vals(A)
        @test D ≈ Diagonal(D3)
    end
end

function test_eigh_trunc(
        T::Type, sz;
        test_blocksize = true,
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "eigh_trunc! $summary_str" begin
        A = instantiate_matrix(T, sz)
        A = A * A'
        A = (A + A') / 2
        Ac = deepcopy(A)

        m = size(A, 1)
        D₀ = reverse(eigh_vals(A))
        r = m - 2
        s = 1 + sqrt(eps(real(eltype(T))))
        atol = sqrt(eps(real(eltype(T))))
        # truncrank
        D1, V1, ϵ1 = @testinferred eigh_trunc(A; trunc = truncrank(r))
        @test length(diagview(D1)) == r
        @test isisometric(V1)
        @test A * V1 ≈ V1 * D1
        @test LinearAlgebra.opnorm(A - V1 * D1 * V1') ≈ D₀[r + 1]
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

        # test for same subspace
        @test V1 * (V1' * V2) ≈ V2
        @test V2 * (V2' * V1) ≈ V1
        @test V1 * (V1' * V3) ≈ V3
        @test V3 * (V3' * V1) ≈ V1

        # TODO
        #=
        @testset "specify truncation algorithm" begin
            atol = sqrt(eps(real(eltype(T))))
            V = qr_compact(instantiate_matrix(T, sz))[1]
            D = Diagonal(real(eltype(T))[0.9, 0.3, 0.1, 0.01])
            A = V * D * V'
            A = (A + A') / 2
            alg = TruncatedAlgorithm(MatrixAlgebraKit.default_qr_algorithm(A), truncrank(2))
            D2, V2, ϵ2 = @testinferred eigh_trunc(A; alg)
            @test diagview(D2) ≈ diagview(D)[1:2]
            @test_throws ArgumentError eigh_trunc(A; alg, trunc = (; maxrank = 2))
            @test ϵ2 ≈ norm(diagview(D)[3:4]) atol = atol

            alg = TruncatedAlgorithm(MatrixAlgebraKit.default_qr_algorithm(A), truncerror(; atol = 0.2))
            D3, V3, ϵ3 = @testinferred eigh_trunc(A; alg)
            @test diagview(D3) ≈ diagview(D)[1:2]
            @test ϵ3 ≈ norm(diagview(D)[3:4]) atol = atol
        end=#
    end
end
