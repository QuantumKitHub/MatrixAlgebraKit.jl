using TestExtras

function test_eig(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "eig $summary_str" begin
        test_eig_full(T, sz; kwargs...)
        if T <: Number || T <: Diagonal{<:Number, <:Vector}
            test_eig_trunc(T, sz; kwargs...)
        end
    end
end

function test_eig_full(
        T::Type, sz;
        test_blocksize = true,
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "eig_full! $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)
        Tc = isa(A, Diagonal) ? eltype(T) : complex(eltype(T))
        D, V = @testinferred eig_full(A)
        @test eltype(D) == eltype(V) == Tc
        @test A * V ≈ V * D

        D2, V2 = @testinferred eig_full!(Ac, (D, V))
        @test D2 === D
        @test V2 === V
        @test A * V ≈ V * D

        Dc = @testinferred eig_vals(A)
        @test eltype(Dc) == Tc
        @test D ≈ Diagonal(Dc)
    end
end

function test_eig_trunc(
        T::Type, sz;
        test_blocksize = true,
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "eig_trunc! $summary_str" begin
        A = instantiate_matrix(T, sz)
        A *= A' # TODO: deal with eigenvalue ordering etc
        Ac = deepcopy(A)
        Tc = complex(eltype(T))
        # eigenvalues are sorted by ascending real component...
        D₀ = sort!(eig_vals(A); by = abs, rev = true)
        m = size(A, 1)
        rmin = findfirst(i -> abs(D₀[end - i]) != abs(D₀[end - i - 1]), 1:(m - 2))
        r = length(D₀) - rmin
        atol = sqrt(eps(real(eltype(T))))

        D1, V1, ϵ1 = @testinferred eig_trunc(A; trunc = truncrank(r))
        @test length(diagview(D1)) == r
        @test A * V1 ≈ V1 * D1
        @test ϵ1 ≈ norm(view(D₀, (r + 1):m)) atol = atol

        s = 1 + sqrt(eps(real(eltype(T))))
        trunc = trunctol(; atol = s * abs(D₀[r + 1]))
        D2, V2, ϵ2 = @testinferred eig_trunc(A; trunc)
        @test length(diagview(D2)) == r
        @test A * V2 ≈ V2 * D2
        @test ϵ2 ≈ norm(view(D₀, (r + 1):m)) atol = atol

        s = 1 - sqrt(eps(real(eltype(T))))
        trunc = truncerror(; atol = s * norm(@view(D₀[r:end]), 1), p = 1)
        D3, V3, ϵ3 = @testinferred eig_trunc(A; trunc)
        @test length(diagview(D3)) == r
        @test A * V3 ≈ V3 * D3
        @test ϵ3 ≈ norm(view(D₀, (r + 1):m)) atol = atol

        # trunctol keeps order, truncrank might not
        # test for same subspace
        @test V1 * ((V1' * V1) \ (V1' * V2)) ≈ V2
        @test V2 * ((V2' * V2) \ (V2' * V1)) ≈ V1
        @test V1 * ((V1' * V1) \ (V1' * V3)) ≈ V3
        @test V3 * ((V3' * V3) \ (V3' * V1)) ≈ V1

        # TODO
        #=atol = sqrt(eps(real(eltype(T))))
        V = randn(rng, T, m, m)
        D = Diagonal(real(T)[0.9, 0.3, 0.1, 0.01])
        A = V * D * inv(V)
        alg = TruncatedAlgorithm(LAPACK_Simple(), truncrank(2))
        D2, V2, ϵ2 = @testinferred eig_trunc(A; alg)
        @test diagview(D2) ≈ diagview(D)[1:2]
        @test ϵ2 ≈ norm(diagview(D)[3:4]) atol = atol
        @test_throws ArgumentError eig_trunc(A; alg, trunc = (; maxrank = 2))

        alg = TruncatedAlgorithm(LAPACK_Simple(), truncerror(; atol = 0.2, p = 1))
        D3, V3, ϵ3 = @testinferred eig_trunc(A; alg)
        @test diagview(D3) ≈ diagview(D)[1:2]
        @test ϵ3 ≈ norm(diagview(D)[3:4]) atol = atol=#
    end
end
