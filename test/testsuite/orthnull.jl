using TestExtras

function test_orthnull(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "orthnull $summary_str" begin
        test_left_orthnull(T, sz; kwargs...)
        test_right_orthnull(T, sz; kwargs...)
    end
end

function test_left_orthnull(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "left_orth! and left_null! $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)
        V, C = @testinferred left_orth(A)
        N = @testinferred left_null(A)
        m, n = size(A)
        minmn = min(m, n)
        @test V isa Matrix{T} && size(V) == (m, minmn)
        @test C isa Matrix{T} && size(C) == (minmn, n)
        @test N isa Matrix{T} && size(N) == (m, m - minmn)
        @test V * C ≈ A
        @test isisometric(V)
        @test LinearAlgebra.norm(A' * N) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(N)
        @test V * V' + N * N' ≈ I

        M = LinearMap(A)
        VM, CM = @testinferred left_orth(M; alg = :svd)
        @test parent(VM) * parent(CM) ≈ A

        if m > n
            nullity = 5
            V, C = @testinferred left_orth(A)
            N = @testinferred left_null(A; trunc = (; maxnullity = nullity))
            @test V isa Matrix{T} && size(V) == (m, minmn)
            @test C isa Matrix{T} && size(C) == (minmn, n)
            @test N isa Matrix{T} && size(N) == (m, nullity)
            @test V * C ≈ A
            @test isisometric(V)
            @test LinearAlgebra.norm(A' * N) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
            @test isisometric(N)
        end

        # passing a kind and some kwargs
        V, C = @testinferred left_orth(A; alg = :qr, positive = true)
        N = @testinferred left_null(A; alg = :qr, positive = true)
        @test V isa Matrix{T} && size(V) == (m, minmn)
        @test C isa Matrix{T} && size(C) == (minmn, n)
        @test N isa Matrix{T} && size(N) == (m, m - minmn)
        @test V * C ≈ A
        @test isisometric(V)
        @test LinearAlgebra.norm(A' * N) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(N)
        @test V * V' + N * N' ≈ I

        # passing an algorithm
        V, C = @testinferred left_orth(A; alg = LAPACK_HouseholderQR())
        N = @testinferred left_null(A; alg = :qr, positive = true)
        @test V isa Matrix{T} && size(V) == (m, minmn)
        @test C isa Matrix{T} && size(C) == (minmn, n)
        @test N isa Matrix{T} && size(N) == (m, m - minmn)
        @test V * C ≈ A
        @test isisometric(V)
        @test LinearAlgebra.norm(A' * N) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(N)
        @test V * V' + N * N' ≈ I

        Ac = similar(A)
        V2, C2 = @testinferred left_orth!(copy!(Ac, A), (V, C))
        N2 = @testinferred left_null!(copy!(Ac, A), N)
        @test V2 * C2 ≈ A
        @test isisometric(V2)
        @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(N2)
        @test V2 * V2' + N2 * N2' ≈ I

        atol = eps(real(eltype(T)))
        V2, C2 = @testinferred left_orth!(copy!(Ac, A), (V, C); trunc = (; atol = atol))
        N2 = @testinferred left_null!(copy!(Ac, A), N; trunc = (; atol = atol))
        @test V2 * C2 ≈ A
        @test isisometric(V2)
        @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(N2)
        @test V2 * V2' + N2 * N2' ≈ I

        rtol = eps(real(eltype(T)))
        for (trunc_orth, trunc_null) in (
                ((; rtol = rtol), (; rtol = rtol)),
                (trunctol(; rtol), trunctol(; rtol, keep_below = true)),
            )
            V2, C2 = @testinferred left_orth!(copy!(Ac, A), (V, C); trunc = trunc_orth)
            N2 = @testinferred left_null!(copy!(Ac, A), N; trunc = trunc_null)
            @test V2 * C2 ≈ A
            @test isisometric(V2)
            @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
            @test isisometric(N2)
            @test V2 * V2' + N2 * N2' ≈ I
        end

        for alg in (:qr, :polar, :svd) # explicit kind kwarg
            m < n && alg === :polar && continue
            V2, C2 = @testinferred left_orth!(copy!(Ac, A), (V, C); alg = $(QuoteNode(alg)))
            @test V2 * C2 ≈ A
            @test isisometric(V2)
            if alg != :polar
                N2 = @testinferred left_null!(copy!(Ac, A), N; alg)
                @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test isisometric(N2)
                @test V2 * V2' + N2 * N2' ≈ I
            end

            # with kind and tol kwargs
            if alg == :svd
                V2, C2 = @testinferred left_orth!(copy!(Ac, A), (V, C); alg = $(QuoteNode(alg)), trunc = (; atol))
                N2 = @testinferred left_null!(copy!(Ac, A), N; alg, trunc = (; atol))
                @test V2 * C2 ≈ A
                @test isisometric(V2)
                @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test isisometric(N2)
                @test V2 * V2' + N2 * N2' ≈ I

                V2, C2 = @testinferred left_orth!(copy!(Ac, A), (V, C); alg = $(QuoteNode(alg)), trunc = (; rtol))
                N2 = @testinferred left_null!(copy!(Ac, A), N; alg, trunc = (; rtol))
                @test V2 * C2 ≈ A
                @test isisometric(V2)
                @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test isisometric(N2)
                @test V2 * V2' + N2 * N2' ≈ I
            else
                @test_throws ArgumentError left_orth!(copy!(Ac, A), (V, C); alg, trunc = (; atol))
                @test_throws ArgumentError left_orth!(copy!(Ac, A), (V, C); alg, trunc = (; rtol))
                alg == :polar && continue
                @test_throws ArgumentError left_null!(copy!(Ac, A), N; alg, trunc = (; atol))
                @test_throws ArgumentError left_null!(copy!(Ac, A), N; alg, trunc = (; rtol))
            end
        end
    end
end

function test_right_orthnull(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "right_orth! and right_null! $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)
        C, Vᴴ = @testinferred right_orth(A)
        Nᴴ = @testinferred right_null(A)
        @test C isa Matrix{T} && size(C) == (m, minmn)
        @test Vᴴ isa Matrix{T} && size(Vᴴ) == (minmn, n)
        @test Nᴴ isa Matrix{T} && size(Nᴴ) == (n - minmn, n)
        @test C * Vᴴ ≈ A
        @test isisometric(Vᴴ; side = :right)
        @test LinearAlgebra.norm(A * adjoint(Nᴴ)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(Nᴴ; side = :right)
        @test Vᴴ' * Vᴴ + Nᴴ' * Nᴴ ≈ I

        M = LinearMap(A)
        CM, VMᴴ = @testinferred right_orth(M; alg = :svd)
        @test parent(CM) * parent(VMᴴ) ≈ A

        Ac = similar(A)
        C2, Vᴴ2 = @testinferred right_orth!(copy!(Ac, A), (C, Vᴴ))
        Nᴴ2 = @testinferred right_null!(copy!(Ac, A), Nᴴ)
        @test C2 * Vᴴ2 ≈ A
        @test isisometric(Vᴴ2; side = :right)
        @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(Nᴴ; side = :right)
        @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I

        atol = eps(real(T))
        C2, Vᴴ2 = @testinferred right_orth!(copy!(Ac, A), (C, Vᴴ); trunc = (; atol))
        Nᴴ2 = @testinferred right_null!(copy!(Ac, A), Nᴴ; trunc = (; atol))
        @test C2 * Vᴴ2 ≈ A
        @test isisometric(Vᴴ2; side = :right)
        @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(Nᴴ; side = :right)
        @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I

        rtol = eps(real(T))
        C2, Vᴴ2 = @testinferred right_orth!(copy!(Ac, A), (C, Vᴴ); trunc = (; rtol))
        Nᴴ2 = @testinferred right_null!(copy!(Ac, A), Nᴴ; trunc = (; rtol))
        @test C2 * Vᴴ2 ≈ A
        @test isisometric(Vᴴ2; side = :right)
        @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(Nᴴ2; side = :right)
        @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I

        for alg in (:lq, :polar, :svd)
            n < m && alg == :polar && continue
            C2, Vᴴ2 = @testinferred right_orth!(copy!(Ac, A), (C, Vᴴ); alg = $(QuoteNode(alg)))
            @test C2 * Vᴴ2 ≈ A
            @test isisometric(Vᴴ2; side = :right)
            if alg != :polar
                Nᴴ2 = @testinferred right_null!(copy!(Ac, A), Nᴴ; alg = $(QuoteNode(alg)))
                @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test isisometric(Nᴴ2; side = :right)
                @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I
            end

            if alg == :svd
                C2, Vᴴ2 = @testinferred right_orth!(copy!(Ac, A), (C, Vᴴ); alg = $(QuoteNode(alg)), trunc = (; atol))
                Nᴴ2 = @testinferred right_null!(copy!(Ac, A), Nᴴ; alg = $(QuoteNode(alg)), trunc = (; atol))
                @test C2 * Vᴴ2 ≈ A
                @test isisometric(Vᴴ2; side = :right)
                @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test isisometric(Nᴴ2; side = :right)
                @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I

                C2, Vᴴ2 = @testinferred right_orth!(copy!(Ac, A), (C, Vᴴ); alg = $(QuoteNode(alg)), trunc = (; rtol))
                Nᴴ2 = @testinferred right_null!(copy!(Ac, A), Nᴴ; alg = $(QuoteNode(alg)), trunc = (; rtol))
                @test C2 * Vᴴ2 ≈ A
                @test isisometric(Vᴴ2; side = :right)
                @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test isisometric(Nᴴ2; side = :right)
                @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I
            else
                @test_throws ArgumentError right_orth!(copy!(Ac, A), (C, Vᴴ); alg, trunc = (; atol))
                @test_throws ArgumentError right_orth!(copy!(Ac, A), (C, Vᴴ); alg, trunc = (; rtol))
                alg == :polar && continue
                @test_throws ArgumentError right_null!(copy!(Ac, A), Nᴴ; alg, trunc = (; atol))
                @test_throws ArgumentError right_null!(copy!(Ac, A), Nᴴ; alg, trunc = (; rtol))
            end
        end
    end
end
