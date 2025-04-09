@testset "left_orth and left_null for T = $T" for T in (Float32, Float64, ComplexF32,
                                                        ComplexF64)
    rng = StableRNG(123)
    m = 54
    for n in (37, m, 63)
        minmn = min(m, n)
        A = randn(rng, T, m, n)
        V, C = @constinferred left_orth(A)
        N = @constinferred left_null(A)
        @test V isa Matrix{T} && size(V) == (m, minmn)
        @test C isa Matrix{T} && size(C) == (minmn, n)
        @test N isa Matrix{T} && size(N) == (m, m - minmn)
        @test V * C ≈ A
        @test V' * V ≈ I
        @test LinearAlgebra.norm(A' * N) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test N' * N ≈ I
        @test V * V' + N * N' ≈ I

        Ac = similar(A)
        V2, C2 = @constinferred left_orth!(copy!(Ac, A), (V, C))
        N2 = @constinferred left_null!(copy!(Ac, A), N)
        @test V2 === V
        @test C2 === C
        @test N2 === N
        @test V2 * C2 ≈ A
        @test V2' * V2 ≈ I
        @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test N2' * N2 ≈ I
        @test V2 * V2' + N2 * N2' ≈ I

        atol = eps(real(T))
        V2, C2 = @constinferred left_orth!(copy!(Ac, A), (V, C); atol=atol)
        N2 = @constinferred left_null!(copy!(Ac, A), N; atol=atol)
        @test V2 !== V
        @test C2 !== C
        @test N2 !== C
        @test V2 * C2 ≈ A
        @test V2' * V2 ≈ I
        @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test N2' * N2 ≈ I
        @test V2 * V2' + N2 * N2' ≈ I

        rtol = eps(real(T))
        V2, C2 = @constinferred left_orth!(copy!(Ac, A), (V, C); rtol=rtol)
        N2 = @constinferred left_null!(copy!(Ac, A), N; rtol=rtol)
        @test V2 !== V
        @test C2 !== C
        @test N2 !== C
        @test V2 * C2 ≈ A
        @test V2' * V2 ≈ I
        @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test N2' * N2 ≈ I
        @test V2 * V2' + N2 * N2' ≈ I

        for kind in (:qr, :qrpos, :polar, :svd) # explicit kind kwarg
            m < n && kind == :polar && continue
            V2, C2 = @constinferred left_orth!(copy!(Ac, A), (V, C); kind=kind)
            @test V2 === V
            @test C2 === C
            @test V2 * C2 ≈ A
            @test V2' * V2 ≈ I
            if kind != :polar
                N2 = @constinferred left_null!(copy!(Ac, A), N; kind=kind)
                @test N2 === N
                @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test N2' * N2 ≈ I
                @test V2 * V2' + N2 * N2' ≈ I
            end

            # with kind and tol kwargs
            if kind == :svd
                V2, C2 = @constinferred left_orth!(copy!(Ac, A), (V, C); kind=kind,
                                                   atol=atol)
                N2 = @constinferred left_null!(copy!(Ac, A), N; kind=kind, atol=atol)
                @test V2 !== V
                @test C2 !== C
                @test N2 !== C
                @test V2 * C2 ≈ A
                @test V2' * V2 ≈ I
                @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test N2' * N2 ≈ I
                @test V2 * V2' + N2 * N2' ≈ I

                V2, C2 = @constinferred left_orth!(copy!(Ac, A), (V, C); kind=kind,
                                                   rtol=rtol)
                N2 = @constinferred left_null!(copy!(Ac, A), N; kind=kind, rtol=rtol)
                @test V2 !== V
                @test C2 !== C
                @test N2 !== C
                @test V2 * C2 ≈ A
                @test V2' * V2 ≈ I
                @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test N2' * N2 ≈ I
                @test V2 * V2' + N2 * N2' ≈ I
            else
                @test_throws ArgumentError left_orth!(copy!(Ac, A), (V, C); kind=kind,
                                                      atol=atol)
                @test_throws ArgumentError left_orth!(copy!(Ac, A), (V, C); kind=kind,
                                                      rtol=rtol)
                @test_throws ArgumentError left_null!(copy!(Ac, A), N; kind=kind, atol=atol)
                @test_throws ArgumentError left_null!(copy!(Ac, A), N; kind=kind, rtol=rtol)
            end
        end
    end
end

@testset "right_orth and right_null for T = $T" for T in (Float32, Float64, ComplexF32,
                                                          ComplexF64)
    rng = StableRNG(123)
    m = 54
    for n in (37, m, 63)
        minmn = min(m, n)
        A = randn(rng, T, m, n)
        C, Vᴴ = @constinferred right_orth(A)
        Nᴴ = @constinferred right_null(A)
        @test C isa Matrix{T} && size(C) == (m, minmn)
        @test Vᴴ isa Matrix{T} && size(Vᴴ) == (minmn, n)
        @test Nᴴ isa Matrix{T} && size(Nᴴ) == (n - minmn, n)
        @test C * Vᴴ ≈ A
        @test Vᴴ * Vᴴ' ≈ I
        @test LinearAlgebra.norm(A * adjoint(Nᴴ)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test Nᴴ * Nᴴ' ≈ I
        @test Vᴴ' * Vᴴ + Nᴴ' * Nᴴ ≈ I

        Ac = similar(A)
        C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ))
        Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ)
        @test C2 === C
        @test Vᴴ2 === Vᴴ
        @test Nᴴ2 === Nᴴ
        @test C2 * Vᴴ2 ≈ A
        @test Vᴴ2 * Vᴴ2' ≈ I
        @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test Nᴴ2 * Nᴴ2' ≈ I
        @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I

        atol = eps(real(T))
        C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ); atol=atol)
        Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ; atol=atol)
        @test C2 !== C
        @test Vᴴ2 !== Vᴴ
        @test Nᴴ2 !== Nᴴ
        @test C2 * Vᴴ2 ≈ A
        @test Vᴴ2 * Vᴴ2' ≈ I
        @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test Nᴴ2 * Nᴴ2' ≈ I
        @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I

        rtol = eps(real(T))
        C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ); rtol=rtol)
        Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ; rtol=rtol)
        @test C2 !== C
        @test Vᴴ2 !== Vᴴ
        @test Nᴴ2 !== Nᴴ
        @test C2 * Vᴴ2 ≈ A
        @test Vᴴ2 * Vᴴ2' ≈ I
        @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test Nᴴ2 * Nᴴ2' ≈ I
        @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I

        for kind in (:lq, :lqpos, :polar, :svd)
            n < m && kind == :polar && continue
            C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ); kind=kind)
            @test C2 === C
            @test Vᴴ2 === Vᴴ
            @test C2 * Vᴴ2 ≈ A
            @test Vᴴ2 * Vᴴ2' ≈ I
            if kind != :polar
                Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ; kind=kind)
                @test Nᴴ2 === Nᴴ
                @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test Nᴴ2 * Nᴴ2' ≈ I
                @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I
            end

            if kind == :svd
                C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ); kind=kind,
                                                     atol=atol)
                Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ; kind=kind, atol=atol)
                @test C2 !== C
                @test Vᴴ2 !== Vᴴ
                @test Nᴴ2 !== Nᴴ
                @test C2 * Vᴴ2 ≈ A
                @test Vᴴ2 * Vᴴ2' ≈ I
                @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test Nᴴ2 * Nᴴ2' ≈ I
                @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I

                C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ); kind=kind,
                                                     rtol=rtol)
                Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ; kind=kind, rtol=rtol)
                @test C2 !== C
                @test Vᴴ2 !== Vᴴ
                @test Nᴴ2 !== Nᴴ
                @test C2 * Vᴴ2 ≈ A
                @test Vᴴ2 * Vᴴ2' ≈ I
                @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test Nᴴ2 * Nᴴ2' ≈ I
                @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I
            else
                @test_throws ArgumentError right_orth!(copy!(Ac, A), (C, Vᴴ); kind=kind,
                                                       atol=atol)
                @test_throws ArgumentError right_orth!(copy!(Ac, A), (C, Vᴴ); kind=kind,
                                                       rtol=rtol)
                @test_throws ArgumentError right_null!(copy!(Ac, A), Nᴴ; kind=kind,
                                                       atol=atol)
                @test_throws ArgumentError right_null!(copy!(Ac, A), Nᴴ; kind=kind,
                                                       rtol=rtol)
            end
        end
    end
end
