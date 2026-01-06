using TestExtras
using LinearAlgebra

include("../linearmap.jl")

_left_orth_svd(x; kwargs...) = left_orth(x; alg = :svd, kwargs...)
_left_orth_svd!(x, VC; kwargs...) = left_orth!(x, VC; alg = :svd, kwargs...)
_left_orth_qr(x; kwargs...) = left_orth(x; alg = :qr, kwargs...)
_left_orth_qr!(x, VC; kwargs...) = left_orth!(x, VC; alg = :qr, kwargs...)
_left_orth_polar(x; kwargs...) = left_orth(x; alg = :polar, kwargs...)
_left_orth_polar!(x, VC; kwargs...) = left_orth!(x, VC; alg = :polar, kwargs...)

_right_orth_svd(x; kwargs...) = right_orth(x; alg = :svd, kwargs...)
_right_orth_svd!(x, CVᴴ; kwargs...) = right_orth!(x, CVᴴ; alg = :svd, kwargs...)
_right_orth_lq(x; kwargs...) = right_orth(x; alg = :lq, kwargs...)
_right_orth_lq!(x, CVᴴ; kwargs...) = right_orth!(x, CVᴴ; alg = :lq, kwargs...)
_right_orth_polar(x; kwargs...) = right_orth(x; alg = :polar, kwargs...)
_right_orth_polar!(x, CVᴴ; kwargs...) = right_orth!(x, CVᴴ; alg = :polar, kwargs...)

function test_orthnull(T::Type, sz; test_nullity = true, test_orthnull = true, kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "orthnull $summary_str" begin
        test_orthnull && test_left_orthnull(T, sz; kwargs...)
        test_nullity  && test_left_nullity(T, sz; kwargs...)
        test_orthnull && test_right_orthnull(T, sz; kwargs...)
        test_nullity  && test_right_nullity(T, sz; kwargs...)
    end
end

function test_left_nullity(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "left_nullity $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)
        m, n = size(A)
        minmn = min(m, n)
        if m > n
            nullity = 5
            V, C = @testinferred left_orth(A)
            N = @testinferred left_null(A; trunc = (; maxnullity = nullity))
            @test V isa typeof(A) && size(V) == (m, minmn)
            @test C isa typeof(A) && size(C) == (minmn, n)
            @test eltype(N) == eltype(A) && size(N) == (m, nullity)
            @test V * C ≈ A
            @test isisometric(V)
            @test LinearAlgebra.norm(A' * N) ≈ 0 atol = MatrixAlgebraKit.defaulttol(eltype(T))
            @test isisometric(N)
        end

        rtol = eps(real(eltype(T)))
        for (trunc_orth, trunc_null) in (
                ((; rtol = rtol), (; rtol = rtol)),
                (trunctol(; rtol), trunctol(; rtol, keep_below = true)),
            )
            V, C = left_orth(A)
            N = left_null(A)
            V2, C2 = @testinferred left_orth!(copy!(Ac, A), (V, C); trunc = trunc_orth)
            N2 = @testinferred left_null!(copy!(Ac, A), N; trunc = trunc_null)
            @test V2 * C2 ≈ A
            @test isisometric(V2)
            @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(eltype(T))
            @test isisometric(N2)
            @test isleftcomplete(V2, N2)
        end

        alg = :svd
        V2, C2 = @testinferred _left_orth_svd!(copy!(Ac, A), (V, C); trunc = (; atol))
        N2 = @testinferred left_null!(copy!(Ac, A), N; alg, trunc = (; atol))
        @test V2 * C2 ≈ A
        @test isisometric(V2)
        @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(eltype(T))
        @test isisometric(N2)
        @test isleftcomplete(V2, N2)

        V2, C2 = @testinferred _left_orth_svd!(copy!(Ac, A), (V, C); trunc = (; rtol))
        N2 = @testinferred left_null!(copy!(Ac, A), N; alg, trunc = (; rtol))
        @test V2 * C2 ≈ A
        @test isisometric(V2)
        @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(eltype(T))
        @test isisometric(N2)
        @test isleftcomplete(V2, N2)

        # doesn't work on AMD...
        atol = eps(real(eltype(T)))
        V2, C2 = @testinferred left_orth!(copy!(Ac, A), (V, C); trunc = (; atol = atol))
        N2 = @testinferred left_null!(copy!(Ac, A), N; trunc = (; atol = atol))
        @test V2 * C2 ≈ A
        @test isisometric(V2)
        @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(N2)
        @test isleftcomplete(V2, N2)

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
        @test V isa typeof(A) && size(V) == (m, minmn)
        @test C isa typeof(A) && size(C) == (minmn, n)
        @test eltype(N) == eltype(A) && size(N) == (m, m - minmn)
        @test V * C ≈ A
        @test isisometric(V)
        @test LinearAlgebra.norm(A' * N) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(N)
        @test isleftcomplete(V, N)

        M = LinearMap(A)
        VM, CM = @testinferred _left_orth_svd(M)
        @test parent(VM) * parent(CM) ≈ A

        # passing a kind and some kwargs
        V, C = @testinferred _left_orth_qr(A; positive = true)
        N = @testinferred left_null(A; alg = :qr, positive = true)
        @test V isa typeof(A) && size(V) == (m, minmn)
        @test C isa typeof(A) && size(C) == (minmn, n)
        @test eltype(N) == eltype(A) && size(N) == (m, m - minmn)
        @test V * C ≈ A
        @test isisometric(V)
        @test LinearAlgebra.norm(A' * N) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(N)
        @test isleftcomplete(V, N)

        # passing an algorithm
        V, C = @testinferred left_orth(A; alg = MatrixAlgebraKit.default_qr_algorithm(A))
        N = @testinferred left_null(A; alg = :qr, positive = true)
        @test V isa typeof(A) && size(V) == (m, minmn)
        @test C isa typeof(A) && size(C) == (minmn, n)
        @test eltype(N) == eltype(A) && size(N) == (m, m - minmn)
        @test V * C ≈ A
        @test isisometric(V)
        @test LinearAlgebra.norm(A' * N) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(N)
        @test isleftcomplete(V, N)

        V2, C2 = @testinferred left_orth!(copy!(Ac, A), (V, C))
        N2 = @testinferred left_null!(copy!(Ac, A), N)
        @test V2 * C2 ≈ A
        @test isisometric(V2)
        @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(N2)
        @test isleftcomplete(V2, N2)

        for alg in (:qr, :polar, :svd) # explicit kind kwarg
            m < n && alg === :polar && continue
            if alg == :svd
                V2, C2 = @testinferred _left_orth_svd!(copy!(Ac, A), (V, C))
            elseif alg == :qr
                V2, C2 = @testinferred _left_orth_qr!(copy!(Ac, A), (V, C))
            elseif alg == :polar
                V2, C2 = @testinferred _left_orth_polar!(copy!(Ac, A), (V, C))
            end
            @test V2 * C2 ≈ A
            @test isisometric(V2)
            if alg != :polar
                N2 = @testinferred left_null!(copy!(Ac, A), N; alg)
                @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test isisometric(N2)
                @test isleftcomplete(V2, N2)
            end

            # with kind and tol kwargs
            if alg != :svd
                @test_throws ArgumentError left_orth!(copy!(Ac, A), (V, C); alg, trunc = (; atol))
                @test_throws ArgumentError left_orth!(copy!(Ac, A), (V, C); alg, trunc = (; rtol))
                alg == :polar && continue
                @test_throws ArgumentError left_null!(copy!(Ac, A), N; alg, trunc = (; atol))
                @test_throws ArgumentError left_null!(copy!(Ac, A), N; alg, trunc = (; rtol))
            end
        end
    end
end

function test_right_nullity(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "right_nullity $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)
        m, n = size(A)
        minmn = min(m, n)

        C, Vᴴ = @testinferred right_orth(A)
        Nᴴ = @testinferred right_null(A)
        atol = eps(real(eltype(T)))
        C2, Vᴴ2 = @testinferred right_orth!(copy!(Ac, A), (C, Vᴴ); trunc = (; atol))
        Nᴴ2 = @testinferred right_null!(copy!(Ac, A), Nᴴ; trunc = (; atol))
        @test C2 * Vᴴ2 ≈ A
        @test isisometric(Vᴴ2; side = :right)
        @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(eltype(T))
        @test isisometric(Nᴴ; side = :right)
        @test isrightcomplete(Vᴴ2, Nᴴ2)

        rtol = eps(real(eltype(T)))
        C2, Vᴴ2 = @testinferred right_orth!(copy!(Ac, A), (C, Vᴴ); trunc = (; rtol))
        Nᴴ2 = @testinferred right_null!(copy!(Ac, A), Nᴴ; trunc = (; rtol))
        @test C2 * Vᴴ2 ≈ A
        @test isisometric(Vᴴ2; side = :right)
        @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(eltype(T))
        @test isisometric(Nᴴ2; side = :right)
        @test isrightcomplete(Vᴴ2, Nᴴ2)

        alg = :svd
        C2, Vᴴ2 = @testinferred _right_orth_svd!(copy!(Ac, A), (C, Vᴴ); trunc = (; atol))
        Nᴴ2 = @testinferred right_null!(copy!(Ac, A), Nᴴ; alg = alg, trunc = (; atol))
        @test C2 * Vᴴ2 ≈ A
        @test isisometric(Vᴴ2; side = :right)
        @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(Nᴴ2; side = :right)
        @test isrightcomplete(Vᴴ2, Nᴴ2)

        C2, Vᴴ2 = @testinferred _right_orth_svd!(copy!(Ac, A), (C, Vᴴ); trunc = (; rtol))
        Nᴴ2 = @testinferred right_null!(copy!(Ac, A), Nᴴ; alg = alg, trunc = (; rtol))
        @test C2 * Vᴴ2 ≈ A
        @test isisometric(Vᴴ2; side = :right)
        @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test isisometric(Nᴴ2; side = :right)
        @test isrightcomplete(Vᴴ2, Nᴴ2)
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
        m, n = size(A)
        minmn = min(m, n)
        Ac = deepcopy(A)
        C, Vᴴ = @testinferred right_orth(A)
        Nᴴ = @testinferred right_null(A)
        @test C isa typeof(A) && size(C) == (m, minmn)
        @test Vᴴ isa typeof(A) && size(Vᴴ) == (minmn, n)
        @test eltype(Nᴴ) == eltype(A) && size(Nᴴ) == (n - minmn, n)
        @test C * Vᴴ ≈ A
        @test isisometric(Vᴴ; side = :right)
        @test LinearAlgebra.norm(A * adjoint(Nᴴ)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(eltype(T))
        @test isisometric(Nᴴ; side = :right)
        @test isrightcomplete(Vᴴ, Nᴴ)

        M = LinearMap(A)
        CM, VMᴴ = @testinferred _right_orth_svd(M)
        @test parent(CM) * parent(VMᴴ) ≈ A

        # passing a kind and some kwargs
        C, Vᴴ = @testinferred _right_orth_lq(A; positive = true)
        Nᴴ = @testinferred right_null(A; alg = :lq, positive = true)
        @test C isa typeof(A) && size(C) == (m, minmn)
        @test Vᴴ isa typeof(A) && size(Vᴴ) == (minmn, n)
        @test eltype(Nᴴ) == eltype(A) && size(Nᴴ) == (n - minmn, n)
        @test C * Vᴴ ≈ A
        @test isisometric(Vᴴ; side = :right)
        @test LinearAlgebra.norm(A * adjoint(Nᴴ)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(eltype(T))
        @test isisometric(Nᴴ; side = :right)
        @test isrightcomplete(Vᴴ, Nᴴ)

        # passing an algorithm
        C, Vᴴ = @testinferred right_orth(A; alg = MatrixAlgebraKit.default_lq_algorithm(A))
        Nᴴ = @testinferred right_null(A; alg = :lq, positive = true)
        @test C isa typeof(A) && size(C) == (m, minmn)
        @test Vᴴ isa typeof(A) && size(Vᴴ) == (minmn, n)
        @test eltype(Nᴴ) == eltype(A) && size(Nᴴ) == (n - minmn, n)
        @test C * Vᴴ ≈ A
        @test isisometric(Vᴴ; side = :right)
        @test LinearAlgebra.norm(A * adjoint(Nᴴ)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(eltype(T))
        @test isisometric(Nᴴ; side = :right)
        @test isrightcomplete(Vᴴ, Nᴴ)

        C2, Vᴴ2 = @testinferred right_orth!(copy!(Ac, A), (C, Vᴴ))
        Nᴴ2 = @testinferred right_null!(copy!(Ac, A), Nᴴ)
        @test C2 * Vᴴ2 ≈ A
        @test isisometric(Vᴴ2; side = :right)
        @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(eltype(T))
        @test isisometric(Nᴴ; side = :right)
        @test isrightcomplete(Vᴴ2, Nᴴ2)

        for alg in (:lq, :polar, :svd)
            n < m && alg == :polar && continue
            if alg == :lq
                C2, Vᴴ2 = @testinferred _right_orth_lq!(copy!(Ac, A), (C, Vᴴ))
            elseif alg == :polar
                C2, Vᴴ2 = @testinferred _right_orth_polar!(copy!(Ac, A), (C, Vᴴ))
            elseif alg == :svd
                C2, Vᴴ2 = @testinferred _right_orth_svd!(copy!(Ac, A), (C, Vᴴ))
            end
            @test C2 * Vᴴ2 ≈ A
            @test isisometric(Vᴴ2; side = :right)
            if alg != :polar
                Nᴴ2 = @testinferred right_null!(copy!(Ac, A), Nᴴ; alg = alg)
                @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(eltype(T))
                @test isisometric(Nᴴ2; side = :right)
                @test isrightcomplete(Vᴴ2, Nᴴ2)
            end

            if alg != :svd
                @test_throws ArgumentError right_orth!(copy!(Ac, A), (C, Vᴴ); alg, trunc = (; atol))
                @test_throws ArgumentError right_orth!(copy!(Ac, A), (C, Vᴴ); alg, trunc = (; rtol))
                alg == :polar && continue
                @test_throws ArgumentError right_null!(copy!(Ac, A), Nᴴ; alg, trunc = (; atol))
                @test_throws ArgumentError right_null!(copy!(Ac, A), Nᴴ; alg, trunc = (; rtol))
            end
        end
    end
end
