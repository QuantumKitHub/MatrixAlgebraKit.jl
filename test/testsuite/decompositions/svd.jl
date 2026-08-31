using TestExtras
using GenericLinearAlgebra
using LinearAlgebra: opnorm

function test_svd(T::Type, sz; test_compact::Bool = true, test_full::Bool = true, test_trunc::Bool = true, kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "svd $summary_str" begin
        test_compact && test_svd_compact(T, sz; kwargs...)
        test_full && test_svd_full(T, sz; kwargs...)
        test_trunc && test_svd_trunc(T, sz; kwargs...)
    end
end

function test_svd_batched(T::Type, sz, batch_size::Int; test_compact::Bool = true, test_full::Bool = true, test_trunc::Bool = true, kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "svd batched $summary_str batch_size $batch_size" begin
        test_compact && test_svd_compact_batched(T, sz, batch_size; kwargs...)
        test_full && test_svd_full_batched(T, sz, batch_size; kwargs...)
        # TODO
        #test_trunc && test_svd_trunc(T, sz; kwargs...)
    end
end

function test_svd_algs(T::Type, sz, algs; test_compact::Bool = true, test_full::Bool = true, test_trunc::Bool = true, kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "svd algorithms $summary_str" begin
        test_compact && test_svd_compact_algs(T, sz, algs; kwargs...)
        test_full && test_svd_full_algs(T, sz, algs; kwargs...)
        test_trunc && test_svd_trunc_algs(T, sz, algs; kwargs...)
    end
end

function test_svd_batched_algs(T::Type, sz, batch_size::Int, algs; test_compact::Bool = true, test_full::Bool = true, test_trunc::Bool = true, kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "svd batched algorithms $summary_str batch_size $batch_size" begin
        test_compact && test_svd_compact_algs_batched(T, sz, algs, batch_size; kwargs...)
        test_full && test_svd_full_algs_batched(T, sz, algs, batch_size; kwargs...)
        # TODO
        #test_trunc && test_svd_trunc_algs(T, sz, algs; kwargs...)
    end
end

function test_svd_compact(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(eltype(T)),
        test_vals::Bool = true, kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "svd_compact! $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)
        m, n = size(A)
        minmn = min(m, n)
        U, S, Vᴴ = @testinferred svd_compact(A)
        @test size(U) == (m, minmn)
        @test S isa Diagonal{real(eltype(T))} && size(S) == (minmn, minmn)
        @test size(Vᴴ) == (minmn, n)
        @test U * S * Vᴴ ≈ A
        @test isisometric(U)
        @test isisometric(Vᴴ; side = :right)
        @test isposdef(S)

        Sc = similar(A, real(eltype(T)), min(m, n))
        U2, S2, V2ᴴ = @testinferred svd_compact!(Ac, (U, S, Vᴴ))
        @test U2 * S2 * V2ᴴ ≈ A
        @test isisometric(U2)
        @test isisometric(V2ᴴ; side = :right)
        @test isposdef(S2)

        if test_vals
            Sd = @testinferred svd_vals(A)
            @test S ≈ Diagonal(Sd)
        end
    end
end

function test_svd_compact_batched(
        T::Type, sz, batch_size::Int;
        atol::Real = 0, rtol::Real = precision(eltype(T)),
        test_vals::Bool = true, kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "svd_compact! $summary_str batch_size $batch_size" begin
        As = [instantiate_matrix(T, sz) for bi in 1:batch_size]
        Ad = device_batch(As)
        Ac = deepcopy(Ad)
        m, n = size(first(As))
        minmn = min(m, n)
        U, S, Vᴴ = @testinferred svd_compact(Ad)
        @test size(U) == (m, minmn, batch_size)
        @test S isa AbstractMatrix{real(eltype(T))} && size(S) == (minmn, batch_size)
        @test size(Vᴴ) == (minmn, n, batch_size)
        for (a, u, s, vᴴ) in zip(As, eachslice(U, dims = 3), eachslice(S, dims = 2), eachslice(Vᴴ, dims = 3))
            @test u * Diagonal(s) * vᴴ ≈ a
            @test isisometric(u)
            @test isisometric(vᴴ; side = :right)
            @test isposdef(Diagonal(s))
        end

        Sc = similar(diagview(S))
        U2, S2, V2ᴴ = @testinferred svd_compact!(Ac, (U, S, Vᴴ))
        for (a, u, s, vᴴ) in zip(As, eachslice(U2, dims = 3), eachslice(S2, dims = 2), eachslice(V2ᴴ, dims = 3))
            @test u * Diagonal(s) * vᴴ ≈ a
            @test isisometric(u)
            @test isisometric(vᴴ; side = :right)
            @test isposdef(Diagonal(s))
        end

        if test_vals
            Sd = @testinferred svd_vals(Ad)
            for (s, sd) in zip(eachslice(S, dims = 2), eachslice(Sd, dims = 2))
                @test s ≈ sd
            end
        end
    end
end

function test_svd_compact_algs(
        T::Type, sz, algs;
        atol::Real = 0, rtol::Real = precision(eltype(T)),
        test_vals::Bool = true, kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "svd_compact! algorithm $alg $summary_str" for alg in algs
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)
        m, n = size(A)
        minmn = min(m, n)
        U, S, Vᴴ = @testinferred svd_compact(A; alg)
        @test size(U) == (m, minmn)
        @test S isa Diagonal{real(eltype(T))} && size(S) == (minmn, minmn)
        @test size(Vᴴ) == (minmn, n)
        @test U * S * Vᴴ ≈ A
        @test isisometric(U)
        @test isisometric(Vᴴ; side = :right)
        @test isposdef(S)

        U2, S2, V2ᴴ = @testinferred svd_compact!(Ac, (U, S, Vᴴ); alg)
        @test U2 * S2 * V2ᴴ ≈ A
        @test isisometric(U2)
        @test isisometric(V2ᴴ; side = :right)
        @test isposdef(S2)

        if test_vals
            Sd = @testinferred svd_vals(A; alg)
            @test S ≈ Diagonal(Sd)
        end
    end
end

function test_svd_compact_algs_batched(
        T::Type, sz, algs, batch_size::Int;
        atol::Real = 0, rtol::Real = precision(eltype(T)),
        test_vals::Bool = true, kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "svd_compact! algorithm $alg $summary_str batch_size $batch_size" for alg in algs
        As = [instantiate_matrix(T, sz) for bi in 1:batch_size]
        Ad = device_batch(As)
        Ac = deepcopy(Ad)
        m, n = size(first(As))
        minmn = min(m, n)
        U, S, Vᴴ = @testinferred svd_compact(Ad; alg)
        @test size(U) == (m, minmn, batch_size)
        @test S isa AbstractMatrix{real(eltype(T))} && size(S) == (minmn, batch_size)
        @test size(Vᴴ) == (minmn, n, batch_size)
        for (a, u, s, vᴴ) in zip(As, eachslice(U, dims = 3), eachslice(S, dims = 2), eachslice(Vᴴ, dims = 3))
            @test u * Diagonal(s) * vᴴ ≈ a
            @test isisometric(u)
            @test isisometric(vᴴ; side = :right)
            @test isposdef(Diagonal(s))
        end

        U2, S2, V2ᴴ = @testinferred svd_compact!(Ac, (U, S, Vᴴ); alg)
        for (a, u, s, vᴴ) in zip(As, eachslice(U2, dims = 3), eachslice(S2, dims = 2), eachslice(V2ᴴ, dims = 3))
            @test u * Diagonal(s) * vᴴ ≈ a
            @test isisometric(u)
            @test isisometric(vᴴ; side = :right)
            @test isposdef(Diagonal(s))
        end

        if test_vals
            Sd = @testinferred svd_vals(Ad; alg)
            for (s, sd) in zip(eachslice(S, dims = 2), eachslice(Sd, dims = 2))
                @test s ≈ sd
            end
        end
    end
end

function test_svd_full(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(eltype(T)),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "svd_full! $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)
        m, n = size(A)
        minmn = min(m, n)

        U, S, Vᴴ = @testinferred svd_full(A)
        @test size(U) == (m, m)
        @test eltype(S) == real(eltype(T)) && size(S) == (m, n)
        @test size(Vᴴ) == (n, n)
        @test U * S * Vᴴ ≈ A
        @test isunitary(U)
        @test isunitary(Vᴴ)
        @test all(isposdef, diagview(S))

        U2, S2, V2ᴴ = @testinferred svd_full!(Ac, (U, S, Vᴴ))
        @test U2 * S2 * V2ᴴ ≈ A
        @test isunitary(U2)
        @test isunitary(V2ᴴ)
        @test all(isposdef, diagview(S2))

        Sc = similar(A, real(eltype(T)), min(m, n))
        Sc2 = @testinferred svd_vals!(copy!(Ac, A), Sc)
        @test collect(diagview(S)) ≈ collect(Sc2)
    end
end

function test_svd_full_batched(
        T::Type, sz, batch_size::Int;
        atol::Real = 0, rtol::Real = precision(eltype(T)),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "svd_full! $summary_str batch_size $batch_size" begin
        As = [instantiate_matrix(T, sz) for bi in 1:batch_size]
        Ad = device_batch(As)
        Ac = deepcopy(Ad)
        m, n = size(first(As))
        minmn = min(m, n)
        U, S, Vᴴ = @testinferred svd_full(Ad)
        @test size(U) == (m, m, batch_size)
        @test S isa AbstractArray{real(eltype(T)), 3} && size(S) == (m, n, batch_size)
        @test size(Vᴴ) == (n, n, batch_size)
        for (a, u, s, vᴴ) in zip(As, eachslice(U, dims = 3), eachslice(S, dims = 3), eachslice(Vᴴ, dims = 3))
            @test u * s * vᴴ ≈ a
            @test isunitary(u)
            @test isunitary(vᴴ)
            @test all(isposdef, diagview(s))
        end

        U2, S2, V2ᴴ = @testinferred svd_full!(Ac, (U, S, Vᴴ))
        for (a, u, s, vᴴ) in zip(As, eachslice(U2, dims = 3), eachslice(S2, dims = 3), eachslice(V2ᴴ, dims = 3))
            @test u * s * vᴴ ≈ a
            @test isunitary(u)
            @test isunitary(vᴴ)
            @test all(isposdef, diagview(s))
        end

        Sc = similar(first(As), real(eltype(T)), min(m, n), batch_size)
        Sc2 = @testinferred svd_vals!(copy!(Ac, Ad), Sc)
        for (s, s2) in zip(eachslice(S, dims = 3), eachslice(Sc, dims = 2))
            @test collect(diagview(s)) ≈ collect(s2)
        end
    end
end

function test_svd_full_algs(
        T::Type, sz, algs;
        atol::Real = 0, rtol::Real = precision(eltype(T)),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "svd_full! algorithm $alg $summary_str" for alg in algs
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)
        m, n = size(A)
        minmn = min(m, n)

        U, S, Vᴴ = @testinferred svd_full(A; alg)
        @test size(U) == (m, m)
        @test eltype(S) == real(eltype(T)) && size(S) == (m, n)
        @test size(Vᴴ) == (n, n)
        @test U * S * Vᴴ ≈ A
        @test isunitary(U)
        @test isunitary(Vᴴ)
        @test all(isposdef, diagview(S))

        U2, S2, V2ᴴ = @testinferred svd_full!(Ac, (U, S, Vᴴ); alg)
        @test U2 * S2 * V2ᴴ ≈ A
        @test isunitary(U2)
        @test isunitary(V2ᴴ)
        @test all(isposdef, diagview(S2))

        Sc = similar(A, real(eltype(T)), min(m, n))
        Sc2 = @testinferred svd_vals!(copy!(Ac, A), Sc; alg)
        @test collect(diagview(S)) ≈ collect(Sc2)
    end
end

function test_svd_full_algs_batched(
        T::Type, sz, algs, batch_size::Int;
        atol::Real = 0, rtol::Real = precision(eltype(T)),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "svd_full! algorithm $alg $summary_str batch_size $batch_size" for alg in algs
        As = [instantiate_matrix(T, sz) for bi in 1:batch_size]
        Ad = device_batch(As)
        Ac = deepcopy(Ad)
        m, n = size(first(As))
        minmn = min(m, n)
        U, S, Vᴴ = @testinferred svd_full(Ad; alg)
        @test size(U) == (m, m, batch_size)
        @test S isa AbstractArray{real(eltype(T)), 3} && size(S) == (m, n, batch_size)
        @test size(Vᴴ) == (n, n, batch_size)
        for (a, u, s, vᴴ) in zip(As, eachslice(U, dims = 3), eachslice(S, dims = 3), eachslice(Vᴴ, dims = 3))
            @test u * s * vᴴ ≈ a
            @test isunitary(u)
            @test isunitary(vᴴ)
            @test all(isposdef, diagview(s))
        end

        U2, S2, V2ᴴ = @testinferred svd_full!(Ac, (U, S, Vᴴ); alg)
        for (a, u, s, vᴴ) in zip(As, eachslice(U2, dims = 3), eachslice(S2, dims = 3), eachslice(V2ᴴ, dims = 3))
            @test u * s * vᴴ ≈ a
            @test isunitary(u)
            @test isunitary(vᴴ)
            @test all(isposdef, diagview(s))
        end

        Sc = similar(first(As), real(eltype(T)), min(m, n), batch_size)
        Sc2 = @testinferred svd_vals!(copy!(Ac, Ad), Sc; alg)
        for (s, s2) in zip(eachslice(S, dims = 3), eachslice(Sc, dims = 2))
            @test collect(diagview(s)) ≈ collect(s2)
        end
    end
end

function test_svd_trunc(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(eltype(T)),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "svd_trunc! $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)
        m, n = size(A)
        minmn = min(m, n)
        S₀ = collect(svd_vals(A))
        r = minmn - 2

        if m > 0 && n > 0
            U1, S1, V1ᴴ, ϵ1 = @testinferred svd_trunc(A; trunc = truncrank(r))
            @test length(diagview(S1)) == r
            @test collect(diagview(S1)) ≈ S₀[1:r]
            AUSV_vals = svd_vals(A - U1 * S1 * V1ᴴ) # bypass broken svdvals on AMDGPU
            @test mapreduce(sv -> opnorm(sv, 2), max, AUSV_vals) ≈ S₀[r + 1]
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
        end

        @testset "mix maxrank and tol" begin
            m4 = 4
            U = instantiate_unitary(T, A, m4)
            Sdiag = similar(A, real(eltype(T)), m4)
            copyto!(Sdiag, [0.9, 0.3, 0.1, 0.01])
            S = Diagonal(Sdiag)
            Vᴴ = instantiate_unitary(T, A, m4)
            A = U * S * Vᴴ
            for trunc_fun in (
                    (rtol, maxrank) -> (; rtol, maxrank),
                    (rtol, maxrank) -> truncrank(maxrank) & trunctol(; rtol),
                )
                U1, S1, V1ᴴ, ϵ1 = svd_trunc(A; trunc = trunc_fun(0.2, 1))
                @test length(diagview(S1)) == 1
                @test diagview(S1) ≈ diagview(S)[1:1]

                U2, S2, V2ᴴ = svd_trunc_no_error(A; trunc = trunc_fun(0.2, 3))
                @test length(diagview(S2)) == 2
                @test diagview(S2) ≈ diagview(S)[1:2]
            end
        end
        @testset "mix minrank and tol" begin
            m4 = 4
            U = instantiate_unitary(T, A, m4)
            Sdiag = similar(A, real(eltype(T)), m4)
            copyto!(Sdiag, [0.9, 0.3, 0.1, 0.01])
            S = Diagonal(Sdiag)
            Vᴴ = instantiate_unitary(T, A, m4)
            A = U * S * Vᴴ
            for trunc_fun in (
                    (rtol, minrank) -> (; rtol, minrank),
                    (rtol, minrank) -> trunctol(; rtol) | truncrank(minrank),
                )
                # trunctol(rtol=0.5) keeps 1 value, truncrank(3) keeps 3, union keeps 3
                U1, S1, V1ᴴ, ϵ1 = svd_trunc(A; trunc = trunc_fun(0.5, 3))
                @test length(diagview(S1)) == 3
                @test diagview(S1) ≈ diagview(S)[1:3]

                # trunctol(rtol=0.2) keeps 2 values, truncrank(1) keeps 1, union keeps 2
                U2, S2, V2ᴴ = svd_trunc_no_error(A; trunc = trunc_fun(0.2, 1))
                @test length(diagview(S2)) == 2
                @test diagview(S2) ≈ diagview(S)[1:2]
            end
        end
        @testset "specify truncation algorithm" begin
            atol = sqrt(eps(real(eltype(T))))
            m4 = 4
            U = instantiate_unitary(T, A, m4)
            Sdiag = similar(A, real(eltype(T)), m4)
            copyto!(Sdiag, [0.9, 0.3, 0.1, 0.01])
            Vᴴ = instantiate_unitary(T, A, m4)
            S = Diagonal(Sdiag)
            A = U * S * Vᴴ
            alg = TruncatedAlgorithm(MatrixAlgebraKit.default_svd_algorithm(A), trunctol(; atol = 0.2))
            U2, S2, V2ᴴ, ϵ2 = @testinferred svd_trunc(A; alg)
            @test diagview(S2) ≈ diagview(S)[1:2]
            @test ϵ2 ≈ norm(diagview(S)[3:4]) atol = atol
            @test_throws ArgumentError svd_trunc(A; alg, trunc = (; maxrank = 2))
            @test_throws ArgumentError svd_trunc_no_error(A; alg, trunc = (; maxrank = 2))
        end
    end
end

function test_svd_trunc_algs(
        T::Type, sz, algs;
        atol::Real = 0, rtol::Real = precision(eltype(T)),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "svd_trunc! algorithm $alg $summary_str" for alg in algs
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)
        m, n = size(A)
        minmn = min(m, n)
        S₀ = collect(svd_vals(A))
        r = minmn - 2

        if m > 0 && n > 0
            U1, S1, V1ᴴ, ϵ1 = @testinferred svd_trunc(A; trunc = truncrank(r), alg)
            @test length(diagview(S1)) == r
            @test collect(diagview(S1)) ≈ S₀[1:r]
            AUSV_vals = svd_vals(A - U1 * S1 * V1ᴴ) # bypass broken svdvals on AMDGPU
            @test mapreduce(sv -> opnorm(sv, 2), max, AUSV_vals) ≈ S₀[r + 1]
            # Test truncation error
            @test ϵ1 ≈ norm(view(S₀, (r + 1):minmn)) atol = atol

            s = 1 + sqrt(eps(real(eltype(T))))
            trunc = trunctol(; atol = s * S₀[r + 1])

            U2, S2, V2ᴴ, ϵ2 = @testinferred svd_trunc(A; trunc, alg)
            @test length(diagview(S2)) == r
            @test U1 ≈ U2
            @test S1 ≈ S2
            @test V1ᴴ ≈ V2ᴴ
            @test ϵ2 ≈ norm(view(S₀, (r + 1):minmn)) atol = atol

            trunc = truncerror(; atol = s * norm(@view(S₀[(r + 1):end])))
            U3, S3, V3ᴴ, ϵ3 = @testinferred svd_trunc(A; trunc, alg)
            @test length(diagview(S3)) == r
            @test U1 ≈ U3
            @test S1 ≈ S3
            @test V1ᴴ ≈ V3ᴴ
            @test ϵ3 ≈ norm(view(S₀, (r + 1):minmn)) atol = atol
        end

        @testset "mix maxrank and tol" begin
            m4 = 4
            U = instantiate_unitary(T, A, m4)
            Sdiag = similar(A, real(eltype(T)), m4)
            copyto!(Sdiag, real(eltype(T))[0.9, 0.3, 0.1, 0.01])
            S = Diagonal(Sdiag)
            Vᴴ = instantiate_unitary(T, A, m4)
            A = U * S * Vᴴ
            for trunc_fun in (
                    (rtol, maxrank) -> (; rtol, maxrank),
                    (rtol, maxrank) -> truncrank(maxrank) & trunctol(; rtol),
                )
                U1, S1, V1ᴴ, ϵ1 = svd_trunc(A; trunc = trunc_fun(0.2, 1), alg)
                @test length(diagview(S1)) == 1
                @test collect(diagview(S1)) ≈ collect(diagview(S)[1:1])

                U2, S2, V2ᴴ, ϵ2 = svd_trunc(A; trunc = trunc_fun(0.2, 3), alg)
                @test length(diagview(S2)) == 2
                @test collect(diagview(S2)) ≈ collect(diagview(S)[1:2])
            end
        end
        @testset "mix minrank and tol" begin
            m4 = 4
            U = instantiate_unitary(T, A, m4)
            Sdiag = similar(A, real(eltype(T)), m4)
            copyto!(Sdiag, real(eltype(T))[0.9, 0.3, 0.1, 0.01])
            S = Diagonal(Sdiag)
            Vᴴ = instantiate_unitary(T, A, m4)
            A = U * S * Vᴴ
            for trunc_fun in (
                    (rtol, minrank) -> (; rtol, minrank),
                    (rtol, minrank) -> trunctol(; rtol) | truncrank(minrank),
                )
                # trunctol(rtol=0.5) keeps 1 value, truncrank(3) keeps 3, union keeps 3
                U1, S1, V1ᴴ, ϵ1 = svd_trunc(A; trunc = trunc_fun(0.5, 3), alg)
                @test length(diagview(S1)) == 3
                @test collect(diagview(S1)) ≈ collect(diagview(S)[1:3])

                # trunctol(rtol=0.2) keeps 2 values, truncrank(1) keeps 1, union keeps 2
                U2, S2, V2ᴴ, ϵ2 = svd_trunc(A; trunc = trunc_fun(0.2, 1), alg)
                @test length(diagview(S2)) == 2
                @test collect(diagview(S2)) ≈ collect(diagview(S)[1:2])
            end
        end
        @testset "specify truncation algorithm" begin
            atol = sqrt(eps(real(eltype(T))))
            m4 = 4
            U = instantiate_unitary(T, A, m4)
            Sdiag = similar(A, real(eltype(T)), m4)
            copyto!(Sdiag, real(eltype(T))[0.9, 0.3, 0.1, 0.01])
            S = Diagonal(Sdiag)
            Vᴴ = instantiate_unitary(T, A, m4)
            A = U * S * Vᴴ
            truncalg = TruncatedAlgorithm(alg, trunctol(; atol = 0.2))
            U2, S2, V2ᴴ, ϵ2 = @testinferred svd_trunc(A; alg = truncalg)
            @test collect(diagview(S2)) ≈ collect(diagview(S)[1:2])
            @test ϵ2 ≈ norm(diagview(S)[3:4]) atol = atol
            @test_throws ArgumentError svd_trunc(A; alg = truncalg, trunc = (; maxrank = 2))
            @test_throws ArgumentError svd_trunc_no_error(A; alg = truncalg, trunc = (; maxrank = 2))
        end
    end
end

function test_randomized_svd(T::Type, sz, algs; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "randomized svd_trunc! algorithm $alg $summary_str" for alg in algs
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)
        m, n = size(A)
        minmn = min(m, n)
        S₀ = collect(svd_vals(A))
        U1, S1, V1ᴴ, ϵ1 = @testinferred svd_trunc(A; alg)
        @test collect(diagview(S1))[1:alg.alg.k] ≈ S₀[1:alg.alg.k]
    end
end
