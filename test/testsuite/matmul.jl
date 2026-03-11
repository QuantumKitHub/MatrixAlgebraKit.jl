using TestExtras

function test_strided_batched_mul(T::Type, sz; kwargs...)
    m, p, n, batch = sz
    summary_str = testargs_summary(T, (m, p, n, batch))
    return @testset "strided_batched_mul $summary_str" begin
        A = instantiate_matrix(T, (m, p, batch))
        B = instantiate_matrix(T, (p, n, batch))
        Ac = deepcopy(A)
        Bc = deepcopy(B)

        # out-of-place
        C = @testinferred strided_batched_mul(A, B)
        @test size(C) == (m, n, batch)
        for k in 1:batch
            @test C[:, :, k] ≈ A[:, :, k] * B[:, :, k]
        end
        @test A == Ac
        @test B == Bc

        # in-place: alpha=1, beta=0
        Te = promote_type(eltype(T), Float64)
        C2 = similar(A, Te, (m, n, batch))
        C3 = @testinferred strided_batched_mul!(C2, A, B, true, false)
        @test C3 === C2
        for k in 1:batch
            @test C3[:, :, k] ≈ A[:, :, k] * B[:, :, k]
        end

        # in-place: explicit alpha and beta
        alpha = Te(2)
        fill!(C2, zero(Te))
        C4 = @testinferred strided_batched_mul!(C2, A, B, alpha, false)
        @test C4 === C2
        for k in 1:batch
            @test C4[:, :, k] ≈ alpha * A[:, :, k] * B[:, :, k]
        end
    end
end

function test_strided_batched_mul_algs(T::Type, sz, algs; kwargs...)
    m, p, n, batch = sz
    summary_str = testargs_summary(T, (m, p, n, batch))
    return @testset "strided_batched_mul algorithms $summary_str" begin
        A = instantiate_matrix(T, (m, p, batch))
        B = instantiate_matrix(T, (p, n, batch))
        Cref = strided_batched_mul(A, B)
        for alg in algs
            @testset "$alg" begin
                C = @testinferred strided_batched_mul(A, B, alg)
                @test C ≈ Cref
            end
        end
    end
end

function test_batched_mul(T::Type, sz; kwargs...)
    m, p, n, batch = sz
    summary_str = testargs_summary(T, (m, p, n, batch))
    return @testset "batched_mul $summary_str" begin
        As = [instantiate_matrix(T, (m, p)) for _ in 1:batch]
        Bs = [instantiate_matrix(T, (p, n)) for _ in 1:batch]
        Acs = deepcopy(As)
        Bcs = deepcopy(Bs)

        # out-of-place
        Cs = @testinferred batched_mul(As, Bs)
        @test length(Cs) == batch
        for k in 1:batch
            @test size(Cs[k]) == (m, n)
            @test Cs[k] ≈ As[k] * Bs[k]
        end
        @test all(As .== Acs)
        @test all(Bs .== Bcs)

        # in-place: alpha=1, beta=0
        Te = promote_type(eltype(T), Float64)
        Cs2 = [similar(As[k], Te, (m, n)) for k in 1:batch]
        Cs3 = @testinferred batched_mul!(Cs2, As, Bs, true, false)
        @test Cs3 === Cs2
        for k in 1:batch
            @test Cs3[k] ≈ As[k] * Bs[k]
        end

        # in-place: explicit alpha and beta
        alpha = Te(2)
        for C in Cs2
            fill!(C, zero(Te))
        end
        Cs4 = @testinferred batched_mul!(Cs2, As, Bs, alpha, false)
        @test Cs4 === Cs2
        for k in 1:batch
            @test Cs4[k] ≈ alpha * As[k] * Bs[k]
        end
    end
end

function test_batched_mul_algs(T::Type, sz, algs; kwargs...)
    m, p, n, batch = sz
    summary_str = testargs_summary(T, (m, p, n, batch))
    return @testset "batched_mul algorithms $summary_str" begin
        As = [instantiate_matrix(T, (m, p)) for _ in 1:batch]
        Bs = [instantiate_matrix(T, (p, n)) for _ in 1:batch]
        Csref = batched_mul(As, Bs)
        for alg in algs
            @testset "$alg" begin
                Cs = @testinferred batched_mul(As, Bs, alg)
                @test all(Cs .≈ Csref)
            end
        end
    end
end
