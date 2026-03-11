using TestExtras

function test_strided_batched_mul(T::Type, sz; kwargs...)
    m, p, n, batch = sz
    summary_str = testargs_summary(T, (m, p, n, batch))
    return @testset "strided_batched_mul $summary_str" begin
        A = instantiate_matrix(T, (m, p, batch))
        B = instantiate_matrix(T, (p, n, batch))
        Te = promote_type(eltype(T), Float64)
        C = similar(A, Te, (m, n, batch))

        # in-place: alpha=1, beta=0
        C2 = @testinferred strided_batched_mul!(C, A, B, true, false)
        @test C2 === C
        for k in 1:batch
            @test C2[:, :, k] ≈ A[:, :, k] * B[:, :, k]
        end

        # in-place: explicit alpha and beta
        alpha = Te(2)
        fill!(C, zero(Te))
        C3 = @testinferred strided_batched_mul!(C, A, B, alpha, false)
        @test C3 === C
        for k in 1:batch
            @test C3[:, :, k] ≈ alpha * A[:, :, k] * B[:, :, k]
        end
    end
end

function test_strided_batched_mul_algs(T::Type, sz, algs; kwargs...)
    m, p, n, batch = sz
    summary_str = testargs_summary(T, (m, p, n, batch))
    return @testset "strided_batched_mul algorithms $summary_str" begin
        A = instantiate_matrix(T, (m, p, batch))
        B = instantiate_matrix(T, (p, n, batch))
        Te = promote_type(eltype(T), Float64)
        Cref = similar(A, Te, (m, n, batch))
        strided_batched_mul!(Cref, A, B, true, false)
        for alg in algs
            @testset "$alg" begin
                C = similar(A, Te, (m, n, batch))
                @testinferred strided_batched_mul!(C, A, B, true, false, alg)
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
        Te = promote_type(eltype(T), Float64)
        Cs = [similar(As[k], Te, (m, n)) for k in 1:batch]

        # in-place: alpha=1, beta=0
        Cs2 = @testinferred batched_mul!(Cs, As, Bs, true, false)
        @test Cs2 === Cs
        for k in 1:batch
            @test Cs2[k] ≈ As[k] * Bs[k]
        end

        # in-place: explicit alpha and beta
        alpha = Te(2)
        for C in Cs
            fill!(C, zero(Te))
        end
        Cs3 = @testinferred batched_mul!(Cs, As, Bs, alpha, false)
        @test Cs3 === Cs
        for k in 1:batch
            @test Cs3[k] ≈ alpha * As[k] * Bs[k]
        end
    end
end

function test_batched_mul_algs(T::Type, sz, algs; kwargs...)
    m, p, n, batch = sz
    summary_str = testargs_summary(T, (m, p, n, batch))
    return @testset "batched_mul algorithms $summary_str" begin
        As = [instantiate_matrix(T, (m, p)) for _ in 1:batch]
        Bs = [instantiate_matrix(T, (p, n)) for _ in 1:batch]
        Te = promote_type(eltype(T), Float64)
        Csref = [similar(As[k], Te, (m, n)) for k in 1:batch]
        batched_mul!(Csref, As, Bs, true, false)
        for alg in algs
            @testset "$alg" begin
                Cs = [similar(As[k], Te, (m, n)) for k in 1:batch]
                @testinferred batched_mul!(Cs, As, Bs, true, false, alg)
                @test all(Cs .≈ Csref)
            end
        end
    end
end
