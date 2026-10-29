"""Test omitted bases for QR/LQ algorithms and full factorizations."""
function test_orth_without_basis(T::Type, sz, qr_algs, lq_algs)
    summary_str = testargs_summary(T, sz)
    return @testset "Orthogonalization without a basis $summary_str" begin
        A = instantiate_matrix(T, sz)
        Q0 = similar(A, 0, 0)
        for alg in qr_algs
            Q, R = left_orth(A; alg)
            Q2, R2 = left_orth!(copy(A), (Q0, similar(R)), alg)
            @test Q2 === Q0
            @test R2 ≈ R
            @test Q * R2 ≈ A

            # LQ pivoting is supported through transposed QR.
            lq_alg = LQViaTransposedQR(alg)
            L, Q = right_orth(A; alg = lq_alg)
            L2, Q2 = right_orth!(copy(A), (similar(L), Q0), lq_alg)
            @test Q2 === Q0
            @test L2 ≈ L
            @test L2 * Q ≈ A
        end
        for alg in lq_algs
            L, Q = right_orth(A; alg)
            L2, Q2 = right_orth!(copy(A), (similar(L), Q0), alg)
            @test Q2 === Q0
            @test L2 ≈ L
            @test L2 * Q ≈ A
        end

        alg = Householder(; driver = MatrixAlgebraKit.Native())
        L, _ = right_orth(A; alg)
        L2, Q2 = right_orth!(copy(A), (similar(L), Q0), alg)
        @test Q2 === Q0
        @test L2 ≈ L

        Q, R = qr_full(A)
        Q2, R2 = qr_full!(copy(A), (Q0, similar(R)))
        @test Q2 === Q0
        @test R2 ≈ R
        L, Q = lq_full(A)
        L2, Q2 = lq_full!(copy(A), (similar(L), Q0))
        @test Q2 === Q0
        @test L2 ≈ L
    end
end

"""Test omitted bases for diagonal factorizations."""
function test_orth_without_basis(T::Type{<:Diagonal}, sz)
    summary_str = testargs_summary(T, sz)
    return @testset "Diagonal orthogonalization without a basis $summary_str" begin
        for positive in (false, true)
            A = instantiate_matrix(T, sz)
            Q0 = Diagonal(similar(parent(A), 0))
            Q, R = left_orth(A; positive)
            Q2, R2 = left_orth!(copy(A), (Q0, similar(R)); positive)
            @test Q2 === Q0
            @test Q * R2 ≈ A
            L, Q = right_orth(A; positive)
            L2, Q2 = right_orth!(copy(A), (similar(L), Q0); positive)
            @test Q2 === Q0
            @test L2 * Q ≈ A
        end
    end
end

"""Test omitted corestrictions for blocked and unblocked QR/LQ."""
function test_orth_basis_only(T::Type, sz)
    summary_str = testargs_summary(T, sz)
    return @testset "Orthogonalization with only a basis $summary_str" begin
        for positive in (false, true), blocksize in (1, 2)
            A = instantiate_matrix(T, sz)
            empty_factor = similar(A, 0, 0)
            Q, R = left_orth(A; positive, blocksize)
            Q2, R2 = left_orth!(copy(A), (similar(Q), empty_factor); positive, blocksize)
            @test R2 === empty_factor
            @test Q2 ≈ Q
            @test Q2 * R ≈ A

            L, Q = right_orth(A; positive, blocksize)
            L2, Q2 = right_orth!(copy(A), (empty_factor, similar(Q)); positive, blocksize)
            @test L2 === empty_factor
            @test Q2 ≈ Q
            @test L * Q2 ≈ A
        end
    end
end

"""Test full L-only LQ via transposed QR for a wide input."""
function test_lq_full_without_basis(T::Type, sz)
    summary_str = testargs_summary(T, sz)
    return @testset "Full L-only LQ via transposed QR $summary_str" begin
        A = instantiate_matrix(T, sz)
        alg = LQViaTransposedQR(Householder())
        L, Q = lq_full(A, alg)
        Q0 = similar(A, 0, 0)
        L2, Q2 = lq_full!(copy(A), (similar(L), Q0), alg)
        @test Q2 === Q0
        @test L2 ≈ L
        @test L2 * Q ≈ A
    end
end
