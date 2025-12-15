using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using ChainRulesCore, ChainRulesTestUtils, Zygote
using MatrixAlgebraKit: diagview, TruncatedAlgorithm, PolarViaSVD
using LinearAlgebra: UpperTriangular, Diagonal, Hermitian, mul!

include("ad_utils.jl")

for f in
    (
        :qr_compact, :qr_full, :qr_null, :lq_compact, :lq_full, :lq_null,
        :eig_full, :eig_trunc, :eig_vals, :eigh_full, :eigh_trunc, :eigh_vals,
        :svd_compact, :svd_trunc, :svd_trunc_no_error, :svd_vals,
        :left_polar, :right_polar,
    )
    copy_f = Symbol(:copy_, f)
    f! = Symbol(f, '!')
    _hermitian = startswith(string(f), "eigh")
    @eval begin
        function $copy_f(input, alg)
            if $_hermitian
                input = (input + input') / 2
            end
            return $f(input, alg)
        end
        function ChainRulesCore.rrule(::typeof($copy_f), input, alg)
            output = MatrixAlgebraKit.initialize_output($f!, input, alg)
            if $_hermitian
                input = (input + input') / 2
            else
                input = copy(input)
            end
            output, pb = ChainRulesCore.rrule($f!, input, output, alg)
            return output, x -> (NoTangent(), pb(x)[2], NoTangent())
        end
    end
end

@timedtestset "QR AD Rules with eltype $T" for T in (Float64, ComplexF64, Float32)
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        # qr_compact
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        minmn = min(m, n)
        alg = LAPACK_HouseholderQR(; positive = true)
        Q, R = copy_qr_compact(A, alg)
        ΔQ = randn(rng, T, m, minmn)
        ΔR = randn(rng, T, minmn, n)
        ΔR2 = UpperTriangular(randn(rng, T, minmn, minmn))
        ΔN = Q * randn(rng, T, minmn, max(0, m - minmn))
        test_rrule(
            copy_qr_compact, A, alg ⊢ NoTangent();
            output_tangent = (ΔQ, ΔR), atol = atol, rtol = rtol
        )
        test_rrule(
            copy_qr_null, A, alg ⊢ NoTangent();
            output_tangent = ΔN, atol = atol, rtol = rtol
        )
        config = Zygote.ZygoteRuleConfig()
        test_rrule(
            config, qr_compact, A;
            fkwargs = (; positive = true), output_tangent = (ΔQ, ΔR),
            atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
        test_rrule(
            config, first ∘ qr_compact, A;
            fkwargs = (; positive = true), output_tangent = ΔQ,
            atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
        test_rrule(
            config, last ∘ qr_compact, A;
            fkwargs = (; positive = true), output_tangent = ΔR,
            atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
        test_rrule(
            config, qr_null, A;
            fkwargs = (; positive = true), output_tangent = ΔN,
            atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
        # qr_full
        Q, R = copy_qr_full(A, alg)
        Q1 = view(Q, 1:m, 1:minmn)
        ΔQ = randn(rng, T, m, m)
        ΔQ2 = view(ΔQ, :, (minmn + 1):m)
        mul!(ΔQ2, Q1, Q1' * ΔQ2)
        ΔR = randn(rng, T, m, n)
        test_rrule(
            copy_qr_full, A, alg ⊢ NoTangent();
            output_tangent = (ΔQ, ΔR), atol = atol, rtol = rtol
        )
        config = Zygote.ZygoteRuleConfig()
        test_rrule(
            config, qr_full, A;
            fkwargs = (; positive = true), output_tangent = (ΔQ, ΔR),
            atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
        if m > n
            _, null_pb = Zygote.pullback(qr_null, A, alg)
            @test_logs (:warn,) null_pb(randn(rng, T, m, max(0, m - minmn)))
            _, full_pb = Zygote.pullback(qr_full, A, alg)
            @test_logs (:warn,) full_pb((randn(rng, T, m, m), randn(rng, T, m, n)))
        end
        # rank-deficient A
        r = minmn - 5
        A = randn(rng, T, m, r) * randn(rng, T, r, n)
        Q, R = qr_compact(A, alg)
        ΔQ = randn(rng, T, m, minmn)
        Q1 = view(Q, 1:m, 1:r)
        Q2 = view(Q, 1:m, (r + 1):minmn)
        ΔQ2 = view(ΔQ, 1:m, (r + 1):minmn)
        ΔQ2 .= 0
        ΔR = randn(rng, T, minmn, n)
        view(ΔR, (r + 1):minmn, :) .= 0
        test_rrule(
            copy_qr_compact, A, alg ⊢ NoTangent();
            output_tangent = (ΔQ, ΔR), atol = atol, rtol = rtol
        )
        config = Zygote.ZygoteRuleConfig()
        test_rrule(
            config, qr_compact, A;
            fkwargs = (; positive = true), output_tangent = (ΔQ, ΔR),
            atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
    end
end

@timedtestset "LQ AD Rules with eltype $T" for T in (Float64, ComplexF64, Float32)
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        # lq_compact
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        minmn = min(m, n)
        alg = LAPACK_HouseholderLQ(; positive = true)
        L, Q = copy_lq_compact(A, alg)
        ΔL = randn(rng, T, m, minmn)
        ΔQ = randn(rng, T, minmn, n)
        ΔNᴴ = randn(rng, T, max(0, n - minmn), minmn) * Q
        test_rrule(
            copy_lq_compact, A, alg ⊢ NoTangent();
            output_tangent = (ΔL, ΔQ), atol = atol, rtol = rtol
        )
        test_rrule(
            copy_lq_null, A, alg ⊢ NoTangent();
            output_tangent = ΔNᴴ, atol = atol, rtol = rtol
        )
        config = Zygote.ZygoteRuleConfig()
        test_rrule(
            config, lq_compact, A;
            fkwargs = (; positive = true), output_tangent = (ΔL, ΔQ),
            atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
        test_rrule(
            config, first ∘ lq_compact, A;
            fkwargs = (; positive = true), output_tangent = ΔL,
            atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
        test_rrule(
            config, last ∘ lq_compact, A;
            fkwargs = (; positive = true), output_tangent = ΔQ,
            atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
        test_rrule(
            config, lq_null, A;
            fkwargs = (; positive = true), output_tangent = ΔNᴴ,
            atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
        # lq_full
        L, Q = copy_lq_full(A, alg)
        Q1 = view(Q, 1:minmn, 1:n)
        ΔQ = randn(rng, T, n, n)
        ΔQ2 = view(ΔQ, (minmn + 1):n, 1:n)
        mul!(ΔQ2, ΔQ2 * Q1', Q1)
        ΔL = randn(rng, T, m, n)
        test_rrule(
            copy_lq_full, A, alg ⊢ NoTangent();
            output_tangent = (ΔL, ΔQ), atol = atol, rtol = rtol
        )
        config = Zygote.ZygoteRuleConfig()
        test_rrule(
            config, lq_full, A;
            fkwargs = (; positive = true), output_tangent = (ΔL, ΔQ),
            atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
        if m < n
            Nᴴ, null_pb = Zygote.pullback(lq_null, A, alg)
            @test_logs (:warn,) null_pb(randn(rng, T, max(0, n - minmn), n))
            _, full_pb = Zygote.pullback(lq_full, A, alg)
            @test_logs (:warn,) full_pb((randn(rng, T, m, n), randn(rng, T, n, n)))
        end
        # rank-deficient A
        r = minmn - 5
        A = randn(rng, T, m, r) * randn(rng, T, r, n)
        L, Q = lq_compact(A, alg)
        ΔL = randn(rng, T, m, minmn)
        ΔQ = randn(rng, T, minmn, n)
        Q1 = view(Q, 1:r, 1:n)
        Q2 = view(Q, (r + 1):minmn, 1:n)
        ΔQ2 = view(ΔQ, (r + 1):minmn, 1:n)
        ΔQ2 .= 0
        view(ΔL, :, (r + 1):minmn) .= 0
        test_rrule(
            copy_lq_compact, A, alg ⊢ NoTangent();
            output_tangent = (ΔL, ΔQ), atol = atol, rtol = rtol
        )
        config = Zygote.ZygoteRuleConfig()
        test_rrule(
            config, lq_compact, A;
            fkwargs = (; positive = true), output_tangent = (ΔL, ΔQ),
            atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
    end
end

@timedtestset "EIG AD Rules with eltype $T" for T in (Float64, ComplexF64, Float32)
    rng = StableRNG(12345)
    m = 19
    atol = rtol = m * m * precision(T)
    A = randn(rng, T, m, m)
    D, V = eig_full(A)
    Ddiag = diagview(D)
    ΔV = randn(rng, complex(T), m, m)
    ΔV = remove_eiggauge_dependence!(ΔV, D, V; degeneracy_atol = atol)
    ΔD = randn(rng, complex(T), m, m)
    ΔD2 = Diagonal(randn(rng, complex(T), m))
    for alg in (LAPACK_Simple(), LAPACK_Expert())
        test_rrule(
            copy_eig_full, A, alg ⊢ NoTangent(); output_tangent = (ΔD, ΔV), atol, rtol
        )
        test_rrule(
            copy_eig_full, A, alg ⊢ NoTangent(); output_tangent = (ΔD2, ΔV), atol, rtol
        )
        test_rrule(
            copy_eig_vals, A, alg ⊢ NoTangent(); output_tangent = diagview(ΔD), atol, rtol
        )
        for r in 1:4:m
            truncalg = TruncatedAlgorithm(alg, truncrank(r; by = abs))
            ind = MatrixAlgebraKit.findtruncated(Ddiag, truncalg.trunc)
            Dtrunc = Diagonal(diagview(D)[ind])
            Vtrunc = V[:, ind]
            ΔDtrunc = Diagonal(diagview(ΔD2)[ind])
            ΔVtrunc = ΔV[:, ind]
            test_rrule(
                copy_eig_trunc, A, truncalg ⊢ NoTangent();
                output_tangent = (ΔDtrunc, ΔVtrunc, zero(real(T))),
                atol = atol, rtol = rtol
            )
            dA1 = MatrixAlgebraKit.eig_pullback!(zero(A), A, (D, V), (ΔDtrunc, ΔVtrunc), ind)
            dA2 = MatrixAlgebraKit.eig_trunc_pullback!(zero(A), A, (Dtrunc, Vtrunc), (ΔDtrunc, ΔVtrunc))
            @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
        end
        truncalg = TruncatedAlgorithm(alg, truncrank(5; by = real))
        ind = MatrixAlgebraKit.findtruncated(Ddiag, truncalg.trunc)
        Dtrunc = Diagonal(Ddiag[ind])
        Vtrunc = V[:, ind]
        ΔDtrunc = Diagonal(diagview(ΔD2)[ind])
        ΔVtrunc = ΔV[:, ind]
        test_rrule(
            copy_eig_trunc, A, truncalg ⊢ NoTangent();
            output_tangent = (ΔDtrunc, ΔVtrunc, zero(real(T))),
            atol = atol, rtol = rtol
        )
        dA1 = MatrixAlgebraKit.eig_pullback!(zero(A), A, (D, V), (ΔDtrunc, ΔVtrunc), ind)
        dA2 = MatrixAlgebraKit.eig_trunc_pullback!(zero(A), A, (Dtrunc, Vtrunc), (ΔDtrunc, ΔVtrunc))
        @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
    end
    # Zygote part
    config = Zygote.ZygoteRuleConfig()
    test_rrule(
        config, eig_full, A;
        output_tangent = (ΔD, ΔV), atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
    )
    test_rrule(
        config, eig_full, A;
        output_tangent = (ΔD2, ΔV), atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
    )
    test_rrule(
        config, first ∘ eig_full, A;
        output_tangent = ΔD, atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
    )
    test_rrule(
        config, last ∘ eig_full, A;
        output_tangent = ΔV, atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
    )
    test_rrule(
        config, eig_vals, A;
        output_tangent = diagview(ΔD), atol, rtol, rrule_f = rrule_via_ad, check_inferred = false
    )
end

@timedtestset "EIGH AD Rules with eltype $T" for T in (Float64, ComplexF64, Float32)
    rng = StableRNG(12345)
    m = 19
    atol = rtol = m * m * precision(T)
    A = randn(rng, T, m, m)
    A = A + A'
    D, V = eigh_full(A)
    Ddiag = diagview(D)
    ΔV = randn(rng, T, m, m)
    ΔV = remove_eighgauge_dependence!(ΔV, D, V; degeneracy_atol = atol)
    ΔD = randn(rng, real(T), m, m)
    ΔD2 = Diagonal(randn(rng, real(T), m))
    for alg in (
            LAPACK_QRIteration(), LAPACK_DivideAndConquer(), LAPACK_Bisection(),
            LAPACK_MultipleRelativelyRobustRepresentations(),
        )
        # copy_eigh_full includes a projector onto the Hermitian part of the matrix
        test_rrule(
            copy_eigh_full, A, alg ⊢ NoTangent(); output_tangent = (ΔD, ΔV), atol, rtol
        )
        test_rrule(
            copy_eigh_full, A, alg ⊢ NoTangent(); output_tangent = (ΔD2, ΔV), atol, rtol
        )
        test_rrule(
            copy_eigh_vals, A, alg ⊢ NoTangent(); output_tangent = diagview(ΔD), atol, rtol
        )
        for r in 1:4:m
            truncalg = TruncatedAlgorithm(alg, truncrank(r; by = abs))
            ind = MatrixAlgebraKit.findtruncated(Ddiag, truncalg.trunc)
            Dtrunc = Diagonal(diagview(D)[ind])
            Vtrunc = V[:, ind]
            ΔDtrunc = Diagonal(diagview(ΔD2)[ind])
            ΔVtrunc = ΔV[:, ind]
            test_rrule(
                copy_eigh_trunc, A, truncalg ⊢ NoTangent();
                output_tangent = (ΔDtrunc, ΔVtrunc, zero(real(T))),
                atol = atol, rtol = rtol
            )
            dA1 = MatrixAlgebraKit.eigh_pullback!(zero(A), A, (D, V), (ΔDtrunc, ΔVtrunc), ind)
            dA2 = MatrixAlgebraKit.eigh_trunc_pullback!(zero(A), A, (Dtrunc, Vtrunc), (ΔDtrunc, ΔVtrunc))
            @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
        end
        truncalg = TruncatedAlgorithm(alg, trunctol(; atol = maximum(abs, Ddiag) / 2))
        ind = MatrixAlgebraKit.findtruncated(Ddiag, truncalg.trunc)
        Dtrunc = Diagonal(diagview(D)[ind])
        Vtrunc = V[:, ind]
        ΔDtrunc = Diagonal(diagview(ΔD2)[ind])
        ΔVtrunc = ΔV[:, ind]
        test_rrule(
            copy_eigh_trunc, A, truncalg ⊢ NoTangent();
            output_tangent = (ΔDtrunc, ΔVtrunc, zero(real(T))),
            atol = atol, rtol = rtol
        )
        dA1 = MatrixAlgebraKit.eigh_pullback!(zero(A), A, (D, V), (ΔDtrunc, ΔVtrunc), ind)
        dA2 = MatrixAlgebraKit.eigh_trunc_pullback!(zero(A), A, (Dtrunc, Vtrunc), (ΔDtrunc, ΔVtrunc))
        @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
    end
    # Zygote part
    config = Zygote.ZygoteRuleConfig()
    # eigh_full does not include a projector onto the Hermitian part of the matrix
    test_rrule(
        config, eigh_full ∘ Matrix ∘ Hermitian, A;
        output_tangent = (ΔD, ΔV), atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
    )
    test_rrule(
        config, eigh_full ∘ Matrix ∘ Hermitian, A;
        output_tangent = (ΔD2, ΔV), atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
    )
    test_rrule(
        config, first ∘ eigh_full ∘ Matrix ∘ Hermitian, A;
        output_tangent = ΔD, atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
    )
    test_rrule(
        config, last ∘ eigh_full ∘ Matrix ∘ Hermitian, A;
        output_tangent = ΔV, atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
    )
    test_rrule(
        config, eigh_vals ∘ Matrix ∘ Hermitian, A;
        output_tangent = diagview(ΔD), atol, rtol, rrule_f = rrule_via_ad, check_inferred = false
    )
    eigh_trunc2(A; kwargs...) = eigh_trunc(Matrix(Hermitian(A)); kwargs...)
    for r in 1:4:m
        trunc = truncrank(r; by = real)
        ind = MatrixAlgebraKit.findtruncated(Ddiag, trunc)
        test_rrule(
            config, eigh_trunc2, A;
            fkwargs = (; trunc = trunc),
            output_tangent = (ΔD[ind, ind], ΔV[:, ind], zero(real(T))),
            atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
    end
    trunc = trunctol(; rtol = 1 / 2)
    ind = MatrixAlgebraKit.findtruncated(Ddiag, trunc)
    test_rrule(
        config, eigh_trunc2, A;
        fkwargs = (; trunc = trunc),
        output_tangent = (ΔD[ind, ind], ΔV[:, ind], zero(real(T))),
        atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
    )
end

@timedtestset "SVD AD Rules with eltype $T" for T in (Float64, ComplexF64, Float32)
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        minmn = min(m, n)
        U, S, Vᴴ = svd_compact(A)
        ΔU = randn(rng, T, m, minmn)
        ΔS = randn(rng, real(T), minmn, minmn)
        ΔS2 = Diagonal(randn(rng, real(T), minmn))
        ΔVᴴ = randn(rng, T, minmn, n)
        ΔU, ΔVᴴ = remove_svdgauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ; degeneracy_atol = atol)
        for alg in (LAPACK_QRIteration(), LAPACK_DivideAndConquer())
            test_rrule(
                copy_svd_compact, A, alg ⊢ NoTangent();
                output_tangent = (ΔU, ΔS, ΔVᴴ), atol = atol, rtol = rtol
            )
            test_rrule(
                copy_svd_compact, A, alg ⊢ NoTangent();
                output_tangent = (ΔU, ΔS2, ΔVᴴ), atol = atol, rtol = rtol
            )
            test_rrule(
                copy_svd_vals, A, alg ⊢ NoTangent();
                output_tangent = diagview(ΔS), atol, rtol
            )
            for r in 1:4:minmn
                truncalg = TruncatedAlgorithm(alg, truncrank(r))
                ind = MatrixAlgebraKit.findtruncated(diagview(S), truncalg.trunc)
                Strunc = Diagonal(diagview(S)[ind])
                Utrunc = U[:, ind]
                Vᴴtrunc = Vᴴ[ind, :]
                ΔStrunc = Diagonal(diagview(ΔS2)[ind])
                ΔUtrunc = ΔU[:, ind]
                ΔVᴴtrunc = ΔVᴴ[ind, :]
                test_rrule(
                    copy_svd_trunc, A, truncalg ⊢ NoTangent();
                    output_tangent = (ΔUtrunc, ΔStrunc, ΔVᴴtrunc, zero(real(T))),
                    atol = atol, rtol = rtol
                )
                test_rrule(
                    copy_svd_trunc_no_error, A, truncalg ⊢ NoTangent();
                    output_tangent = (ΔUtrunc, ΔStrunc, ΔVᴴtrunc),
                    atol = atol, rtol = rtol
                )
                dA1 = MatrixAlgebraKit.svd_pullback!(zero(A), A, (U, S, Vᴴ), (ΔUtrunc, ΔStrunc, ΔVᴴtrunc), ind)
                dA2 = MatrixAlgebraKit.svd_trunc_pullback!(zero(A), A, (Utrunc, Strunc, Vᴴtrunc), (ΔUtrunc, ΔStrunc, ΔVᴴtrunc))
                @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
            end
            truncalg = TruncatedAlgorithm(alg, trunctol(atol = S[1, 1] / 2))
            ind = MatrixAlgebraKit.findtruncated(diagview(S), truncalg.trunc)
            Strunc = Diagonal(diagview(S)[ind])
            Utrunc = U[:, ind]
            Vᴴtrunc = Vᴴ[ind, :]
            ΔStrunc = Diagonal(diagview(ΔS2)[ind])
            ΔUtrunc = ΔU[:, ind]
            ΔVᴴtrunc = ΔVᴴ[ind, :]
            test_rrule(
                copy_svd_trunc, A, truncalg ⊢ NoTangent();
                output_tangent = (ΔUtrunc, ΔStrunc, ΔVᴴtrunc, zero(real(T))),
                atol = atol, rtol = rtol
            )
            test_rrule(
                copy_svd_trunc_no_error, A, truncalg ⊢ NoTangent();
                output_tangent = (ΔUtrunc, ΔStrunc, ΔVᴴtrunc),
                atol = atol, rtol = rtol
            )
            dA1 = MatrixAlgebraKit.svd_pullback!(zero(A), A, (U, S, Vᴴ), (ΔUtrunc, ΔStrunc, ΔVᴴtrunc), ind)
            dA2 = MatrixAlgebraKit.svd_trunc_pullback!(zero(A), A, (Utrunc, Strunc, Vᴴtrunc), (ΔUtrunc, ΔStrunc, ΔVᴴtrunc))
            @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
        end
        # Zygote part
        config = Zygote.ZygoteRuleConfig()
        test_rrule(
            config, svd_compact, A;
            output_tangent = (ΔU, ΔS, ΔVᴴ), atol = atol, rtol = rtol,
            rrule_f = rrule_via_ad, check_inferred = false
        )
        test_rrule(
            config, svd_compact, A;
            output_tangent = (ΔU, ΔS2, ΔVᴴ), atol = atol, rtol = rtol,
            rrule_f = rrule_via_ad, check_inferred = false
        )
        test_rrule(
            config, svd_vals, A;
            output_tangent = diagview(ΔS), atol, rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
        for r in 1:4:minmn
            trunc = truncrank(r)
            ind = MatrixAlgebraKit.findtruncated(diagview(S), trunc)
            test_rrule(
                config, svd_trunc, A;
                fkwargs = (; trunc = trunc),
                output_tangent = (ΔU[:, ind], ΔS[ind, ind], ΔVᴴ[ind, :], zero(real(T))),
                atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
            )
            test_rrule(
                config, svd_trunc_no_error, A;
                fkwargs = (; trunc = trunc),
                output_tangent = (ΔU[:, ind], ΔS[ind, ind], ΔVᴴ[ind, :]),
                atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
            )
        end
        trunc = trunctol(; atol = S[1, 1] / 2)
        ind = MatrixAlgebraKit.findtruncated(diagview(S), trunc)
        test_rrule(
            config, svd_trunc, A;
            fkwargs = (; trunc = trunc),
            output_tangent = (ΔU[:, ind], ΔS[ind, ind], ΔVᴴ[ind, :], zero(real(T))),
            atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
        test_rrule(
            config, svd_trunc_no_error, A;
            fkwargs = (; trunc = trunc),
            output_tangent = (ΔU[:, ind], ΔS[ind, ind], ΔVᴴ[ind, :]),
            atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
    end
end

@timedtestset "Polar AD Rules with eltype $T" for T in (Float64, ComplexF64, Float32)
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        for alg in PolarViaSVD.((LAPACK_QRIteration(), LAPACK_DivideAndConquer()))
            m >= n &&
                test_rrule(copy_left_polar, A, alg ⊢ NoTangent(); atol = atol, rtol = rtol)
            m <= n &&
                test_rrule(copy_right_polar, A, alg ⊢ NoTangent(); atol = atol, rtol = rtol)
        end
        # Zygote part
        config = Zygote.ZygoteRuleConfig()
        m >= n && test_rrule(
            config, left_polar, A;
            atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
        m <= n && test_rrule(
            config, right_polar, A;
            atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
    end
end

@timedtestset "Orth and null with eltype $T" for T in (Float64, ComplexF64, Float32)
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        config = Zygote.ZygoteRuleConfig()
        test_rrule(
            config, left_orth, A;
            atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
        test_rrule(
            config, left_orth, A;
            fkwargs = (; alg = :qr), atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
        m >= n &&
            test_rrule(
            config, left_orth, A;
            fkwargs = (; alg = :polar), atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )

        ΔN = left_orth(A; alg = :qr)[1] * randn(rng, T, min(m, n), m - min(m, n))
        test_rrule(
            config, left_null, A;
            fkwargs = (; alg = :qr), output_tangent = ΔN, atol = atol, rtol = rtol,
            rrule_f = rrule_via_ad, check_inferred = false
        )

        test_rrule(
            config, right_orth, A;
            atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
        test_rrule(
            config, right_orth, A; fkwargs = (; alg = :lq),
            atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
        m <= n &&
            test_rrule(
            config, right_orth, A; fkwargs = (; alg = :polar),
            atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )

        ΔNᴴ = randn(rng, T, n - min(m, n), min(m, n)) * right_orth(A; alg = :lq)[2]
        test_rrule(
            config, right_null, A;
            fkwargs = (; alg = :lq), output_tangent = ΔNᴴ,
            atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
    end
end
