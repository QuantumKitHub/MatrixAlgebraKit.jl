using MatrixAlgebraKit
using ChainRulesCore, ChainRulesTestUtils, Zygote
using MatrixAlgebraKit: diagview, TruncatedAlgorithm, PolarViaSVD
using LinearAlgebra: UpperTriangular, Diagonal, Hermitian, mul!

for f in
    (
        :qr_compact, :qr_full, :qr_null, :lq_compact, :lq_full, :lq_null,
        :eig_full, :eig_trunc, :eig_vals, :eigh_full, :eigh_trunc, :eigh_vals,
        :svd_compact, :svd_trunc, :svd_vals,
        :left_polar, :right_polar,
    )
    copy_f = Symbol(:cr_copy_, f)
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

function test_chainrules(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "Chainrules AD $summary_str" begin
        test_chainrules_qr(T, sz; kwargs...)
        test_chainrules_lq(T, sz; kwargs...)
        if length(sz) == 1 || sz[1] == sz[2]
            test_chainrules_eig(T, sz; kwargs...)
            test_chainrules_eigh(T, sz; kwargs...)
        end
        test_chainrules_svd(T, sz; kwargs...)
        test_chainrules_polar(T, sz; kwargs...)
        test_chainrules_orthnull(T, sz; kwargs...)
    end
end

function test_chainrules_qr(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "QR ChainRules AD rules $summary_str" begin
        A = instantiate_matrix(T, sz)
        config = Zygote.ZygoteRuleConfig()
        alg = MatrixAlgebraKit.default_qr_algorithm(A)
        @testset "qr_compact" begin
            QR, ΔQR = ad_qr_compact_setup(A)
            ΔQ, ΔR = ΔQR
            test_rrule(
                cr_copy_qr_compact, A, alg ⊢ NoTangent();
                output_tangent = ΔQR, atol = atol, rtol = rtol
            )
            test_rrule(
                config, qr_compact, A;
                fkwargs = (; positive = true), output_tangent = ΔQR,
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
        end
        @testset "qr_null" begin
            N, ΔN = ad_qr_null_setup(A)
            test_rrule(
                cr_copy_qr_null, A, alg ⊢ NoTangent();
                output_tangent = ΔN, atol = atol, rtol = rtol
            )
            test_rrule(
                config, qr_null, A;
                fkwargs = (; positive = true), output_tangent = ΔN,
                atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
            )
            m, n = size(A)
        end
        @testset "qr_full" begin
            QR, ΔQR = ad_qr_full_setup(A)
            test_rrule(
                cr_copy_qr_full, A, alg ⊢ NoTangent();
                output_tangent = ΔQR, atol = atol, rtol = rtol
            )
            test_rrule(
                config, qr_full, A;
                fkwargs = (; positive = true), output_tangent = ΔQR,
                atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
            )
            m, n = size(A)
        end
        @testset "qr_compact - rank-deficient A" begin
            m, n = size(A)
            r = min(m, n) - 5
            Ard = instantiate_matrix(T, (m, r)) * instantiate_matrix(T, (r, n))
            QR, ΔQR = ad_qr_rd_compact_setup(Ard)
            ΔQ, ΔR = ΔQR
            test_rrule(
                cr_copy_qr_compact, Ard, alg ⊢ NoTangent();
                output_tangent = ΔQR, atol = atol, rtol = rtol
            )
            test_rrule(
                config, qr_compact, Ard;
                fkwargs = (; positive = true), output_tangent = ΔQR,
                atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
            )
        end
    end
end

function test_chainrules_lq(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "LQ Chainrules AD rules $summary_str" begin
        A = instantiate_matrix(T, sz)
        m, n = size(A)
        config = Zygote.ZygoteRuleConfig()
        alg = MatrixAlgebraKit.default_lq_algorithm(A)
        @testset "lq_compact" begin
            LQ, ΔLQ = ad_lq_compact_setup(A)
            ΔL, ΔQ = ΔLQ
            test_rrule(
                cr_copy_lq_compact, A, alg ⊢ NoTangent();
                output_tangent = ΔLQ, atol = atol, rtol = rtol
            )
            test_rrule(
                config, lq_compact, A;
                fkwargs = (; positive = true), output_tangent = ΔLQ,
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
        end
        @testset "lq_null" begin
            Nᴴ, ΔNᴴ = ad_lq_null_setup(A)
            test_rrule(
                cr_copy_lq_null, A, alg ⊢ NoTangent();
                output_tangent = ΔNᴴ, atol = atol, rtol = rtol
            )
            test_rrule(
                config, lq_null, A;
                fkwargs = (; positive = true), output_tangent = ΔNᴴ,
                atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
            )
        end
        @testset "lq_full" begin
            LQ, ΔLQ = ad_lq_full_setup(A)
            test_rrule(
                cr_copy_lq_full, A, alg ⊢ NoTangent();
                output_tangent = ΔLQ, atol = atol, rtol = rtol
            )
            test_rrule(
                config, lq_full, A;
                fkwargs = (; positive = true), output_tangent = ΔLQ,
                atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
            )
        end
        @testset "lq_compact - rank-deficient A" begin
            m, n = size(A)
            r = min(m, n) - 5
            Ard = instantiate_matrix(T, (m, r)) * instantiate_matrix(T, (r, n))
            LQ, ΔLQ = ad_lq_rd_compact_setup(Ard)
            test_rrule(
                cr_copy_lq_compact, Ard, alg ⊢ NoTangent();
                output_tangent = ΔLQ, atol = atol, rtol = rtol
            )
            test_rrule(
                config, lq_compact, Ard;
                fkwargs = (; positive = true), output_tangent = ΔLQ,
                atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
            )
        end
    end
end

function test_chainrules_eig(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "EIG Chainrules AD rules $summary_str" begin
        A = instantiate_matrix(T, sz)
        m = size(A, 1)
        config = Zygote.ZygoteRuleConfig()
        alg = MatrixAlgebraKit.default_eig_algorithm(A)
        @testset "eig_full" begin
            DV, ΔDV, ΔD2V = ad_eig_full_setup(A)
            ΔD, ΔV = ΔDV
            test_rrule(
                cr_copy_eig_full, A, alg ⊢ NoTangent(); output_tangent = ΔDV, atol, rtol
            )
            test_rrule(
                cr_copy_eig_full, A, alg ⊢ NoTangent(); output_tangent = ΔD2V, atol, rtol
            )
            test_rrule(
                config, eig_full, A, alg ⊢ NoTangent();
                output_tangent = ΔDV, atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
            )
            test_rrule(
                config, eig_full, A, alg ⊢ NoTangent();
                output_tangent = ΔD2V, atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
            )
            test_rrule(
                config, first ∘ eig_full, A, alg ⊢ NoTangent();
                output_tangent = ΔD, atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
            )
            test_rrule(
                config, last ∘ eig_full, A, alg ⊢ NoTangent();
                output_tangent = ΔV, atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
            )
        end
        @testset "eig_vals" begin
            D, ΔD = ad_eig_vals_setup(A)
            test_rrule(
                cr_copy_eig_vals, A, alg ⊢ NoTangent(); output_tangent = ΔD, atol, rtol
            )
            test_rrule(
                config, eig_vals, A, alg ⊢ NoTangent();
                output_tangent = ΔD, atol, rtol, rrule_f = rrule_via_ad, check_inferred = false
            )
        end
        @testset "eig_trunc" begin
            for r in 1:4:m
                truncalg = TruncatedAlgorithm(alg, truncrank(r; by = abs))
                DV, DVtrunc, ΔDV, ΔDVtrunc = ad_eig_trunc_setup(A, truncalg)
                test_rrule(
                    cr_copy_eig_trunc, A, truncalg ⊢ NoTangent();
                    output_tangent = (ΔDVtrunc..., zero(real(T))),
                    atol = atol, rtol = rtol
                )
                ind = MatrixAlgebraKit.findtruncated(diagview(DV[1]), truncalg.trunc)
                dA1 = MatrixAlgebraKit.eig_pullback!(zero(A), A, DV, ΔDVtrunc, ind)
                dA2 = MatrixAlgebraKit.eig_trunc_pullback!(zero(A), A, DVtrunc, ΔDVtrunc)
                @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
            end
            truncalg = TruncatedAlgorithm(alg, truncrank(5; by = real))
            DV, DVtrunc, ΔDV, ΔDVtrunc = ad_eig_trunc_setup(A, truncalg)
            test_rrule(
                cr_copy_eig_trunc, A, truncalg ⊢ NoTangent();
                output_tangent = (ΔDVtrunc..., zero(real(T))),
                atol = atol, rtol = rtol
            )
            ind = MatrixAlgebraKit.findtruncated(diagview(DV[1]), truncalg.trunc)
            dA1 = MatrixAlgebraKit.eig_pullback!(zero(A), A, DV, ΔDVtrunc, ind)
            dA2 = MatrixAlgebraKit.eig_trunc_pullback!(zero(A), A, DVtrunc, ΔDVtrunc)
            @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
        end
    end
end

function test_chainrules_eigh(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "EIGH ChainRules AD rules $summary_str" begin
        A = instantiate_matrix(T, sz)
        A = A + A'
        m = size(A, 1)
        config = Zygote.ZygoteRuleConfig()
        alg = MatrixAlgebraKit.default_eigh_algorithm(A)
        # copy_eigh_xxxx includes a projector onto the Hermitian part of the matrix
        @testset "eigh_full" begin
            DV, ΔDV, ΔD2V = ad_eigh_full_setup(A)
            ΔD, ΔV = ΔDV
            test_rrule(
                cr_copy_eigh_full, A, alg ⊢ NoTangent(); output_tangent = ΔDV, atol, rtol
            )
            test_rrule(
                cr_copy_eigh_full, A, alg ⊢ NoTangent(); output_tangent = ΔD2V, atol, rtol
            )
            # eigh_full does not include a projector onto the Hermitian part of the matrix
            test_rrule(
                config, eigh_full ∘ Matrix ∘ Hermitian, A;
                output_tangent = ΔDV, atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
            )
            test_rrule(
                config, eigh_full ∘ Matrix ∘ Hermitian, A;
                output_tangent = ΔD2V, atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
            )
            test_rrule(
                config, first ∘ eigh_full ∘ Matrix ∘ Hermitian, A;
                output_tangent = ΔD, atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
            )
            test_rrule(
                config, last ∘ eigh_full ∘ Matrix ∘ Hermitian, A;
                output_tangent = ΔV, atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
            )
        end
        @testset "eigh_vals" begin
            D, ΔD = ad_eigh_vals_setup(A)
            test_rrule(
                cr_copy_eigh_vals, A, alg ⊢ NoTangent(); output_tangent = ΔD, atol, rtol
            )
            test_rrule(
                config, eigh_vals ∘ Matrix ∘ Hermitian, A;
                output_tangent = ΔD, atol, rtol, rrule_f = rrule_via_ad, check_inferred = false
            )
        end
        @testset "eigh_trunc" begin
            eigh_trunc2(A; kwargs...) = eigh_trunc(Matrix(Hermitian(A)); kwargs...)
            for r in 1:4:m
                truncalg = TruncatedAlgorithm(alg, truncrank(r; by = abs))
                DV, DVtrunc, ΔDV, ΔDVtrunc = ad_eigh_trunc_setup(A, truncalg)
                test_rrule(
                    cr_copy_eigh_trunc, A, truncalg ⊢ NoTangent();
                    output_tangent = (ΔDVtrunc..., zero(real(T))),
                    atol = atol, rtol = rtol
                )
                ind = MatrixAlgebraKit.findtruncated(diagview(DV[1]), truncalg.trunc)
                dA1 = MatrixAlgebraKit.eigh_pullback!(zero(A), A, DV, ΔDVtrunc, ind)
                dA2 = MatrixAlgebraKit.eigh_trunc_pullback!(zero(A), A, DVtrunc, ΔDVtrunc)
                @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
                trunc = truncrank(r; by = real)
                ind = MatrixAlgebraKit.findtruncated(diagview(DV[1]), trunc)
                truncalg = TruncatedAlgorithm(alg, trunc)
                DV, DVtrunc, ΔDV, ΔDVtrunc = ad_eigh_trunc_setup(A, truncalg)
                test_rrule(
                    config, eigh_trunc2, A;
                    fkwargs = (; trunc = trunc),
                    output_tangent = (ΔDVtrunc..., zero(real(T))),
                    atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
                )
            end
            D, ΔD = ad_eigh_vals_setup(A / 2)
            truncalg = TruncatedAlgorithm(alg, trunctol(; atol = maximum(abs, D) / 2))
            DV, DVtrunc, ΔDV, ΔDVtrunc = ad_eigh_trunc_setup(A, truncalg)
            ind = MatrixAlgebraKit.findtruncated(diagview(DV[1]), truncalg.trunc)
            test_rrule(
                cr_copy_eigh_trunc, A, truncalg ⊢ NoTangent();
                output_tangent = (ΔDVtrunc..., zero(real(T))),
                atol = atol, rtol = rtol
            )
            dA1 = MatrixAlgebraKit.eigh_pullback!(zero(A), A, DV, ΔDVtrunc, ind)
            dA2 = MatrixAlgebraKit.eigh_trunc_pullback!(zero(A), A, DVtrunc, ΔDVtrunc)
            @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
            trunc = trunctol(; rtol = 1 / 2)
            truncalg = TruncatedAlgorithm(alg, trunc)
            DV, DVtrunc, ΔDV, ΔDVtrunc = ad_eigh_trunc_setup(A, truncalg)
            ind = MatrixAlgebraKit.findtruncated(diagview(DV[1]), truncalg.trunc)
            test_rrule(
                config, eigh_trunc2, A;
                fkwargs = (; trunc = trunc),
                output_tangent = (ΔDVtrunc..., zero(real(T))),
                atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
            )
        end
    end
end

function test_chainrules_svd(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "SVD Chainrules AD rules $summary_str" begin
        A = instantiate_matrix(T, sz)
        minmn = min(size(A)...)
        config = Zygote.ZygoteRuleConfig()
        alg = MatrixAlgebraKit.default_svd_algorithm(A)
        @testset "svd_compact" begin
            USV, ΔUSVᴴ, ΔUS2Vᴴ = ad_svd_compact_setup(A)
            test_rrule(
                cr_copy_svd_compact, A, alg ⊢ NoTangent();
                output_tangent = ΔUSVᴴ, atol = atol, rtol = rtol
            )
            test_rrule(
                cr_copy_svd_compact, A, alg ⊢ NoTangent();
                output_tangent = ΔUS2Vᴴ, atol = atol, rtol = rtol
            )
            test_rrule(
                config, svd_compact, A, alg ⊢ NoTangent();
                output_tangent = ΔUSVᴴ, atol = atol, rtol = rtol,
                rrule_f = rrule_via_ad, check_inferred = false
            )
            test_rrule(
                config, svd_compact, A, alg ⊢ NoTangent();
                output_tangent = ΔUS2Vᴴ, atol = atol, rtol = rtol,
                rrule_f = rrule_via_ad, check_inferred = false
            )
        end
        @testset "svd_vals" begin
            S, ΔS = ad_svd_vals_setup(A)
            test_rrule(
                cr_copy_svd_vals, A, alg ⊢ NoTangent();
                output_tangent = ΔS, atol, rtol
            )
            test_rrule(
                config, svd_vals, A, alg ⊢ NoTangent();
                output_tangent = ΔS, atol, rtol, rrule_f = rrule_via_ad, check_inferred = false
            )
        end
        @testset "svd_trunc" begin
            @testset for r in 1:4:minmn
                truncalg = TruncatedAlgorithm(alg, truncrank(r))
                USVᴴ, ΔUSVᴴ, ΔUSVᴴtrunc = ad_svd_trunc_setup(A, truncalg)
                test_rrule(
                    cr_copy_svd_trunc, A, truncalg ⊢ NoTangent();
                    output_tangent = (ΔUSVᴴtrunc..., zero(real(T))),
                    atol = atol, rtol = rtol
                )
                U, S, Vᴴ = USVᴴ
                ind = MatrixAlgebraKit.findtruncated(diagview(S), truncalg.trunc)
                Strunc = Diagonal(diagview(S)[ind])
                Utrunc = U[:, ind]
                Vᴴtrunc = Vᴴ[ind, :]
                dA1 = MatrixAlgebraKit.svd_pullback!(zero(A), A, USVᴴ, ΔUSVᴴtrunc, ind)
                dA2 = MatrixAlgebraKit.svd_trunc_pullback!(zero(A), A, (Utrunc, Strunc, Vᴴtrunc), ΔUSVᴴtrunc)
                ind = MatrixAlgebraKit.findtruncated(diagview(S), truncalg.trunc)
                @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
                trunc = truncrank(r)
                ind = MatrixAlgebraKit.findtruncated(diagview(S), trunc)
                test_rrule(
                    config, svd_trunc, A;
                    fkwargs = (; trunc = trunc),
                    output_tangent = (ΔUSVᴴtrunc..., zero(real(T))),
                    atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
                )
            end
            S, ΔS = ad_svd_vals_setup(A)
            truncalg = TruncatedAlgorithm(alg, trunctol(atol = S[1, 1] / 2))
            USVᴴ, ΔUSVᴴ, ΔUSVᴴtrunc = ad_svd_trunc_setup(A, truncalg)
            test_rrule(
                cr_copy_svd_trunc, A, truncalg ⊢ NoTangent();
                output_tangent = (ΔUSVᴴtrunc..., zero(real(T))),
                atol = atol, rtol = rtol
            )
            U, S, Vᴴ = USVᴴ
            ind = MatrixAlgebraKit.findtruncated(diagview(S), truncalg.trunc)
            Strunc = Diagonal(diagview(S)[ind])
            Utrunc = U[:, ind]
            Vᴴtrunc = Vᴴ[ind, :]
            dA1 = MatrixAlgebraKit.svd_pullback!(zero(A), A, USVᴴ, ΔUSVᴴtrunc, ind)
            dA2 = MatrixAlgebraKit.svd_trunc_pullback!(zero(A), A, (Utrunc, Strunc, Vᴴtrunc), ΔUSVᴴtrunc)
            @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
            trunc = trunctol(; atol = S[1, 1] / 2)
            ind = MatrixAlgebraKit.findtruncated(diagview(S), trunc)
            test_rrule(
                config, svd_trunc, A;
                fkwargs = (; trunc = trunc),
                output_tangent = (ΔUSVᴴtrunc..., zero(real(T))),
                atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
            )
        end
    end
end

function test_chainrules_polar(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "Polar Chainrules AD rules $summary_str" begin
        A = instantiate_matrix(T, sz)
        m, n = size(A)
        config = Zygote.ZygoteRuleConfig()
        alg = MatrixAlgebraKit.default_polar_algorithm(A)
        @testset "left_polar" begin
            if m >= n
                test_rrule(cr_copy_left_polar, A, alg ⊢ NoTangent(); atol = atol, rtol = rtol)
                test_rrule(
                    config, left_polar, A, alg ⊢ NoTangent();
                    atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
                )
            end
        end
        @testset "right_polar" begin
            if m <= n
                test_rrule(cr_copy_right_polar, A, alg ⊢ NoTangent(); atol = atol, rtol = rtol)
                test_rrule(
                    config, right_polar, A, alg ⊢ NoTangent();
                    atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
                )
            end
        end
    end
end

function test_chainrules_orthnull(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "Orthnull Chainrules AD rules $summary_str" begin
        A = instantiate_matrix(T, sz)
        m, n = size(A)
        config = Zygote.ZygoteRuleConfig()
        N, ΔN = ad_left_null_setup(A)
        Nᴴ, ΔNᴴ = ad_right_null_setup(A)
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
        test_rrule(
            config, right_null, A;
            fkwargs = (; alg = :lq), output_tangent = ΔNᴴ,
            atol = atol, rtol = rtol, rrule_f = rrule_via_ad, check_inferred = false
        )
    end
end
