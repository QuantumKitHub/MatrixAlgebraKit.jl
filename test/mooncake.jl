using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using Mooncake, Mooncake.TestUtils, ChainRulesCore
using Mooncake: rrule!!
using MatrixAlgebraKit: diagview, TruncatedAlgorithm, PolarViaSVD
using LinearAlgebra: UpperTriangular, Diagonal, Hermitian, mul!

include("ad_utils.jl")

@timedtestset "QR AD Rules with eltype $T" for T in (Float64, Float32, ComplexF64)
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol  = rtol = m * n * precision(T)
        A     = randn(rng, T, m, n)
        minmn = min(m, n)
        @testset for alg in (LAPACK_HouseholderQR(),
                             LAPACK_HouseholderQR(; positive=true),
                            )
            @testset "qr_compact" begin 
                Mooncake.TestUtils.test_rule(rng, qr_compact, A, alg; mode=Mooncake.ReverseMode, is_primitive=false, atol=atol, rtol=rtol)
            end
            @testset "qr_null" begin
                Q, R = qr_compact(A, alg)
                ΔN   = Q * randn(rng, T, minmn, max(0, m - minmn))
                if T <: Real
                    dN   = ΔN
                else
                    dN   = [Mooncake.build_tangent(typeof(ΔN[i,j]), real(ΔN[i,j]), imag(ΔN[i,j])) for i in 1:size(ΔN, 1), j in 1:size(ΔN, 2)]
                end
                Mooncake.TestUtils.test_rule(rng, qr_null, A, alg; mode=Mooncake.ReverseMode, output_tangent = dN, is_primitive=false, atol=atol, rtol=rtol)
            end
            @testset "qr_full" begin
                Q, R = qr_full(A, alg)
                Q1   = view(Q, 1:m, 1:minmn)
                ΔQ   = randn(rng, T, m, m)
                ΔQ2  = view(ΔQ, :, (minmn + 1):m)
                mul!(ΔQ2, Q1, Q1' * ΔQ2)
                ΔR   = randn(rng, T, m, n)
                if T <: Real
                    dQ   = ΔQ
                    dR   = ΔR
                else
                    dQ   = [Mooncake.build_tangent(typeof(ΔQ[i,j]), real(ΔQ[i,j]), imag(ΔQ[i,j])) for i in 1:size(ΔQ, 1), j in 1:size(ΔQ, 2)]
                    dR   = [Mooncake.build_tangent(typeof(ΔR[i,j]), real(ΔR[i,j]), imag(ΔR[i,j])) for i in 1:size(ΔR, 1), j in 1:size(ΔR, 2)]
                end
                dQR = Mooncake.build_tangent(typeof((ΔQ,ΔR)), dQ, dR)
                Mooncake.TestUtils.test_rule(rng, qr_full, A, alg; mode=Mooncake.ReverseMode, output_tangent = dQR, is_primitive=false, atol=atol, rtol=rtol)
            end
            @testset "qr_compact - rank-deficient A" begin
                r    = minmn - 5
                Ard  = randn(rng, T, m, r) * randn(rng, T, r, n)
                Q, R = qr_compact(Ard, alg)
                ΔQ   = randn(rng, T, m, minmn)
                Q1   = view(Q, 1:m, 1:r)
                Q2   = view(Q, 1:m, (r + 1):minmn)
                ΔQ2  = view(ΔQ, 1:m, (r + 1):minmn)
                ΔQ2 .= 0
                ΔR   = randn(rng, T, minmn, n)
                view(ΔR, (r + 1):minmn, :) .= 0
                if T <: Real
                    dQ   = ΔQ
                    dR   = ΔR
                else
                    dQ   = [Mooncake.build_tangent(typeof(ΔQ[i,j]), real(ΔQ[i,j]), imag(ΔQ[i,j])) for i in 1:size(ΔQ, 1), j in 1:size(ΔQ, 2)]
                    dR   = [Mooncake.build_tangent(typeof(ΔR[i,j]), real(ΔR[i,j]), imag(ΔR[i,j])) for i in 1:size(ΔR, 1), j in 1:size(ΔR, 2)]
                end
                dQR = Mooncake.build_tangent(typeof((ΔQ,ΔR)), dQ, dR)
                Mooncake.TestUtils.test_rule(rng, qr_compact, copy(Ard), alg; mode=Mooncake.ReverseMode, output_tangent = dQR, is_primitive=false, atol=atol, rtol=rtol)
            end
        end
    end
end

@timedtestset "LQ AD Rules with eltype $T" for T in (Float64, Float32, ComplexF64)
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        # lq_compact
        atol  = rtol = m * n * precision(T)
        A     = randn(rng, T, m, n)
        minmn = min(m, n)
        @testset for alg in (LAPACK_HouseholderLQ(),
                             LAPACK_HouseholderLQ(; positive=true),
                            )
            @testset "lq_compact" begin
                ΔL  = randn(rng, T, m, minmn)
                ΔQ  = randn(rng, T, minmn, n)
                if T <: Real
                    dL   = ΔL
                    dQ   = ΔQ
                else
                    dL   = [Mooncake.build_tangent(typeof(ΔL[i,j]), real(ΔL[i,j]), imag(ΔL[i,j])) for i in 1:size(ΔL, 1), j in 1:size(ΔL, 2)]
                    dQ   = [Mooncake.build_tangent(typeof(ΔQ[i,j]), real(ΔQ[i,j]), imag(ΔQ[i,j])) for i in 1:size(ΔQ, 1), j in 1:size(ΔQ, 2)]
                end
                dLQ = Mooncake.build_tangent(typeof((ΔL,ΔQ)), dL, dQ)
                Mooncake.TestUtils.test_rule(rng, lq_compact, A, alg; mode=Mooncake.ReverseMode, is_primitive=false, atol=atol, rtol=rtol, output_tangent = dLQ)
                #=if n ≥ m
                    Mooncake.TestUtils.test_rule(rng, lq_compact, copy(A), alg; mode=Mooncake.ForwardMode, is_primitive=false, atol=atol, rtol=rtol)
                end=#
            end
            @testset "lq_null" begin
                L, Q = lq_compact(A, alg)
                ΔNᴴ = randn(rng, T, max(0, n - minmn), minmn) * Q
                if T <: Real
                    dNᴴ  = ΔNᴴ
                else
                    dNᴴ  = [Mooncake.build_tangent(typeof(ΔNᴴ[i,j]), real(ΔNᴴ[i,j]), imag(ΔNᴴ[i,j])) for i in 1:size(ΔNᴴ, 1), j in 1:size(ΔNᴴ, 2)]
                end
                Mooncake.TestUtils.test_rule(rng, lq_null, A, alg; mode=Mooncake.ReverseMode, output_tangent = dNᴴ, is_primitive=false, atol=atol, rtol=rtol)
            end
            @testset "lq_full" begin
                L, Q = lq_full(A, alg)
                Q1   = view(Q, 1:minmn, 1:n)
                ΔQ   = randn(rng, T, n, n)
                ΔQ2  = view(ΔQ, (minmn + 1):n, 1:n)
                mul!(ΔQ2, ΔQ2 * Q1', Q1)
                ΔL   = randn(rng, T, m, n)
                if T <: Real
                    dL   = ΔL
                    dQ   = ΔQ
                else
                    dL   = [Mooncake.build_tangent(typeof(ΔL[i,j]), real(ΔL[i,j]), imag(ΔL[i,j])) for i in 1:size(ΔL, 1), j in 1:size(ΔL, 2)]
                    dQ   = [Mooncake.build_tangent(typeof(ΔQ[i,j]), real(ΔQ[i,j]), imag(ΔQ[i,j])) for i in 1:size(ΔQ, 1), j in 1:size(ΔQ, 2)]
                end
                dLQ = Mooncake.build_tangent(typeof((ΔL,ΔQ)), dL, dQ)
                Mooncake.TestUtils.test_rule(rng, lq_full, A, alg; mode=Mooncake.ReverseMode, output_tangent = dLQ, is_primitive=false, atol=atol, rtol=rtol)
            end
            @testset "lq_compact - rank-deficient A" begin
                r = minmn - 5
                Ard = randn(rng, T, m, r) * randn(rng, T, r, n)
                L, Q = lq_compact(Ard, alg)
                ΔL = randn(rng, T, m, minmn)
                ΔQ = randn(rng, T, minmn, n)
                Q1 = view(Q, 1:r, 1:n)
                Q2 = view(Q, (r + 1):minmn, 1:n)
                ΔQ2 = view(ΔQ, (r + 1):minmn, 1:n)
                ΔQ2 .= 0
                view(ΔL, :, (r + 1):minmn) .= 0
                if T <: Real
                    dL   = ΔL
                    dQ   = ΔQ
                else
                    dL   = [Mooncake.build_tangent(typeof(ΔL[i,j]), real(ΔL[i,j]), imag(ΔL[i,j])) for i in 1:size(ΔL, 1), j in 1:size(ΔL, 2)]
                    dQ   = [Mooncake.build_tangent(typeof(ΔQ[i,j]), real(ΔQ[i,j]), imag(ΔQ[i,j])) for i in 1:size(ΔQ, 1), j in 1:size(ΔQ, 2)]
                end
                dLQ = Mooncake.build_tangent(typeof((ΔL,ΔQ)), dL, dQ)
                Mooncake.TestUtils.test_rule(rng, lq_compact, Ard, alg; mode=Mooncake.ReverseMode, output_tangent = dLQ, is_primitive=false, atol=atol, rtol=rtol)
            end
        end
    end
end

@timedtestset "EIG AD Rules with eltype $T" for T in (Float64, Float32, ComplexF64)
    rng  = StableRNG(12345)
    m    = 19
    atol = rtol = m * m * precision(T)
    A    = randn(rng, T, m, m)
    DV   = eig_full(A)
    D, V = DV
    ΔV   = randn(rng, complex(T), m, m)
    ΔV   = remove_eiggauge_dependence!(ΔV, D, V; degeneracy_atol=atol)
    ΔD   = randn(rng, complex(T), m, m)
    ΔD2  = Diagonal(randn(rng, complex(T), m))

    diag_tangent = [Mooncake.build_tangent(typeof(ΔD2.diag[ix]), real(ΔD2.diag[ix]), imag(ΔD2.diag[ix])) for ix in 1:m]
    dD  = Mooncake.build_tangent(typeof(ΔD2), diag_tangent)
    dV  = [Mooncake.build_tangent(typeof(ΔV[i,j]), real(ΔV[i,j]), imag(ΔV[i,j])) for i in 1:m, j in 1:m]
    dDV = Mooncake.build_tangent(typeof((ΔD2,ΔV)), dD, dV)
    # compute the dA corresponding to the above dD, dV
    @testset for alg in (LAPACK_Simple(), LAPACK_Expert())
        @testset "eig_full" begin
            Mooncake.TestUtils.test_rule(rng, eig_full, A, alg; mode=Mooncake.ReverseMode, output_tangent = dDV, is_primitive=false, atol=atol, rtol=rtol)
        end
        @testset "eig_vals" begin
            Mooncake.TestUtils.test_rule(rng, eig_vals, A, alg; atol=atol, rtol=rtol, is_primitive=false)
        end
    end
end

@timedtestset "EIGH AD Rules with eltype $T" for T in (Float64, Float32, ComplexF64)
    rng  = StableRNG(12345)
    m    = 19
    atol = rtol = m * m * precision(T)
    A    = randn(rng, T, m, m)
    A    = A + A'
    D, V = eigh_full(A)
    ΔV   = randn(rng, T, m, m)
    ΔV   = remove_eighgauge_dependence!(ΔV, D, V; degeneracy_atol=atol)
    ΔD   = randn(rng, real(T), m, m)
    ΔD2  = Diagonal(randn(rng, real(T), m))
    dD   = Mooncake.build_tangent(typeof(ΔD2), ΔD2.diag)
    if T <: Real
        dV = ΔV
    else
        dV  = [Mooncake.build_tangent(typeof(ΔV[i,j]), real(ΔV[i,j]), imag(ΔV[i,j])) for i in 1:m, j in 1:m]
    end
    dDV = Mooncake.build_tangent(typeof((ΔD2,ΔV)), dD, dV)
    @testset for alg in (LAPACK_QRIteration(),
                         LAPACK_DivideAndConquer(),
                         LAPACK_Bisection(),
                         LAPACK_MultipleRelativelyRobustRepresentations(),
                        )
        @testset "eigh_full" begin
            Mooncake.TestUtils.test_rule(rng, eigh_full, A, alg; mode=Mooncake.ReverseMode, output_tangent=dDV, is_primitive=false, atol=atol, rtol=rtol)
        end
        @testset "eigh_vals" begin
            Mooncake.TestUtils.test_rule(rng, eigh_vals, A, alg; is_primitive=false, atol=atol, rtol=rtol)
        end
    end
end

@timedtestset "SVD AD Rules with eltype $T" for T in (Float64, Float32, ComplexF64)
    rng = StableRNG(12345)
    m   = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol     = rtol = m * n * precision(T)
        A        = randn(rng, T, m, n)
        minmn    = min(m, n)
        @testset for alg in (LAPACK_QRIteration(), LAPACK_DivideAndConquer())
            U, S, Vᴴ = svd_full(A; alg=alg)
            #=@testset "svd_full" begin
                ΔU  = randn(rng, T, m, m)
                ΔS  = randn(rng, real(T), m, n)
                ΔVᴴ = randn(rng, T, n, n)
                remove_svdgauge_dependence!(view(ΔU, :, 1:minmn), view(ΔVᴴ, 1:minmn, :), view(U, :, 1:minmn), view(S, 1:minmn, 1:minmn), view(Vᴴ, 1:minmn, :); degeneracy_atol=atol)
                if T <: Real
                    dU   = ΔU
                    dS   = ΔS
                    dVᴴ  = ΔVᴴ
                else
                    dU   = [Mooncake.build_tangent(typeof(ΔU[i,j]), real(ΔU[i,j]), imag(ΔU[i,j])) for i in 1:size(ΔU, 1), j in 1:size(ΔU, 2)]
                    dS   = ΔS
                    dVᴴ  = [Mooncake.build_tangent(typeof(ΔVᴴ[i,j]), real(ΔVᴴ[i,j]), imag(ΔVᴴ[i,j])) for i in 1:size(ΔVᴴ, 1), j in 1:size(ΔVᴴ, 2)]
                end
                dUSVᴴ = Mooncake.build_tangent(typeof((ΔU,ΔS,ΔVᴴ)), dU, dS, dVᴴ)
                Mooncake.TestUtils.test_rule(rng, svd_full, A, alg; mode=Mooncake.ReverseMode, output_tangent=dUSVᴴ, is_primitive=false, atol=atol, rtol=rtol)
            end=# # TODO
            ΔU  = randn(rng, T, m, minmn)
            ΔS  = randn(rng, real(T), minmn, minmn)
            ΔS2 = Diagonal(randn(rng, real(T), minmn))
            ΔVᴴ = randn(rng, T, minmn, n)
            U, S, Vᴴ = svd_compact(A)
            ΔU, ΔVᴴ  = remove_svdgauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ; degeneracy_atol=atol)
            @testset "svd_compact" begin
                dS   = Mooncake.build_tangent(typeof(ΔS2), ΔS2.diag)
                if T <: Real
                    dU   = ΔU
                    dVᴴ  = ΔVᴴ
                else
                    dU   = [Mooncake.build_tangent(typeof(ΔU[i,j]), real(ΔU[i,j]), imag(ΔU[i,j])) for i in 1:size(ΔU, 1), j in 1:size(ΔU, 2)]
                    dVᴴ  = [Mooncake.build_tangent(typeof(ΔVᴴ[i,j]), real(ΔVᴴ[i,j]), imag(ΔVᴴ[i,j])) for i in 1:size(ΔVᴴ, 1), j in 1:size(ΔVᴴ, 2)]
                end
                dUSVᴴ = Mooncake.build_tangent(typeof((ΔU,ΔS2,ΔVᴴ)), dU, dS, dVᴴ)
                Mooncake.TestUtils.test_rule(rng, svd_compact, A, alg; mode=Mooncake.ReverseMode, output_tangent=dUSVᴴ, is_primitive=false, atol=atol, rtol=rtol)
            end
            @testset "svd_vals" begin
                Mooncake.TestUtils.test_rule(rng, svd_vals, A, alg; is_primitive=false, atol=atol, rtol=rtol)
            end
            #=@testset "svd_trunc" begin
                @testset for r in 1:4:minmn
                    truncalg = TruncatedAlgorithm(alg, truncrank(r))
                    ind      = MatrixAlgebraKit.findtruncated(diagview(S), truncalg.trunc)
                    Strunc   = Diagonal(diagview(S)[ind])
                    Utrunc   = U[:, ind]
                    Vᴴtrunc  = Vᴴ[ind, :]
                    ΔStrunc  = Diagonal(diagview(ΔS2)[ind])
                    ΔUtrunc  = ΔU[:, ind]
                    ΔVᴴtrunc = ΔVᴴ[ind, :]
                    dS       = Mooncake.build_tangent(typeof(ΔStrunc), ΔStrunc.diag)
                    if T <: Real
                        dU   = ΔUtrunc
                        dVᴴ  = ΔVᴴtrunc
                    else
                        dU   = [Mooncake.build_tangent(typeof(ΔUtrunc[i,j]), real(ΔUtrunc[i,j]), imag(ΔUtrunc[i,j])) for i in 1:size(ΔUtrunc, 1), j in 1:size(ΔUtrunc, 2)]
                        dVᴴ  = [Mooncake.build_tangent(typeof(ΔVᴴtrunc[i,j]), real(ΔVᴴtrunc[i,j]), imag(ΔVᴴtrunc[i,j])) for i in 1:size(ΔVᴴtrunc, 1), j in 1:size(ΔVᴴtrunc, 2)]
                    end
                    dUSVᴴ    = Mooncake.build_tangent(typeof((ΔU,ΔS2,ΔVᴴ)), dU, dS, dVᴴ)
                    Mooncake.TestUtils.test_rule(rng, svd_trunc, A, truncalg; mode=Mooncake.ReverseMode, output_tangent=dUSVᴴ, atol=atol, rtol=rtol, is_primitive=false)
                end
                truncalg = TruncatedAlgorithm(alg, trunctol(atol = S[1, 1] / 2))
                ind      = MatrixAlgebraKit.findtruncated(diagview(S), truncalg.trunc)
                Strunc   = Diagonal(diagview(S)[ind])
                Utrunc   = U[:, ind]
                Vᴴtrunc  = Vᴴ[ind, :]
                ΔStrunc  = Diagonal(diagview(ΔS2)[ind])
                ΔUtrunc  = ΔU[:, ind]
                ΔVᴴtrunc = ΔVᴴ[ind, :]
                dS       = Mooncake.build_tangent(typeof(ΔStrunc), ΔStrunc.diag)
                if T <: Real
                    dU   = ΔUtrunc
                    dVᴴ  = ΔVᴴtrunc
                else
                    dU   = [Mooncake.build_tangent(typeof(ΔUtrunc[i,j]), real(ΔUtrunc[i,j]), imag(ΔUtrunc[i,j])) for i in 1:size(ΔUtrunc, 1), j in 1:size(ΔUtrunc, 2)]
                    dVᴴ  = [Mooncake.build_tangent(typeof(ΔVᴴtrunc[i,j]), real(ΔVᴴtrunc[i,j]), imag(ΔVᴴtrunc[i,j])) for i in 1:size(ΔVᴴtrunc, 1), j in 1:size(ΔVᴴtrunc, 2)]
                end
                dUSVᴴ    = Mooncake.build_tangent(typeof((ΔU,ΔS2,ΔVᴴ)), dU, dS, dVᴴ)
                Mooncake.TestUtils.test_rule(rng, svd_trunc, A, truncalg; mode=Mooncake.ReverseMode, output_tangent=dUSVᴴ, atol=atol, rtol=rtol, is_primitive=false)
            end=# # TODO currently broken due to dimension mismatch
        end
    end
end

@timedtestset "Polar AD Rules with eltype $T" for T in (Float64, Float32, ComplexF64)
    rng = StableRNG(12345)
    m   = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        @testset for alg in PolarViaSVD.((LAPACK_QRIteration(), LAPACK_DivideAndConquer()))
            m >= n &&
                Mooncake.TestUtils.test_rule(rng, left_polar, A, alg; mode=Mooncake.ReverseMode, is_primitive=false, atol=atol, rtol=rtol)

            m <= n &&
                Mooncake.TestUtils.test_rule(rng, right_polar, A, alg; mode=Mooncake.ReverseMode, is_primitive=false, atol=atol, rtol=rtol)
        end
    end
end

@timedtestset "Orth and null with eltype $T" for T in (Float64, Float32, ComplexF64)
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol   = rtol = m * n * precision(T)
        A      = randn(rng, T, m, n)
        Mooncake.TestUtils.test_rule(rng, left_orth, A; mode=Mooncake.ReverseMode, atol=atol, rtol=rtol, is_primitive=false)
        Mooncake.TestUtils.test_rule(rng, right_orth, A; mode=Mooncake.ReverseMode, atol=atol, rtol=rtol, is_primitive=false)

        Mooncake.TestUtils.test_rule(rng, (X->left_orth(X; kind=:qr)), A; mode=Mooncake.ReverseMode, atol=atol, rtol=rtol, is_primitive=false)
        if m >= n
            Mooncake.TestUtils.test_rule(rng, (X->left_orth(X; kind=:polar)), A; mode=Mooncake.ReverseMode, atol=atol, rtol=rtol, is_primitive=false)
        end

        ΔN = left_orth(A; kind=:qr)[1] * randn(rng, T, min(m, n), m - min(m, n))
        if T <: Real
            dN   = ΔN
        else
            dN   = [Mooncake.build_tangent(typeof(ΔN[i,j]), real(ΔN[i,j]), imag(ΔN[i,j])) for i in 1:size(ΔN, 1), j in 1:size(ΔN, 2)]
        end
        Mooncake.TestUtils.test_rule(rng, (X->left_null(X; kind=:qr)), A; mode=Mooncake.ReverseMode, atol=atol, rtol=rtol, is_primitive=false, output_tangent = dN)

        Mooncake.TestUtils.test_rule(rng, (X->right_orth(X; kind=:lq)), A; mode=Mooncake.ReverseMode, atol=atol, rtol=rtol, is_primitive=false)

        if m <= n
            Mooncake.TestUtils.test_rule(rng, (X->right_orth(X; kind=:polar)), A; mode=Mooncake.ReverseMode, atol=atol, rtol=rtol, is_primitive=false)
        end

        ΔNᴴ = randn(rng, T, n - min(m, n), min(m, n)) * right_orth(A; kind=:lq)[2]
        if T <: Real
            dNᴴ   = ΔNᴴ
        else
            dNᴴ   = [Mooncake.build_tangent(typeof(ΔNᴴ[i,j]), real(ΔNᴴ[i,j]), imag(ΔNᴴ[i,j])) for i in 1:size(ΔNᴴ, 1), j in 1:size(ΔNᴴ, 2)]
        end
        Mooncake.TestUtils.test_rule(rng, (X->right_null(X; kind=:lq)), A; mode=Mooncake.ReverseMode, atol=atol, rtol=rtol, is_primitive=false, output_tangent = dNᴴ)
    end
end
