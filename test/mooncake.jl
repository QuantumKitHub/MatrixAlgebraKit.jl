using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using Mooncake, Mooncake.TestUtils, ChainRulesCore
using Mooncake: rrule!!
using MatrixAlgebraKit: diagview, TruncatedAlgorithm, PolarViaSVD
using LinearAlgebra: UpperTriangular, Diagonal, Hermitian, mul!

function Mooncake.increment!!(x::Tuple{Matrix{T}, Mooncake.Tangent{@NamedTuple{diag::Vector{T}}}, Matrix{T}}, y::Tuple{Matrix{T}, Mooncake.Tangent{@NamedTuple{diag::Vector{T}}}, Matrix{T}}) where {T<:Real}
    return (Mooncake.increment!!(x[1], y[1]), Mooncake.increment!!(x[2], y[2]), Mooncake.increment!!(x[3], y[3]))
end
function Mooncake.increment!!(x::Tuple{Mooncake.Tangent{@NamedTuple{diag::Vector{Mooncake.Tangent{@NamedTuple{re::T, im::T}}}}}, Matrix{Mooncake.Tangent{@NamedTuple{re::T, im::T}}}}, y::Tuple{Mooncake.Tangent{@NamedTuple{diag::Vector{Mooncake.Tangent{@NamedTuple{re::T, im::T}}}}}, Matrix{Mooncake.Tangent{@NamedTuple{re::T, im::T}}}}) where {T<:Real}
    return (Mooncake.increment!!(x[1], y[1]), Mooncake.increment!!(x[2], y[2]))
end
function Mooncake.increment!!(x::Tuple{Mooncake.Tangent{@NamedTuple{diag::Vector{Mooncake.Tangent{@NamedTuple{re::T, im::T}}}}}, Matrix{Mooncake.Tangent{@NamedTuple{re::T, im::T}}}}, y::Tuple{Mooncake.Tangent{@NamedTuple{diag::Vector{Mooncake.Tangent{@NamedTuple{re::T, im::T}}}}}, Matrix{Mooncake.Tangent{@NamedTuple{re::T, im::T}}}, Vector{T}}) where {T<:Real}
    return (Mooncake.increment!!(x[1], y[1]), Mooncake.increment!!(x[2], y[2]))
end

include("ad_utils.jl")

make_mooncake_tangent(ΔAelem::T) where {T<:Complex} = Mooncake.build_tangent(T, real(ΔAelem), imag(ΔAelem))                                                                                                          
make_mooncake_tangent(ΔA::Matrix{<:Real}) = ΔA                                                                                                                                                                       
make_mooncake_tangent(ΔA::Matrix{T}) where {T<:Complex} = map(make_mooncake_tangent, ΔA)                                                                                                                             
function make_mooncake_tangent(ΔD::Diagonal{T}) where {T<:Real}                                                                                                                                                      
    return Mooncake.build_tangent(typeof(ΔD), diagview(ΔD))                                                                                                                                                          
end                                                                                                                                                                                                                  
function make_mooncake_tangent(ΔD::Diagonal{T}) where {T<:Complex}                                                                                                                                                   
    diag_tangent = map(make_mooncake_tangent, diagview(ΔD))                                                                                                                                                          
    return Mooncake.build_tangent(typeof(ΔD), diag_tangent)                                                                                                                                                          
end 

ETs = (Float64,)# Float32,)# ComplexF64, ComplexF32)

@timedtestset "QR AD Rules with eltype $T" for T in ETs
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17,)# m, 23)
        atol  = rtol = m * n * precision(T)
        A     = randn(rng, T, m, n)
        minmn = min(m, n)
        @testset for alg in (LAPACK_HouseholderQR(),
                             #LAPACK_HouseholderQR(; positive=true),
                            )
            #=@testset "qr_compact" begin
                Mooncake.TestUtils.test_rule(rng, qr_compact, A, alg; is_primitive=false, atol=atol, rtol=rtol)
            end=#
            #=@testset "qr_null" begin
                Q, R = qr_compact(A, alg)
                ΔN   = Q * randn(rng, T, minmn, max(0, m - minmn))
                dN   = make_mooncake_tangent(ΔN)
                Mooncake.TestUtils.test_rule(rng, qr_null, A, alg; output_tangent = dN, is_primitive=false, atol=atol, rtol=rtol)
            end=#
            @testset "qr_full" begin
                Q, R = qr_full(A, alg)
                Q1   = view(Q, 1:m, 1:minmn)
                ΔQ   = randn(rng, T, m, m)
                ΔQ2  = view(ΔQ, :, (minmn + 1):m)
                mul!(ΔQ2, Q1, Q1' * ΔQ2)
                ΔR   = randn(rng, T, m, n)
                dQ   = make_mooncake_tangent(ΔQ)
                dR   = make_mooncake_tangent(ΔR)
                dQR  = Mooncake.build_tangent(typeof((ΔQ,ΔR)), dQ, dR)
                #Mooncake.TestUtils.test_rule(rng, ((A, alg)->qr_full(A, alg)[2]), A, alg; mode=Mooncake.ForwardMode, is_primitive=false, atol=atol, rtol=rtol)
                #Mooncake.TestUtils.test_rule(rng, ((A, alg)->qr_full(A, alg)[1][1:m, 1:minmn]), A, alg; mode=Mooncake.ForwardMode, is_primitive=false, atol=atol, rtol=rtol)
                Mooncake.TestUtils.test_rule(rng, ((A, alg)->qr_full(A, alg)[1][1:m, minmn+1:m]), A, alg; mode=Mooncake.ForwardMode, is_primitive=false, atol=atol, rtol=rtol)
            end
            #=@testset "qr_compact - rank-deficient A" begin
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
                dQ   = make_mooncake_tangent(ΔQ)
                dR   = make_mooncake_tangent(ΔR)
                dQR  = Mooncake.build_tangent(typeof((ΔQ,ΔR)), dQ, dR)
                Mooncake.TestUtils.test_rule(rng, qr_compact, copy(Ard), alg; mode=Mooncake.ForwardMode, is_primitive=false, atol=atol, rtol=rtol)
            end=#
        end
    end
end

#=
@timedtestset "LQ AD Rules with eltype $T" for T in ETs
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol  = rtol = m * n * precision(T)
        A     = randn(rng, T, m, n)
        minmn = min(m, n)
        @testset for alg in (LAPACK_HouseholderLQ(),
                             LAPACK_HouseholderLQ(; positive=true),
                            )
            @testset "lq_compact" begin
                ΔL  = randn(rng, T, m, minmn)
                ΔQ  = randn(rng, T, minmn, n)
                dL  = make_mooncake_tangent(ΔL)
                dQ  = make_mooncake_tangent(ΔQ)
                dLQ = Mooncake.build_tangent(typeof((ΔL,ΔQ)), dL, dQ)
                Mooncake.TestUtils.test_rule(rng, lq_compact, A, alg; is_primitive=false, atol=atol, rtol=rtol, output_tangent = dLQ)
            end
            #=@testset "lq_null" begin
                L, Q = lq_compact(A, alg)
                ΔNᴴ = randn(rng, T, max(0, n - minmn), minmn) * Q
                dNᴴ = make_mooncake_tangent(ΔNᴴ)
                Mooncake.TestUtils.test_rule(rng, lq_null, A, alg; output_tangent = dNᴴ, is_primitive=false, atol=atol, rtol=rtol)
            end=#
            @testset "lq_full" begin
                L, Q = lq_full(A, alg)
                Q1   = view(Q, 1:minmn, 1:n)
                ΔQ   = randn(rng, T, n, n)
                ΔQ2  = view(ΔQ, (minmn + 1):n, 1:n)
                mul!(ΔQ2, ΔQ2 * Q1', Q1)
                ΔL   = randn(rng, T, m, n)
                dL   = make_mooncake_tangent(ΔL)
                dQ   = make_mooncake_tangent(ΔQ)
                dLQ  = Mooncake.build_tangent(typeof((ΔL,ΔQ)), dL, dQ)
                Mooncake.TestUtils.test_rule(rng, lq_full, A, alg; output_tangent = dLQ, is_primitive=false, atol=atol, rtol=rtol)
            end
            #=@testset "lq_compact - rank-deficient A" begin
                r    = minmn - 5
                Ard  = randn(rng, T, m, r) * randn(rng, T, r, n)
                L, Q = lq_compact(Ard, alg)
                ΔL   = randn(rng, T, m, minmn)
                ΔQ   = randn(rng, T, minmn, n)
                Q1   = view(Q, 1:r, 1:n)
                Q2   = view(Q, (r + 1):minmn, 1:n)
                ΔQ2  = view(ΔQ, (r + 1):minmn, 1:n)
                ΔQ2 .= 0
                view(ΔL, :, (r + 1):minmn) .= 0
                dL   = make_mooncake_tangent(ΔL)
                dQ   = make_mooncake_tangent(ΔQ)
                dLQ  = Mooncake.build_tangent(typeof((ΔL,ΔQ)), dL, dQ)
                Mooncake.TestUtils.test_rule(rng, lq_compact, Ard, alg; output_tangent = dLQ, is_primitive=false, atol=atol, rtol=rtol)
            end=#
        end
    end
end
=#
#=
@timedtestset "EIG AD Rules with eltype $T" for T in ETs
    rng  = StableRNG(12345)
    m    = 19
    atol = rtol = m * m * precision(T)
    A    = randn(rng, T, m, m)
    DV   = eig_full(A)
    D, V = DV
    Ddiag = diagview(D)
    ΔV   = randn(rng, complex(T), m, m)
    ΔV   = remove_eiggauge_dependence!(ΔV, D, V; degeneracy_atol=atol)
    ΔD   = randn(rng, complex(T), m, m)
    ΔD2  = Diagonal(randn(rng, complex(T), m))

    dD  = make_mooncake_tangent(ΔD2)
    dV  = make_mooncake_tangent(ΔV)
    dDV = Mooncake.build_tangent(typeof((ΔD2,ΔV)), dD, dV)
    # compute the dA corresponding to the above dD, dV
    @testset for alg in (LAPACK_Simple(), LAPACK_Expert())
        @testset "eig_full" begin
            Ah = (A + A')/2
            Mooncake.TestUtils.test_rule(rng, (A, alg) -> eig_full(exp(A), alg)[1], Ah, alg; mode=Mooncake.ForwardMode, is_primitive=false, atol=atol, rtol=rtol)
            Mooncake.TestUtils.test_rule(rng, (A, alg) -> eig_full(exp(A), alg)[2], Ah, alg; mode=Mooncake.ForwardMode, is_primitive=false, atol=atol, rtol=rtol)
        end
        @testset "eig_vals" begin
            Mooncake.TestUtils.test_rule(rng, eig_vals, A, alg; atol=atol, rtol=rtol, is_primitive=false)
        end
        @testset "eig_trunc" begin
            for r in 1:4:m
                truncalg = TruncatedAlgorithm(alg, truncrank(r; by = abs))
                ind      = MatrixAlgebraKit.findtruncated(Ddiag, truncalg.trunc)
                Dtrunc   = Diagonal(diagview(D)[ind])
                Vtrunc   = V[:, ind]
                ΔDtrunc  = Diagonal(diagview(ΔD2)[ind])
                ΔVtrunc  = ΔV[:, ind]
                dDtrunc  = make_mooncake_tangent(ΔDtrunc)
                dVtrunc  = make_mooncake_tangent(ΔVtrunc)
                dDVtrunc = Mooncake.build_tangent(typeof((ΔDtrunc,ΔVtrunc,zero(real(T)))), dDtrunc, dVtrunc, zero(real(T)))
                Mooncake.TestUtils.test_rule(rng, eig_trunc, A, truncalg; mode=Mooncake.ReverseMode, output_tangent=dDVtrunc, atol=atol, rtol=rtol, is_primitive=false)
                dA1 = MatrixAlgebraKit.eig_pullback!(zero(A), A, (D, V), (ΔDtrunc, ΔVtrunc), ind)
                dA2 = MatrixAlgebraKit.eig_trunc_pullback!(zero(A), A, (Dtrunc, Vtrunc), (ΔDtrunc, ΔVtrunc))
                @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
            end
            truncalg = TruncatedAlgorithm(alg, truncrank(5; by = real))
            ind      = MatrixAlgebraKit.findtruncated(Ddiag, truncalg.trunc)
            Dtrunc   = Diagonal(diagview(D)[ind])
            Vtrunc   = V[:, ind]
            ΔDtrunc  = Diagonal(diagview(ΔD2)[ind])
            ΔVtrunc  = ΔV[:, ind]
            dDtrunc  = make_mooncake_tangent(ΔDtrunc)
            dVtrunc  = make_mooncake_tangent(ΔVtrunc)
            dDVtrunc = Mooncake.build_tangent(typeof((ΔDtrunc,ΔVtrunc,zero(real(T)))), dDtrunc, dVtrunc, zero(real(T)))
            Mooncake.TestUtils.test_rule(rng, eig_trunc, A, truncalg; mode=Mooncake.ReverseMode, output_tangent=dDVtrunc, atol=atol, rtol=rtol, is_primitive=false)
            dA1 = MatrixAlgebraKit.eig_pullback!(zero(A), A, (D, V), (ΔDtrunc, ΔVtrunc), ind)
            dA2 = MatrixAlgebraKit.eig_trunc_pullback!(zero(A), A, (Dtrunc, Vtrunc), (ΔDtrunc, ΔVtrunc))
            @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
        end
    end
end

function copy_eigh_full(A, alg; kwargs...)
    A = (A + A')/2
    eigh_full(A, alg; kwargs...)
end

function copy_eigh_vals(A, alg; kwargs...)
    A = (A + A')/2
    eigh_vals(A, alg; kwargs...)
end

function copy_eigh_trunc(A, alg; kwargs...)
    A = (A + A')/2
    eigh_trunc(A, alg; kwargs...)
end

@timedtestset "EIGH AD Rules with eltype $T" for T in ETs
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
    dD   = make_mooncake_tangent(ΔD2)
    dV   = make_mooncake_tangent(ΔV)
    dDV  = Mooncake.build_tangent(typeof((ΔD2,ΔV)), dD, dV)
    Ddiag = diagview(D)
    @testset for alg in (LAPACK_QRIteration(),
                         LAPACK_DivideAndConquer(),
                         LAPACK_Bisection(),
                         LAPACK_MultipleRelativelyRobustRepresentations(),
                        )
        @testset "eigh_full" begin
            Mooncake.TestUtils.test_rule(rng, copy_eigh_full, A, alg; mode=Mooncake.ReverseMode, output_tangent=dDV, is_primitive=false, atol=atol, rtol=rtol)
        end
        @testset "eigh_vals" begin
            Mooncake.TestUtils.test_rule(rng, copy_eigh_vals, A, alg; is_primitive=false, atol=atol, rtol=rtol)
        end
        @testset "eigh_trunc" begin
            for r in 1:4:m
                truncalg = TruncatedAlgorithm(alg, truncrank(r; by = abs))
                ind      = MatrixAlgebraKit.findtruncated(Ddiag, truncalg.trunc)
                Dtrunc   = Diagonal(diagview(D)[ind])
                Vtrunc   = V[:, ind]
                ΔDtrunc  = Diagonal(diagview(ΔD2)[ind])
                ΔVtrunc  = ΔV[:, ind]
                dDtrunc  = make_mooncake_tangent(ΔDtrunc)
                dVtrunc  = make_mooncake_tangent(ΔVtrunc)
                dDVtrunc = Mooncake.build_tangent(typeof((ΔDtrunc,ΔVtrunc,zero(real(T)))), dDtrunc, dVtrunc, zero(real(T)))
                Mooncake.TestUtils.test_rule(rng, copy_eigh_trunc, A, truncalg; mode=Mooncake.ReverseMode, output_tangent=dDVtrunc, atol=atol, rtol=rtol, is_primitive=false)
                dA1 = MatrixAlgebraKit.eigh_pullback!(zero(A), A, (D, V), (ΔDtrunc, ΔVtrunc), ind)
                dA2 = MatrixAlgebraKit.eigh_trunc_pullback!(zero(A), A, (Dtrunc, Vtrunc), (ΔDtrunc, ΔVtrunc))
                @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
            end
            truncalg = TruncatedAlgorithm(alg, trunctol(; atol = maximum(abs, Ddiag) / 2))
            ind      = MatrixAlgebraKit.findtruncated(Ddiag, truncalg.trunc)
            Dtrunc   = Diagonal(diagview(D)[ind])
            Vtrunc   = V[:, ind]
            ΔDtrunc  = Diagonal(diagview(ΔD2)[ind])
            ΔVtrunc  = ΔV[:, ind]
            dDtrunc  = make_mooncake_tangent(ΔDtrunc)
            dVtrunc  = make_mooncake_tangent(ΔVtrunc)
            dDVtrunc = Mooncake.build_tangent(typeof((ΔDtrunc,ΔVtrunc,zero(real(T)))), dDtrunc, dVtrunc, zero(real(T)))
            Mooncake.TestUtils.test_rule(rng, copy_eigh_trunc, A, truncalg; mode=Mooncake.ReverseMode, output_tangent=dDVtrunc, atol=atol, rtol=rtol, is_primitive=false)
            dA1 = MatrixAlgebraKit.eigh_pullback!(zero(A), A, (D, V), (ΔDtrunc, ΔVtrunc), ind)
            dA2 = MatrixAlgebraKit.eigh_trunc_pullback!(zero(A), A, (Dtrunc, Vtrunc), (ΔDtrunc, ΔVtrunc))
            @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
        end
    end
end

function dummy_svd_trunc(A, args...; kwargs...)
    U, S, Vᴴ, ϵ = svd_trunc(A, args...; kwargs...)
    return U, S, Vᴴ
end

@timedtestset "SVD AD Rules with eltype $T" for T in ETs
    rng = StableRNG(12345)
    m   = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol     = rtol = m * n * precision(T)
        A        = randn(rng, T, m, n)
        minmn    = min(m, n)
        @testset for alg in (LAPACK_QRIteration(),
                             #LAPACK_DivideAndConquer(),
                            )
            ΔU       = randn(rng, T, m, minmn)
            ΔS       = randn(rng, real(T), minmn, minmn)
            ΔS2      = Diagonal(randn(rng, real(T), minmn))
            ΔVᴴ      = randn(rng, T, minmn, n)
            U, S, Vᴴ = svd_compact(A)
            ΔU, ΔVᴴ  = remove_svdgauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ; degeneracy_atol=atol)
            ΔUref    = copy(ΔU)
            ΔSref    = copy(ΔS2)
            ΔVᴴref   = copy(ΔVᴴ)
            @testset "svd_compact" begin
                dS       = make_mooncake_tangent(ΔS2)
                dU       = make_mooncake_tangent(ΔU)
                dVᴴ      = make_mooncake_tangent(ΔVᴴ)
                dUSVᴴ    = Mooncake.build_tangent(typeof((ΔU,ΔS2,ΔVᴴ)), dU, dS, dVᴴ)
                Mooncake.TestUtils.test_rule(rng, svd_compact, A, alg; mode=Mooncake.ReverseMode, output_tangent=dUSVᴴ, is_primitive=false, atol=atol, rtol=rtol)
            end
            @testset "svd_full" begin
                ΔUfull   = zeros(T, m, m)
                ΔSfull   = zeros(real(T), m, n)
                ΔVᴴfull  = zeros(T, n, n)
                U, S, Vᴴ = svd_full(A)
                view(ΔUfull, :, 1:minmn)  .= ΔUref
                view(ΔVᴴfull, 1:minmn, :) .= ΔVᴴref
                diagview(ΔSfull)[1:minmn] .= diagview(ΔSref)
                dS       = make_mooncake_tangent(ΔSfull)
                dU       = make_mooncake_tangent(ΔUfull)
                dVᴴ      = make_mooncake_tangent(ΔVᴴfull)
                dUSVᴴ    = Mooncake.build_tangent(typeof((ΔUfull,ΔSfull,ΔVᴴfull)), dU, dS, dVᴴ)
                Mooncake.TestUtils.test_rule(rng, svd_full, A, alg; mode=Mooncake.ReverseMode, output_tangent=dUSVᴴ, is_primitive=false, atol=atol, rtol=rtol)
            end
            @testset "svd_vals" begin
                Mooncake.TestUtils.test_rule(rng, svd_vals, A, alg; is_primitive=false, atol=atol, rtol=rtol)
            end
            #=
            @testset "svd_trunc" begin
                @testset for r in 1:4:minmn
                    U, S, Vᴴ = svd_compact(A)
                    ΔU  = randn(rng, T, m, minmn)
                    ΔS  = randn(rng, real(T), minmn, minmn)
                    ΔS2 = Diagonal(randn(rng, real(T), minmn))
                    ΔVᴴ = randn(rng, T, minmn, n)
                    ΔU, ΔVᴴ  = remove_svdgauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ; degeneracy_atol=atol)
                    truncalg = TruncatedAlgorithm(alg, truncrank(r))
                    ind      = MatrixAlgebraKit.findtruncated(diagview(S), truncalg.trunc)
                    Strunc   = Diagonal(diagview(S)[ind])
                    Utrunc   = U[:, ind]
                    Vᴴtrunc  = Vᴴ[ind, :]
                    ΔStrunc  = Diagonal(diagview(ΔS2)[ind])
                    ΔUtrunc  = ΔU[:, ind]
                    ΔVᴴtrunc = ΔVᴴ[ind, :]
                    dStrunc  = make_mooncake_tangent(ΔStrunc)
                    dUtrunc  = make_mooncake_tangent(ΔUtrunc)
                    dVᴴtrunc = make_mooncake_tangent(ΔVᴴtrunc)
                    dUSVᴴerr = Mooncake.build_tangent(typeof((ΔU,ΔS2,ΔVᴴ)), dUtrunc, dStrunc, dVᴴtrunc)
                    Mooncake.TestUtils.test_rule(rng, dummy_svd_trunc, copy(A), truncalg; mode=Mooncake.ReverseMode, output_tangent=dUSVᴴerr, atol=atol, rtol=rtol, is_primitive=false)
                    dA1 = MatrixAlgebraKit.svd_pullback!(zero(A), A, (U, S, Vᴴ), (ΔUtrunc, ΔStrunc, ΔVᴴtrunc), ind)
                    dA2 = MatrixAlgebraKit.svd_trunc_pullback!(zero(A), A, (Utrunc, Strunc, Vᴴtrunc), (ΔUtrunc, ΔStrunc, ΔVᴴtrunc))
                    @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
                end
                U, S, Vᴴ = svd_compact(A)
                ΔU  = randn(rng, T, m, minmn)
                ΔS  = randn(rng, real(T), minmn, minmn)
                ΔS2 = Diagonal(randn(rng, real(T), minmn))
                ΔVᴴ = randn(rng, T, minmn, n)
                ΔU, ΔVᴴ  = remove_svdgauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ; degeneracy_atol=atol)
                truncalg = TruncatedAlgorithm(alg, trunctol(atol = S[1, 1] / 2))
                ind      = MatrixAlgebraKit.findtruncated(diagview(S), truncalg.trunc)
                Strunc   = Diagonal(diagview(S)[ind])
                Utrunc   = U[:, ind]
                Vᴴtrunc  = Vᴴ[ind, :]
                ΔStrunc  = Diagonal(diagview(ΔS2)[ind])
                ΔUtrunc  = ΔU[:, ind]
                ΔVᴴtrunc = ΔVᴴ[ind, :]
                dStrunc  = make_mooncake_tangent(ΔStrunc)
                dUtrunc  = make_mooncake_tangent(ΔUtrunc)
                dVᴴtrunc = make_mooncake_tangent(ΔVᴴtrunc)
                dUSVᴴerr = Mooncake.build_tangent(typeof((ΔU,ΔS2,ΔVᴴ)), dUtrunc, dStrunc, dVᴴtrunc)
                Mooncake.TestUtils.test_rule(rng, dummy_svd_trunc, copy(A), truncalg; mode=Mooncake.ReverseMode, output_tangent=dUSVᴴerr, atol=atol, rtol=rtol, is_primitive=false)
                dA1 = MatrixAlgebraKit.svd_pullback!(zero(A), A, (U, S, Vᴴ), (ΔUtrunc, ΔStrunc, ΔVᴴtrunc), ind)
                dA2 = MatrixAlgebraKit.svd_trunc_pullback!(zero(A), A, (Utrunc, Strunc, Vᴴtrunc), (ΔUtrunc, ΔStrunc, ΔVᴴtrunc))
                @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
            end=#
        end
    end
end

@timedtestset "Polar AD Rules with eltype $T" for T in ETs
    rng = StableRNG(12345)
    m   = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        @testset for alg in PolarViaSVD.((LAPACK_QRIteration(), LAPACK_DivideAndConquer()))
            m >= n &&
                Mooncake.TestUtils.test_rule(rng, left_polar, A, alg; is_primitive=false, atol=atol, rtol=rtol)

            m <= n &&
                Mooncake.TestUtils.test_rule(rng, right_polar, A, alg; is_primitive=false, atol=atol, rtol=rtol)
        end
    end
end

@timedtestset "Orth and null with eltype $T" for T in ETs
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol   = rtol = m * n * precision(T)
        A      = randn(rng, T, m, n)
        Mooncake.TestUtils.test_rule(rng, left_orth, A; atol=atol, rtol=rtol, is_primitive=false)
        Mooncake.TestUtils.test_rule(rng, right_orth, A; atol=atol, rtol=rtol, is_primitive=false)

        Mooncake.TestUtils.test_rule(rng, (X->left_orth(X; kind=:qr)), A; atol=atol, rtol=rtol, is_primitive=false)
        if m >= n
            Mooncake.TestUtils.test_rule(rng, (X->left_orth(X; kind=:polar)), A; atol=atol, rtol=rtol, is_primitive=false)
        end

        ΔN = left_orth(A; kind=:qr)[1] * randn(rng, T, min(m, n), m - min(m, n))
        dN = make_mooncake_tangent(ΔN)
        Mooncake.TestUtils.test_rule(rng, (X->left_null(X; kind=:qr)), A; atol=atol, rtol=rtol, is_primitive=false, output_tangent = dN)

        Mooncake.TestUtils.test_rule(rng, (X->right_orth(X; kind=:lq)), A; atol=atol, rtol=rtol, is_primitive=false)

        if m <= n
            Mooncake.TestUtils.test_rule(rng, (X->right_orth(X; kind=:polar)), A; atol=atol, rtol=rtol, is_primitive=false)
        end

        ΔNᴴ = randn(rng, T, n - min(m, n), min(m, n)) * right_orth(A; kind=:lq)[2]
        dNᴴ = make_mooncake_tangent(ΔNᴴ)
        Mooncake.TestUtils.test_rule(rng, (X->right_null(X; kind=:lq)), A; atol=atol, rtol=rtol, is_primitive=false, output_tangent = dNᴴ)
    end
end
=#
