using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using ChainRulesCore
using Enzyme, EnzymeTestUtils
using MatrixAlgebraKit: diagview, TruncatedAlgorithm, PolarViaSVD
using LinearAlgebra: UpperTriangular, Diagonal, Hermitian, mul!

is_ci = get(ENV, "CI", "false") == "true"

#ETs = is_ci ? (Float64, Float32) : (Float64, Float32, ComplexF32, ComplexF64) # Enzyme/#2631
ETs = (Float64, ComplexF64)
include("ad_utils.jl")
#=
@timedtestset "QR AD Rules with eltype $T" for T in ETs 
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        A     = randn(rng, T, m, n)
        atol  = rtol = m * n * precision(T)
        minmn = min(m, n)
        @testset for alg in (LAPACK_HouseholderQR(),
                             LAPACK_HouseholderQR(; positive=true),
                            )
            @testset "reverse: RT $RT, TA $TA" for RT  in (Duplicated,), TA in (Duplicated,)
                @testset "qr_compact" begin
                    ΔQ  = randn(rng, T, m, minmn)
                    ΔR  = randn(rng, T, minmn, n)
                    fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range=1e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                    test_reverse(qr_compact, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,), output_tangent=(ΔQ, ΔR), fdm=fdm)
                end
                @testset "qr_null" begin
                    Q, R = qr_compact(A, alg)
                    ΔN   = Q * randn(rng, T, minmn, max(0, m - minmn))
                    test_reverse(qr_null, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,), output_tangent=ΔN)
                end
                @testset "qr_full" begin
                    Q, R = qr_full(A, alg)
                    Q1   = view(Q, 1:m, 1:minmn)
                    ΔQ   = randn(rng, T, m, m)
                    ΔQ2  = view(ΔQ, :, (minmn + 1):m)
                    mul!(ΔQ2, Q1, Q1' * ΔQ2)
                    ΔR   = randn(rng, T, m, n)
                    fdm  = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range=1e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                    test_reverse(qr_full, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,), output_tangent=(ΔQ, ΔR), fdm=fdm)
                end
                @testset "qr_compact - rank-deficient A" begin
                    r = minmn - 5
                    Ard = randn(rng, T, m, r) * randn(rng, T, r, n)
                    Q, R = qr_compact(Ard, alg)
                    ΔQ = randn(rng, T, m, minmn)
                    Q1 = view(Q, 1:m, 1:r)
                    Q2 = view(Q, 1:m, (r + 1):minmn)
                    ΔQ2 = view(ΔQ, 1:m, (r + 1):minmn)
                    ΔQ2 .= 0
                    ΔR = randn(rng, T, minmn, n)
                    view(ΔR, (r + 1):minmn, :) .= 0
                    test_reverse(qr_compact, RT, (Ard, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,), output_tangent=(ΔQ, ΔR))
                end
            end
        end
    end
end

@timedtestset "LQ AD Rules with eltype $T" for T in ETs 
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol  = rtol = m * n * precision(T)
        minmn = min(m, n)
        A     = randn(rng, T, m, n)
        @testset for alg in (LAPACK_HouseholderLQ(),
                             LAPACK_HouseholderLQ(; positive=true),
                            )
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                @testset "lq_compact" begin
                    ΔL  = randn(rng, T, m, minmn)
                    ΔQ  = randn(rng, T, minmn, n)
                    fdm  = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range=1e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                    test_reverse(lq_compact, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,), output_tangent = (ΔL, ΔQ), fdm=fdm)
                end
                @testset "lq_null" begin
                    L, Q = lq_compact(A, alg)
                    ΔNᴴ  = randn(rng, T, max(0, n - minmn), minmn) * Q
                    test_reverse(lq_null, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,), output_tangent=ΔNᴴ)
                end
                @testset "lq_full" begin
                    L, Q = lq_full(A, alg)
                    Q1 = view(Q, 1:minmn, 1:n)
                    ΔQ = randn(rng, T, n, n)
                    ΔQ2 = view(ΔQ, (minmn + 1):n, 1:n)
                    mul!(ΔQ2, ΔQ2 * Q1', Q1)
                    ΔL = randn(rng, T, m, n)
                    fdm  = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range=1e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                    test_reverse(lq_full, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,), output_tangent = (ΔL, ΔQ), fdm=fdm)
                end
                @testset "lq_compact -- rank-deficient A" begin
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
                    fdm  = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range=1e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                    test_reverse(lq_compact, RT, (Ard, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,), output_tangent = (ΔL, ΔQ), fdm=fdm)
                end
            end
        end
    end
end
=#
function MatrixAlgebraKit.eig_trunc!(A, DV, ϵ::Vector{T}, alg::MatrixAlgebraKit.TruncatedAlgorithm) where {T}
    D, V = eig_full!(A, DV, alg.alg)
    DVtrunc, ind = MatrixAlgebraKit.truncate(eig_trunc!, (D, V), alg.trunc)
    ϵ[1] = MatrixAlgebraKit.truncation_error!(diagview(D), ind)
    return DVtrunc..., ϵ
end
function dummy_eig_trunc(A, ϵ::Vector{T}, alg::TruncatedAlgorithm) where {T}
    Ac = MatrixAlgebraKit.copy_input(MatrixAlgebraKit.eig_trunc, A)
    DV = MatrixAlgebraKit.initialize_output(eig_trunc!, A, alg)
    Dtrunc, Vtrunc, ϵ = MatrixAlgebraKit.eig_trunc!(Ac, DV, ϵ, alg)
    return Dtrunc, Vtrunc, ϵ
end

@timedtestset "EIG AD Rules with eltype $T" for T in ETs 
    rng  = StableRNG(12345)
    m    = 19
    atol = rtol = m * m * precision(T)
    A    = randn(rng, T, m, m)
    D, V = eig_full(A)
    Ddiag = diagview(D)
    ΔA   = randn(rng, T, m, m)
    ΔV   = randn(rng, complex(T), m, m)
    ΔV   = remove_eiggauge_dependence!(ΔV, D, V; degeneracy_atol=atol)
    ΔD   = randn(rng, complex(T), m, m)
    ΔD2  = Diagonal(randn(rng, complex(T), m))
    @testset for alg in (LAPACK_Simple(), LAPACK_Expert())
        @testset "forward: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
            # make A hermitian
            Ah = (A + A')/2
            test_forward((Ah; kwargs...)->eig_full(A; kwargs...)[1], RT, Duplicated(A, ΔA); fkwargs=(alg=alg,), atol=atol, rtol=rtol)
            test_forward((Ah; kwargs...)->eig_full(A; kwargs...)[2], RT, Duplicated(A, ΔA); fkwargs=(alg=alg,), atol=atol, rtol=rtol)
            #test_forward(eig_vals, RT, (copy(A), TA); fkwargs=(alg=alg,), atol=atol, rtol=rtol)
        end
        #=@testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
            test_reverse(eig_full, RT, (copy(A), TA); fkwargs=(alg=alg,), atol=atol, rtol=rtol, output_tangent=(copy(ΔD2), copy(ΔV)))
            test_reverse(eig_vals, RT, (copy(A), TA); fkwargs=(alg=alg,), atol=atol, rtol=rtol, output_tangent=copy(ΔD2.diag))
        end
        @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
            for r in 1:4:m
                truncalg = TruncatedAlgorithm(alg, truncrank(r; by = abs))
                ind = MatrixAlgebraKit.findtruncated(diagview(D), truncalg.trunc)
                Dtrunc = Diagonal(diagview(D)[ind])
                Vtrunc = V[:, ind]
                ΔDtrunc = Diagonal(diagview(ΔD2)[ind])
                ΔVtrunc = ΔV[:, ind]
                ϵ = [zero(real(T))] 
                test_reverse(dummy_eig_trunc, RT, (A, TA), (ϵ, TA), (truncalg, Const); atol=atol, rtol=rtol, output_tangent=(ΔDtrunc, ΔVtrunc, [zero(real(T))]))
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
            ϵ = [zero(real(T))] 
            test_reverse(dummy_eig_trunc, RT, (A, TA), (ϵ, TA), (truncalg, Const); atol=atol, rtol=rtol, output_tangent=(ΔDtrunc, ΔVtrunc, [zero(real(T))]))
            dA1 = MatrixAlgebraKit.eig_pullback!(zero(A), A, (D, V), (ΔDtrunc, ΔVtrunc), ind)
            dA2 = MatrixAlgebraKit.eig_trunc_pullback!(zero(A), A, (Dtrunc, Vtrunc), (ΔDtrunc, ΔVtrunc))
            @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
        end=#
    end
end

function copy_eigh_full(A; kwargs...)
    A = (A + A')/2
    eigh_full(A; kwargs...)
end

function copy_eigh_vals(A; kwargs...)
    A = (A + A')/2
    eigh_vals(A; kwargs...)
end

function MatrixAlgebraKit.eigh_trunc!(A, DV, ϵ::Vector{T}, alg::MatrixAlgebraKit.TruncatedAlgorithm) where {T}
    D, V = eigh_full!(A, DV, alg.alg)
    DVtrunc, ind = MatrixAlgebraKit.truncate(eigh_trunc!, (D, V), alg.trunc)
    ϵ[1] = MatrixAlgebraKit.truncation_error!(diagview(D), ind)
    return DVtrunc..., ϵ
end
function dummy_eigh_trunc(A, ϵ::Vector{T}, alg::TruncatedAlgorithm) where {T}
    A = (A + A')/2
    Ac = MatrixAlgebraKit.copy_input(MatrixAlgebraKit.eigh_trunc, A)
    DV = MatrixAlgebraKit.initialize_output(eigh_trunc!, A, alg)
    Dtrunc, Vtrunc, ϵ = MatrixAlgebraKit.eigh_trunc!(Ac, DV, ϵ, alg)
    return Dtrunc, Vtrunc, ϵ
end

@timedtestset "EIGH AD Rules with eltype $T" for T in ETs 
    rng  = StableRNG(12345)
    m    = 19
    atol = rtol = m * m * precision(T)
    A    = randn(rng, T, m, m)
    A    = A + A'
    D, V = eigh_full(A)
    D2   = Diagonal(D)
    ΔV   = randn(rng, T, m, m)
    ΔV   = remove_eighgauge_dependence!(ΔV, D, V; degeneracy_atol=atol)
    ΔD   = randn(rng, real(T), m, m)
    ΔD2  = Diagonal(randn(rng, real(T), m))
    @testset for alg in (LAPACK_QRIteration(),
                         LAPACK_DivideAndConquer(),
                         LAPACK_Bisection(),
                         LAPACK_MultipleRelativelyRobustRepresentations(),
                        )
        @testset "forward: RT $RT, TA $TA" for RT in (Const, Duplicated,), TA in (Const, Duplicated,)
            test_forward((A; kwargs...)->copy_eigh_full(A; kwargs...)[1], RT, (A, TA); fkwargs=(alg=alg,), atol=atol, rtol=rtol)
            test_forward((A; kwargs...)->copy_eigh_full(A; kwargs...)[2], RT, (A, TA); fkwargs=(alg=alg,), atol=atol, rtol=rtol)
            test_forward(copy_eigh_vals, RT, (copy(A), TA); fkwargs=(alg=alg,), atol=atol, rtol=rtol)
        end
        #=@testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
            test_reverse(copy_eigh_full, RT, (copy(A), TA); fkwargs=(alg=alg,), atol=atol, rtol=rtol, output_tangent=(copy(ΔD2), copy(ΔV)))
            test_reverse(copy_eigh_vals, RT, (copy(A), TA); fkwargs=(alg=alg,), atol=atol, rtol=rtol, output_tangent=copy(ΔD2.diag))
        end
        @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
            for r in 1:4:m
                Ddiag    = diagview(D)
                truncalg = TruncatedAlgorithm(alg, truncrank(r; by = abs))
                ind      = MatrixAlgebraKit.findtruncated(Ddiag, truncalg.trunc)
                Dtrunc   = Diagonal(diagview(D)[ind])
                Vtrunc   = V[:, ind]
                ΔDtrunc  = Diagonal(diagview(ΔD2)[ind])
                ΔVtrunc  = ΔV[:, ind]
                ϵ = [zero(real(T))] 
                test_reverse(dummy_eigh_trunc, RT, (A, TA), (ϵ, TA), (truncalg, Const); atol=atol, rtol=rtol, output_tangent=(ΔDtrunc, ΔVtrunc, [zero(real(T))]))
                dA1 = MatrixAlgebraKit.eigh_pullback!(zero(A), A, (D, V), (ΔDtrunc, ΔVtrunc), ind)
                dA2 = MatrixAlgebraKit.eigh_trunc_pullback!(zero(A), A, (Dtrunc, Vtrunc), (ΔDtrunc, ΔVtrunc))
                @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
            end
            Ddiag = diagview(D)
            truncalg = TruncatedAlgorithm(alg, trunctol(; atol = maximum(abs, Ddiag) / 2))
            ind = MatrixAlgebraKit.findtruncated(Ddiag, truncalg.trunc)
            Dtrunc = Diagonal(diagview(D)[ind])
            Vtrunc = V[:, ind]
            ΔDtrunc = Diagonal(diagview(ΔD2)[ind])
            ΔVtrunc = ΔV[:, ind]
            ϵ = [zero(real(T))] 
            test_reverse(dummy_eigh_trunc, RT, (A, TA), (ϵ, TA), (truncalg, Const); atol=atol, rtol=rtol, output_tangent=(ΔDtrunc, ΔVtrunc, [zero(real(T))]))
            dA1 = MatrixAlgebraKit.eigh_pullback!(zero(A), A, (D, V), (ΔDtrunc, ΔVtrunc), ind)
            dA2 = MatrixAlgebraKit.eigh_trunc_pullback!(zero(A), A, (Dtrunc, Vtrunc), (ΔDtrunc, ΔVtrunc))
            @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
        end=#
    end
end

function MatrixAlgebraKit.svd_trunc!(A, USVᴴ, ϵ::Vector{T}, alg::MatrixAlgebraKit.TruncatedAlgorithm) where {T}
    U, S, Vᴴ = svd_compact!(A, USVᴴ, alg.alg)
    USVᴴtrunc, ind = MatrixAlgebraKit.truncate(svd_trunc!, (U, S, Vᴴ), alg.trunc)
    ϵ[1] = MatrixAlgebraKit.truncation_error!(diagview(S), ind)
    return USVᴴtrunc..., ϵ
end
function dummy_svd_trunc(A, ϵ::Vector{T}, alg::TruncatedAlgorithm) where {T}
    Ac   = MatrixAlgebraKit.copy_input(MatrixAlgebraKit.svd_trunc, A)
    USVᴴ = MatrixAlgebraKit.initialize_output(svd_trunc!, A, alg)
    Utrunc, Strunc, Vᴴtrunc, ϵ = MatrixAlgebraKit.svd_trunc!(Ac, USVᴴ, ϵ, alg)
    return Utrunc, Strunc, Vᴴtrunc, ϵ
end

@timedtestset "SVD AD Rules with eltype $T" for T in ETs 
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        minmn = min(m, n)
        @testset for alg in (LAPACK_QRIteration(),
                             LAPACK_DivideAndConquer(),
                            )
            @testset "forward: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                @testset "svd_compact" begin
                    A = randn(rng, T, m, m)
                    Ah = (A + A')/2
                    fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range=1e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                    test_forward(svd_compact, RT, (Ah, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,), fdm=fdm)
                end
            end
            #=@testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                @testset "svd_compact" begin
                    U, S, Vᴴ = svd_compact(A)
                    ΔU  = randn(rng, T, m, minmn)
                    ΔS  = Diagonal(randn(rng, real(T), minmn))
                    ΔVᴴ = randn(rng, T, minmn, n)
                    ΔU, ΔVᴴ = remove_svdgauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ; degeneracy_atol=atol)
                    fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range=1e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                    test_reverse(svd_compact, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,), output_tangent=(ΔU, ΔS, ΔVᴴ), fdm=fdm)
                end
            end
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                @testset "svd_trunc" begin
                    for r in 1:4:minmn
                        U, S, Vᴴ = svd_compact(A)
                        ΔU       = randn(rng, T, m, minmn)
                        ΔS       = randn(rng, real(T), minmn, minmn)
                        ΔS2      = Diagonal(randn(rng, real(T), minmn))
                        ΔVᴴ      = randn(rng, T, minmn, n)
                        ΔU, ΔVᴴ  = remove_svdgauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ; degeneracy_atol=atol)
                        truncalg = TruncatedAlgorithm(alg, truncrank(r))
                        ind      = MatrixAlgebraKit.findtruncated(diagview(S), truncalg.trunc)
                        Strunc   = Diagonal(diagview(S)[ind])
                        Utrunc   = U[:, ind]
                        Vᴴtrunc  = Vᴴ[ind, :]
                        ΔStrunc  = Diagonal(diagview(ΔS2)[ind])
                        ΔUtrunc  = ΔU[:, ind]
                        ΔVᴴtrunc = ΔVᴴ[ind, :]
                        fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range=1e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                        # broken due to Enzyme
                        ϵ = [zero(real(T))] 
                        test_reverse(dummy_svd_trunc, RT, (A, TA), (ϵ, TA), (truncalg, Const); atol=atol, rtol=rtol, output_tangent=(ΔUtrunc, ΔStrunc, ΔVᴴtrunc, [zero(real(T))]), fdm=fdm)
                        dA1 = MatrixAlgebraKit.svd_pullback!(zero(A), copy(A), (U, S, Vᴴ), (ΔUtrunc, ΔStrunc, ΔVᴴtrunc), ind)
                        dA2 = MatrixAlgebraKit.svd_trunc_pullback!(zero(A), copy(A), (Utrunc, Strunc, Vᴴtrunc), (ΔUtrunc, ΔStrunc, ΔVᴴtrunc))
                        @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
                    end
                    U, S, Vᴴ = svd_compact(A)
                    ΔU       = randn(rng, T, m, minmn)
                    ΔS       = randn(rng, real(T), minmn, minmn)
                    ΔS2      = Diagonal(randn(rng, real(T), minmn))
                    ΔVᴴ      = randn(rng, T, minmn, n)
                    ΔU, ΔVᴴ  = remove_svdgauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ; degeneracy_atol=atol)
                    truncalg = TruncatedAlgorithm(alg, trunctol(atol = S[1, 1] / 2))
                    ind      = MatrixAlgebraKit.findtruncated(diagview(S), truncalg.trunc)
                    Strunc   = Diagonal(diagview(S)[ind])
                    Utrunc   = U[:, ind]
                    Vᴴtrunc  = Vᴴ[ind, :]
                    ΔStrunc  = Diagonal(diagview(ΔS2)[ind])
                    ΔUtrunc  = ΔU[:, ind]
                    ΔVᴴtrunc = ΔVᴴ[ind, :]
                    fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range=1e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                    # broken due to Enzyme
                    ϵ = [zero(real(T))] 
                    test_reverse(dummy_svd_trunc, RT, (A, TA), (ϵ, TA), (truncalg, Const); atol=atol, rtol=rtol, output_tangent=(ΔUtrunc, ΔStrunc, ΔVᴴtrunc, [zero(real(T))]), fdm=fdm)
                    dA1 = MatrixAlgebraKit.svd_pullback!(zero(A), copy(A), (copy(U), copy(S), copy(Vᴴ)), (copy(ΔUtrunc), copy(ΔStrunc), copy(ΔVᴴtrunc)), ind)
                    dA2 = MatrixAlgebraKit.svd_trunc_pullback!(zero(A), copy(A), (Utrunc, Strunc, Vᴴtrunc), (ΔUtrunc, ΔStrunc, ΔVᴴtrunc))
                    @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
                end
            end=#
        end
    end
end

@timedtestset "Polar AD Rules with eltype $T" for T in ETs 
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        @testset for alg in PolarViaSVD.((LAPACK_QRIteration(), LAPACK_DivideAndConquer()))
            @testset "reverse: RT $RT, TA $TA" for RT  in (Duplicated,), TA in (Duplicated,)
                m >= n &&
                    test_reverse(left_polar, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,))
                m <= n &&
                    test_reverse(right_polar, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,))
            end
        end
    end
end

@timedtestset "Orth and null with eltype $T" for T in ETs 
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        @testset "reverse: RT $RT, TA $TA" for RT  in (Duplicated,), TA in (Duplicated,)
            @testset "left_orth" begin
                @testset for kind in (:polar, :qr) 
                    n > m && kind == :polar && continue
                    test_reverse(left_orth, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(kind=kind,))
                end
            end
            @testset "right_orth" begin
                @testset for kind in (:polar, :lq) 
                    n < m && kind == :polar && continue
                    test_reverse(right_orth, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(kind=kind,))
                end
            end
            @testset "left_null" begin
                ΔN = left_orth(A; kind=:qr)[1] * randn(rng, T, min(m, n), m - min(m, n))
                test_reverse(left_null, RT, (A, TA); fkwargs=(; kind=:qr), output_tangent=ΔN, atol=atol, rtol=rtol)
            end
            @testset "right_null" begin
                ΔNᴴ = randn(rng, T, n - min(m, n), min(m, n)) * right_orth(A; kind=:lq)[2]
                test_reverse(right_null, RT, (A, TA); fkwargs=(; kind=:lq), output_tangent=ΔNᴴ, atol=atol, rtol=rtol)
            end
        end
    end
end

