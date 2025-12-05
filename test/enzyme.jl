using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using ChainRulesCore
using Enzyme, EnzymeTestUtils
using MatrixAlgebraKit: diagview, TruncatedAlgorithm, PolarViaSVD
using LinearAlgebra: UpperTriangular, Diagonal, Hermitian, mul!

is_ci = get(ENV, "CI", "false") == "true"

ETs = is_ci ? (Float64, Float32) : (Float64, Float32, ComplexF32, ComplexF64) # Enzyme/#2631
include("ad_utils.jl")
function test_pullbacks_match(rng, f!, f, A, args, Δargs, alg = nothing; ȳ = copy.(Δargs), return_act = Duplicated)
    ΔA = randn(rng, eltype(A), size(A)...)
    A_ΔA() = Duplicated(copy(A), copy(ΔA))
    args_Δargs() = Duplicated(copy.(args), copy.(Δargs))
    copy_activities = isnothing(alg) ? (Const(f), A_ΔA()) : (Const(f), A_ΔA(), Const(alg))
    inplace_activities = isnothing(alg) ? (Const(f!), A_ΔA(), args_Δargs()) : (Const(f!), A_ΔA(), args_Δargs(), Const(alg))

    mode = EnzymeTestUtils.set_runtime_activity(ReverseSplitWithPrimal, false)
    c_act = Const(EnzymeTestUtils.call_with_kwargs)
    forward_copy, reverse_copy = autodiff_thunk(
        mode, typeof(c_act), return_act, typeof(Const(())), map(typeof, copy_activities)...
    )
    forward_inplace, reverse_inplace = autodiff_thunk(
        mode, typeof(c_act), return_act, typeof(Const(())), map(typeof, inplace_activities)...
    )
    copy_tape, copy_y_ad, copy_shadow_result = forward_copy(c_act, Const(()), copy_activities...)
    inplace_tape, inplace_y_ad, inplace_shadow_result = forward_inplace(c_act, Const(()), inplace_activities...)
    if !(copy_shadow_result === nothing)
        EnzymeTestUtils.map_fields_recursive(copyto!, copy_shadow_result, copy.(ȳ))
    end
    if !(inplace_shadow_result === nothing)
        EnzymeTestUtils.map_fields_recursive(copyto!, inplace_shadow_result, copy.(ȳ))
    end
    dx_copy_ad = only(reverse_copy(c_act, Const(()), copy_activities..., copy_tape))
    dx_inplace_ad = only(reverse_inplace(c_act, Const(()), inplace_activities..., inplace_tape))
    # check all returned derivatives between copy & inplace
    for (i, (copy_act_i, inplace_act_i)) in enumerate(zip(copy_activities[2:end], inplace_activities[2:end]))
        if copy_act_i isa Duplicated && inplace_act_i isa Duplicated
            msg_deriv = "shadow derivative for argument $(i - 1) should match between copy and inplace"
            EnzymeTestUtils.test_approx(copy_act_i.dval, inplace_act_i.dval, msg_deriv)
        end
    end
    return
end
#=
@timedtestset "QR AD Rules with eltype $T" for T in ETs
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        A = randn(rng, T, m, n)
        atol = rtol = m * n * precision(T)
        minmn = min(m, n)
        @testset for alg in (
                LAPACK_HouseholderQR(),
                LAPACK_HouseholderQR(; positive = true),
            )
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                @testset "qr_compact" begin
                    ΔQ = randn(rng, T, m, minmn)
                    ΔR = randn(rng, T, minmn, n)
                    Q, R = qr_compact(A, alg)
                    fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                    test_reverse(qr_compact, RT, (A, TA); atol = atol, rtol = rtol, fkwargs = (alg = alg,), output_tangent = (ΔQ, ΔR), fdm = fdm)
                    test_pullbacks_match(rng, qr_compact!, qr_compact, A, (Q, R), (ΔQ, ΔR), alg)
                end
                @testset "qr_null" begin
                    Q, R = qr_compact(A, alg)
                    N = zeros(T, m, max(0, m - minmn))
                    ΔN = Q * randn(rng, T, minmn, max(0, m - minmn))
                    test_reverse(qr_null, RT, (A, TA); atol = atol, rtol = rtol, fkwargs = (alg = alg,), output_tangent = ΔN)
                    test_pullbacks_match(rng, qr_null!, qr_null, A, N, ΔN, alg)
                end
                @testset "qr_full" begin
                    Q, R = qr_full(A, alg)
                    Q1 = view(Q, 1:m, 1:minmn)
                    ΔQ = randn(rng, T, m, m)
                    ΔQ2 = view(ΔQ, :, (minmn + 1):m)
                    mul!(ΔQ2, Q1, Q1' * ΔQ2)
                    ΔR = randn(rng, T, m, n)
                    fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                    test_reverse(qr_full, RT, (A, TA); atol = atol, rtol = rtol, fkwargs = (alg = alg,), output_tangent = (ΔQ, ΔR), fdm = fdm)
                    test_pullbacks_match(rng, qr_full!, qr_full, A, (Q, R), (ΔQ, ΔR), alg)
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
                    fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                    test_reverse(qr_compact, RT, (Ard, TA); atol = atol, rtol = rtol, fkwargs = (alg = alg,), output_tangent = (ΔQ, ΔR), fdm = fdm)
                    test_pullbacks_match(rng, qr_compact!, qr_compact, Ard, (Q, R), (ΔQ, ΔR), alg)
                end
            end
        end
    end
end

@timedtestset "LQ AD Rules with eltype $T" for T in ETs
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        minmn = min(m, n)
        A = randn(rng, T, m, n)
        @testset for alg in (
                LAPACK_HouseholderLQ(),
                LAPACK_HouseholderLQ(; positive = true),
            )
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                @testset "lq_compact" begin
                    ΔL = randn(rng, T, m, minmn)
                    ΔQ = randn(rng, T, minmn, n)
                    L, Q = lq_compact(A, alg)
                    fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                    test_reverse(lq_compact, RT, (A, TA); atol = atol, rtol = rtol, fkwargs = (alg = alg,), output_tangent = (ΔL, ΔQ), fdm = fdm)
                    test_pullbacks_match(rng, lq_compact!, lq_compact, A, (L, Q), (ΔL, ΔQ), alg)
                end
                @testset "lq_null" begin
                    L, Q = lq_compact(A, alg)
                    ΔNᴴ = randn(rng, T, max(0, n - minmn), minmn) * Q
                    Nᴴ = randn(rng, T, max(0, n - minmn), minmn) * Q
                    test_reverse(lq_null, RT, (A, TA); atol = atol, rtol = rtol, fkwargs = (alg = alg,), output_tangent = ΔNᴴ)
                    test_pullbacks_match(rng, lq_null!, lq_null, A, Nᴴ, ΔNᴴ, alg)
                end
                @testset "lq_full" begin
                    L, Q = lq_full(A, alg)
                    Q1 = view(Q, 1:minmn, 1:n)
                    ΔQ = randn(rng, T, n, n)
                    ΔQ2 = view(ΔQ, (minmn + 1):n, 1:n)
                    mul!(ΔQ2, ΔQ2 * Q1', Q1)
                    ΔL = randn(rng, T, m, n)
                    fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                    test_reverse(lq_full, RT, (A, TA); atol = atol, rtol = rtol, fkwargs = (alg = alg,), output_tangent = (ΔL, ΔQ), fdm = fdm)
                    test_pullbacks_match(rng, lq_full!, lq_full, A, (L, Q), (ΔL, ΔQ), alg)
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
                    fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                    test_reverse(lq_compact, RT, (Ard, TA); atol = atol, rtol = rtol, fkwargs = (alg = alg,), output_tangent = (ΔL, ΔQ), fdm = fdm)
                    test_pullbacks_match(rng, lq_compact!, lq_compact, Ard, (L, Q), (ΔL, ΔQ), alg)
                end
            end
        end
    end
end

@timedtestset "EIG AD Rules with eltype $T" for T in ETs
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
    @testset for alg in (LAPACK_Simple(), LAPACK_Expert())
        @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
            test_reverse(eig_full, RT, (A, TA); fkwargs = (alg = alg,), atol = atol, rtol = rtol, output_tangent = (copy(ΔD2), copy(ΔV)))
            test_pullbacks_match(rng, eig_full!, eig_full, A, (D, V), (ΔD2, ΔV), alg)
            test_reverse(eig_vals, RT, (A, TA); fkwargs = (alg = alg,), atol = atol, rtol = rtol, output_tangent = copy(ΔD2.diag))
            test_pullbacks_match(rng, eig_vals!, eig_vals, A, D.diag, ΔD2.diag, alg)
        end
        @testset "eig_trunc reverse: RT $RT, TA $TA" for RT in (MixedDuplicated,), TA in (Duplicated,)
            for r in 1:4:m
                truncalg = TruncatedAlgorithm(alg, truncrank(r; by = abs))
                ind = MatrixAlgebraKit.findtruncated(diagview(D), truncalg.trunc)
                Dtrunc = Diagonal(diagview(D)[ind])
                Vtrunc = V[:, ind]
                ΔDtrunc = Diagonal(diagview(ΔD2)[ind])
                ΔVtrunc = ΔV[:, ind]
                # broken due to Enzyme
                test_reverse(eig_trunc, RT, (A, TA); fkwargs = (alg = truncalg,), atol = atol, rtol = rtol, output_tangent = (ΔDtrunc, ΔVtrunc, zero(real(T))))
                # broken due to Enzyme
                test_pullbacks_match(rng, eig_trunc!, eig_trunc, A, (D, V), (ΔD2, ΔV), truncalg, ȳ=(ΔDtrunc, ΔVtrunc, zero(real(T))), return_act=RT)
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
            # broken due to Enzyme
            test_reverse(eig_trunc, RT, (A, TA); fkwargs = (alg = truncalg,), atol = atol, rtol = rtol, output_tangent = (ΔDtrunc, ΔVtrunc, zero(real(T))))
            # broken due to Enzyme
            test_pullbacks_match(rng, eig_trunc!, eig_trunc, A, (D, V), (ΔD2, ΔV), truncalg; ȳ=(ΔDtrunc, ΔVtrunc, zero(real(T))), return_act=RT)
            dA1 = MatrixAlgebraKit.eig_pullback!(zero(A), A, (D, V), (ΔDtrunc, ΔVtrunc), ind)
            dA2 = MatrixAlgebraKit.eig_trunc_pullback!(zero(A), A, (Dtrunc, Vtrunc), (ΔDtrunc, ΔVtrunc))
            @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
        end
    end
end

function copy_eigh_full(A; kwargs...)
    A = (A + A') / 2
    return eigh_full(A; kwargs...)
end

function copy_eigh_full(A, alg; kwargs...)
    A = (A + A') / 2
    return eigh_full(A, alg; kwargs...)
end

function copy_eigh_full!(A, DV; kwargs...)
    A = (A + A') / 2
    return eigh_full!(A, DV; kwargs...)
end

function copy_eigh_full!(A, DV, alg; kwargs...)
    A = (A + A') / 2
    return eigh_full!(A, DV, alg; kwargs...)
end

function copy_eigh_vals(A; kwargs...)
    A = (A + A') / 2
    return eigh_vals(A; kwargs...)
end

function copy_eigh_vals!(A, D; kwargs...)
    A = (A + A') / 2
    return eigh_vals!(A, D; kwargs...)
end

function copy_eigh_vals(A, alg; kwargs...)
    A = (A + A') / 2
    return eigh_vals(A, alg; kwargs...)
end

function copy_eigh_vals!(A, D, alg; kwargs...)
    A = (A + A') / 2
    return eigh_vals!(A, D, alg; kwargs...)
end

function copy_eigh_trunc(A; kwargs...)
    A = (A + A') / 2
    return eigh_trunc(A; kwargs...)
end

function copy_eigh_trunc!(A, DV; kwargs...)
    A = (A + A') / 2
    return eigh_trunc!(A, DV; kwargs...)
end

function copy_eigh_trunc(A, alg; kwargs...)
    A = (A + A') / 2
    return eigh_trunc(A; kwargs...)
end

function copy_eigh_trunc!(A, DV, alg; kwargs...)
    A = (A + A') / 2
    return eigh_trunc!(A, DV; kwargs...)
end

@timedtestset "EIGH AD Rules with eltype $T" for T in ETs
    rng = StableRNG(12345)
    m = 19
    atol = rtol = m * m * precision(T)
    A = randn(rng, T, m, m)
    A = A + A'
    D, V = eigh_full(A)
    D2 = Diagonal(D)
    ΔV = randn(rng, T, m, m)
    ΔV = remove_eighgauge_dependence!(ΔV, D, V; degeneracy_atol = atol)
    ΔD = randn(rng, real(T), m, m)
    ΔD2 = Diagonal(randn(rng, real(T), m))
    @testset for alg in (
            LAPACK_QRIteration(),
            LAPACK_DivideAndConquer(),
            LAPACK_Bisection(),
            LAPACK_MultipleRelativelyRobustRepresentations(),
        )
        @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
            test_reverse(copy_eigh_full, RT, (A, TA); fkwargs = (alg = alg,), atol = atol, rtol = rtol, output_tangent = (copy(ΔD2), copy(ΔV)))
            test_reverse(copy_eigh_full!, RT, (copy(A), TA), ((D, V), TA); fkwargs = (alg = alg,), atol = atol, rtol = rtol, output_tangent = (copy(ΔD2), copy(ΔV)))
            test_pullbacks_match(rng, copy_eigh_full!, copy_eigh_full, A, (D, V), (ΔD2, ΔV), alg)
            test_reverse(copy_eigh_vals, RT, (A, TA); fkwargs = (alg = alg,), atol = atol, rtol = rtol, output_tangent = copy(ΔD2.diag))
            test_pullbacks_match(rng, copy_eigh_vals!, copy_eigh_vals, A, D.diag, ΔD2.diag, alg)
        end
        @testset "eigh_trunc reverse: RT $RT, TA $TA" for RT in (MixedDuplicated,), TA in (Duplicated,)
            for r in 1:4:m
                Ddiag = diagview(D)
                truncalg = TruncatedAlgorithm(alg, truncrank(r; by = abs))
                ind = MatrixAlgebraKit.findtruncated(Ddiag, truncalg.trunc)
                Dtrunc = Diagonal(diagview(D)[ind])
                Vtrunc = V[:, ind]
                ΔDtrunc = Diagonal(diagview(ΔD2)[ind])
                ΔVtrunc = ΔV[:, ind]
                # broken due to Enzyme
                test_reverse(copy_eigh_trunc, RT, (A, TA); fkwargs = (alg = truncalg,), atol = atol, rtol = rtol, output_tangent = (ΔDtrunc, ΔVtrunc, zero(real(T))))
                test_pullbacks_match(rng, copy_eigh_trunc!, copy_eigh_trunc, A, (D, V), (ΔD2, ΔV), truncalg, ȳ=(ΔDtrunc, ΔVtrunc, zero(real(T))), return_act=RT)
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
            # broken due to Enzyme
            test_reverse(copy_eigh_trunc, RT, (A, TA); fkwargs = (alg = truncalg,), atol = atol, rtol = rtol, output_tangent = (ΔDtrunc, ΔVtrunc, zero(real(T))))
            test_pullbacks_match(rng, copy_eigh_trunc!, copy_eigh_trunc, A, (D, V), (ΔD2, ΔV), truncalg, ȳ=(ΔDtrunc, ΔVtrunc, zero(real(T))), return_act=RT)
            dA1 = MatrixAlgebraKit.eigh_pullback!(zero(A), A, (D, V), (ΔDtrunc, ΔVtrunc), ind)
            dA2 = MatrixAlgebraKit.eigh_trunc_pullback!(zero(A), A, (Dtrunc, Vtrunc), (ΔDtrunc, ΔVtrunc))
            @test isapprox(dA1, dA2; atol = atol, rtol = rtol)
        end
    end
end
=#
@timedtestset "SVD AD Rules with eltype $T" for T in ETs
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        minmn = min(m, n)
        @testset for alg in (
                LAPACK_QRIteration(),
                LAPACK_DivideAndConquer(),
            )
            #=@testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                @testset "svd_compact" begin
                    U, S, Vᴴ = svd_compact(A)
                    ΔU = randn(rng, T, m, minmn)
                    ΔS = Diagonal(randn(rng, real(T), minmn))
                    ΔVᴴ = randn(rng, T, minmn, n)
                    ΔU, ΔVᴴ = remove_svdgauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ; degeneracy_atol = atol)
                    fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                    test_reverse(svd_compact, RT, (A, TA); atol = atol, rtol = rtol, fkwargs = (alg = alg,), output_tangent = (ΔU, ΔS, ΔVᴴ), fdm = fdm)
                    test_pullbacks_match(rng, svd_compact!, svd_compact, A, (U, S, Vᴴ), (ΔU, ΔS, ΔVᴴ), alg)
                end
                @testset "svd_full" begin
                    U, S, Vᴴ = svd_compact(A)
                    ΔU = randn(rng, T, m, minmn)
                    ΔS = Diagonal(randn(rng, real(T), minmn))
                    ΔVᴴ = randn(rng, T, minmn, n)
                    ΔU, ΔVᴴ = remove_svdgauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ; degeneracy_atol = atol)
                    ΔUfull = zeros(T, m, m)
                    ΔSfull = zeros(real(T), m, n)
                    ΔVᴴfull = zeros(T, n, n)
                    U, S, Vᴴ = svd_full(A)
                    view(ΔUfull, :, 1:minmn) .= ΔU
                    view(ΔVᴴfull, 1:minmn, :) .= ΔVᴴ
                    diagview(ΔSfull)[1:minmn] .= diagview(ΔS)
                    fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                    test_reverse(svd_full, RT, (A, TA); atol = atol, rtol = rtol, fkwargs = (alg = alg,), output_tangent = (ΔUfull, ΔSfull, ΔVᴴfull), fdm = fdm)
                    test_pullbacks_match(rng, svd_full!, svd_full, A, (U, S, Vᴴ), (ΔUfull, ΔSfull, ΔVᴴfull), alg)
                end
                @testset "svd_vals" begin
                    S = svd_vals(A)
                    ΔS = randn(rng, real(T), minmn)
                    fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                    test_reverse(svd_vals, RT, (A, TA); atol = atol, rtol = rtol, fkwargs = (alg = alg,), output_tangent = ΔS, fdm = fdm)
                    test_pullbacks_match(rng, svd_vals!, svd_vals, A, S, ΔS, alg)
                end
            end=#
            @testset "svd_trunc reverse: RT $RT, TA $TA" for RT in (MixedDuplicated,), TA in (Duplicated,)
                for r in 1:4:minmn
                    U, S, Vᴴ = svd_compact(A)
                    ΔU = randn(rng, T, m, minmn)
                    ΔS = randn(rng, real(T), minmn, minmn)
                    ΔS2 = Diagonal(randn(rng, real(T), minmn))
                    ΔVᴴ = randn(rng, T, minmn, n)
                    ΔU, ΔVᴴ = remove_svdgauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ; degeneracy_atol = atol)
                    truncalg = TruncatedAlgorithm(alg, truncrank(r))
                    ind = MatrixAlgebraKit.findtruncated(diagview(S), truncalg.trunc)
                    Strunc = Diagonal(diagview(S)[ind])
                    Utrunc = U[:, ind]
                    Vᴴtrunc = Vᴴ[ind, :]
                    ΔStrunc = Diagonal(diagview(ΔS2)[ind])
                    ΔUtrunc = ΔU[:, ind]
                    ΔVᴴtrunc = ΔVᴴ[ind, :]
                    fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                    # broken due to Enzyme -- copying in gaugefix????
                    test_reverse(svd_trunc, RT, (A, TA); fkwargs = (alg = truncalg,), atol = atol, rtol = rtol, output_tangent = (ΔUtrunc, ΔStrunc, ΔVᴴtrunc, zero(real(T))), fdm = fdm)
                    test_pullbacks_match(rng, svd_trunc!, svd_trunc, A, (U, S, Vᴴ), (ΔU, ΔS2, ΔVᴴ), truncalg, ȳ=(ΔUtrunc, ΔStrunc, ΔVᴴtrunc, zero(real(T))), return_act=RT)
                end
                U, S, Vᴴ = svd_compact(A)
                ΔU = randn(rng, T, m, minmn)
                ΔS = randn(rng, real(T), minmn, minmn)
                ΔS2 = Diagonal(randn(rng, real(T), minmn))
                ΔVᴴ = randn(rng, T, minmn, n)
                ΔU, ΔVᴴ = remove_svdgauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ; degeneracy_atol = atol)
                truncalg = TruncatedAlgorithm(alg, trunctol(atol = S[1, 1] / 2))
                ind = MatrixAlgebraKit.findtruncated(diagview(S), truncalg.trunc)
                Strunc = Diagonal(diagview(S)[ind])
                Utrunc = U[:, ind]
                Vᴴtrunc = Vᴴ[ind, :]
                ΔStrunc = Diagonal(diagview(ΔS2)[ind])
                ΔUtrunc = ΔU[:, ind]
                ΔVᴴtrunc = ΔVᴴ[ind, :]
                fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                # broken due to Enzyme
                test_reverse(svd_trunc, RT, (A, TA); fkwargs = (alg = truncalg,), atol = atol, rtol = rtol, output_tangent = (ΔUtrunc, ΔStrunc, ΔVᴴtrunc, zero(real(T))), fdm = fdm)
                test_pullbacks_match(rng, svd_trunc!, svd_trunc, A, (U, S, Vᴴ), (ΔU, ΔS2, ΔVᴴ), truncalg, ȳ=(ΔUtrunc, ΔStrunc, ΔVᴴtrunc, zero(real(T))), return_act=RT)
            end
        end
    end
end
#=
@timedtestset "Polar AD Rules with eltype $T" for T in ETs
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        @testset for alg in PolarViaSVD.((LAPACK_QRIteration(), LAPACK_DivideAndConquer()))
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                if m >= n
                    WP = left_polar(A; alg = alg)
                    W, P = WP
                    ΔW = randn(rng, T, size(W)...)
                    ΔP = randn(rng, T, size(P)...)
                    test_reverse(left_polar, RT, (A, TA); atol = atol, rtol = rtol, fkwargs = (alg = alg,))
                    test_pullbacks_match(rng, left_polar!, left_polar, A, (W, P), (ΔW, ΔP), alg)
                elseif m <= n
                    PWᴴ = right_polar(A; alg = alg)
                    P, Wᴴ = PWᴴ
                    ΔWᴴ = randn(rng, T, size(Wᴴ)...)
                    ΔP = randn(rng, T, size(P)...)
                    test_reverse(right_polar, RT, (A, TA); atol = atol, rtol = rtol, fkwargs = (alg = alg,))
                    test_pullbacks_match(rng, right_polar!, right_polar, A, (P, Wᴴ), (ΔP, ΔWᴴ), alg)
                end
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
        @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
            @testset "left_orth" begin
                @testset for alg in (:polar, :qr)
                    n > m && alg == :polar && continue
                    VC = left_orth(A; alg = alg)
                    V, C = VC
                    ΔV = randn(rng, T, size(V)...)
                    ΔC = randn(rng, T, size(C)...)
                    fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                    test_reverse(left_orth, RT, (A, TA); atol = atol, rtol = rtol, fkwargs = (alg = alg,), fdm = fdm)
                    left_orth_alg!(A, VC) = left_orth!(A, VC; alg = alg)
                    left_orth_alg(A) = left_orth(A; alg = alg)
                    test_pullbacks_match(rng, left_orth_alg!, left_orth_alg, A, (V, C), (ΔV, ΔC))
                end
            end
            @testset "right_orth" begin
                @testset for alg in (:polar, :lq)
                    n < m && alg == :polar && continue
                    CVᴴ = right_orth(A; alg = alg)
                    C, Vᴴ = CVᴴ
                    ΔC = randn(rng, T, size(C)...)
                    ΔVᴴ = randn(rng, T, size(Vᴴ)...)
                    fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                    test_reverse(right_orth, RT, (A, TA); atol = atol, rtol = rtol, fkwargs = (alg = alg,), fdm = fdm)
                    right_orth_alg!(A, CVᴴ) = right_orth!(A, CVᴴ; alg = alg)
                    right_orth_alg(A) = right_orth(A; alg = alg)
                    test_pullbacks_match(rng, right_orth_alg!, right_orth_alg, A, (C, Vᴴ), (ΔC, ΔVᴴ))
                end
            end
            @testset "left_null" begin
                ΔN = left_orth(A; alg = :qr)[1] * randn(rng, T, min(m, n), m - min(m, n))
                N = similar(ΔN)
                test_reverse(left_null, RT, (A, TA); fkwargs = (; alg = :qr), output_tangent = ΔN, atol = atol, rtol = rtol)
                left_null_qr!(A, N) = left_null!(A, N; alg = :qr)
                left_null_qr(A) = left_null(A; alg = :qr)
                test_pullbacks_match(rng, left_null_qr!, left_null_qr, A, N, ΔN)
            end
            @testset "right_null" begin
                ΔNᴴ = randn(rng, T, n - min(m, n), min(m, n)) * right_orth(A; alg = :lq)[2]
                Nᴴ = similar(ΔNᴴ)
                test_reverse(right_null, RT, (A, TA); fkwargs = (; alg = :lq), output_tangent = ΔNᴴ, atol = atol, rtol = rtol)
                right_null_lq!(A, Nᴴ) = right_null!(A, Nᴴ; alg = :lq)
                right_null_lq(A) = right_null(A; alg = :lq)
                test_pullbacks_match(rng, right_null_lq!, right_null_lq, A, Nᴴ, ΔNᴴ)
            end
        end
    end
end
=#
