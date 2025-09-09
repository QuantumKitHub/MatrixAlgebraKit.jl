using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using ChainRulesCore
using Enzyme, EnzymeTestUtils
using MatrixAlgebraKit: diagview, TruncatedAlgorithm, PolarViaSVD
using LinearAlgebra: UpperTriangular, Diagonal, Hermitian, mul!, BlasFloat
using GenericLinearAlgebra, GenericSchur

# https://github.com/EnzymeAD/Enzyme.jl/issues/2888,
# test_reverse doesn't work with BigFloat

ETs = @static if VERSION < v"1.12.0"
    (ComplexF64, BigFloat)
else
    (ComplexF64,)
end
include("ad_utils.jl")
function test_pullbacks_match(rng, f!, f, A, args, Δargs, alg = nothing; ȳ = copy.(Δargs), return_act = Duplicated)
    ΔA = randn(rng, eltype(A), size(A)...)
    A_ΔA() = Duplicated(copy(A), copy(ΔA))
    function args_Δargs()
        if isnothing(args)
            return Const(args)
        elseif args isa Tuple && all(isnothing, args)
            return Const(args)
        else
            return Duplicated(copy.(args), copy.(Δargs))
        end
    end
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
        flush(stdout)
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

@timedtestset "QR AD Rules with eltype $T" for T in ETs
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        A = randn(rng, T, m, n)
        atol = rtol = m * n * precision(T)
        minmn = min(m, n)
        alg = MatrixAlgebraKit.default_qr_algorithm(A)
        @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
            @testset "qr_compact" begin
                ΔQR = (randn(rng, T, m, minmn), randn(rng, T, minmn, n))
                Q, R = qr_compact(A, alg)
                QR = MatrixAlgebraKit.initialize_output(qr_compact!, A, alg)
                fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                T <: BlasFloat && test_reverse(qr_compact, RT, (A, TA), (alg, Const); atol = atol, rtol = rtol, output_tangent = ΔQR, fdm = fdm)
                test_pullbacks_match(rng, qr_compact!, qr_compact, A, QR, ΔQR, alg)
            end
            @testset "qr_null" begin
                Q, R = qr_compact(A, alg)
                N = zeros(T, m, max(0, m - minmn))
                ΔN = Q * randn(rng, T, minmn, max(0, m - minmn))
                T <: BlasFloat && test_reverse(qr_null, RT, (A, TA), (alg, Const); atol = atol, rtol = rtol, output_tangent = ΔN)
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
                T <: BlasFloat && test_reverse(qr_full, RT, (A, TA), (alg, Const); atol = atol, rtol = rtol, output_tangent = (ΔQ, ΔR), fdm = fdm)
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
                T <: BlasFloat && test_reverse(qr_compact, RT, (Ard, TA), (alg, Const); atol = atol, rtol = rtol, output_tangent = (ΔQ, ΔR), fdm = fdm)
                test_pullbacks_match(rng, qr_compact!, qr_compact, Ard, (Q, R), (ΔQ, ΔR), alg)
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
        alg = MatrixAlgebraKit.default_lq_algorithm(A)
        @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
            @testset "lq_compact" begin
                ΔL = randn(rng, T, m, minmn)
                ΔQ = randn(rng, T, minmn, n)
                L, Q = lq_compact(A, alg)
                fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                T <: BlasFloat && test_reverse(lq_compact, RT, (A, TA), (alg, Const); atol = atol, rtol = rtol, output_tangent = (ΔL, ΔQ), fdm = fdm)
                test_pullbacks_match(rng, lq_compact!, lq_compact, A, (L, Q), (ΔL, ΔQ), alg)
            end
            @testset "lq_null" begin
                L, Q = lq_compact(A, alg)
                ΔNᴴ = randn(rng, T, max(0, n - minmn), minmn) * Q
                Nᴴ = randn(rng, T, max(0, n - minmn), minmn) * Q
                T <: BlasFloat && test_reverse(lq_null, RT, (A, TA), (alg, Const); atol = atol, rtol = rtol, output_tangent = ΔNᴴ)
                # runtime activity problems here with BigFloat
                T <: BlasFloat && test_pullbacks_match(rng, lq_null!, lq_null, A, Nᴴ, ΔNᴴ, alg)
            end
            @testset "lq_full" begin
                L, Q = lq_full(A, alg)
                Q1 = view(Q, 1:minmn, 1:n)
                ΔQ = randn(rng, T, n, n)
                ΔQ2 = view(ΔQ, (minmn + 1):n, 1:n)
                mul!(ΔQ2, ΔQ2 * Q1', Q1)
                ΔL = randn(rng, T, m, n)
                fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
                T <: BlasFloat && test_reverse(lq_full, RT, (A, TA), (alg, Const); atol = atol, rtol = rtol, output_tangent = (ΔL, ΔQ), fdm = fdm)
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
                T <: BlasFloat && test_reverse(lq_compact, RT, (Ard, TA), (alg, Const); atol = atol, rtol = rtol, output_tangent = (ΔL, ΔQ), fdm = fdm)
                test_pullbacks_match(rng, lq_compact!, lq_compact, Ard, (L, Q), (ΔL, ΔQ), alg)
            end
        end
    end
end

@timedtestset "EIG AD Rules with eltype $T" for T in ETs
    rng = StableRNG(12345)
    m = 19
    atol = rtol = m * m * precision(T)
    A = make_eig_matrix(rng, T, m)
    D, V = eig_full(A)
    Ddiag = diagview(D)
    ΔV = randn(rng, complex(T), m, m)
    ΔV = remove_eiggauge_dependence!(ΔV, D, V; degeneracy_atol = atol)
    ΔD = randn(rng, complex(T), m, m)
    ΔD2 = Diagonal(randn(rng, complex(T), m))
    fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    alg = MatrixAlgebraKit.default_eig_algorithm(A)
    @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        if T <: BlasFloat
            test_reverse(eig_full, RT, (A, TA); fkwargs = (alg = alg,), atol = atol, rtol = rtol, output_tangent = (copy(ΔD2), copy(ΔV)), fdm = fdm)
            test_pullbacks_match(rng, eig_full!, eig_full, A, (D, V), (ΔD2, ΔV), alg)
            test_reverse(eig_vals, RT, (A, TA); fkwargs = (alg = alg,), atol = atol, rtol = rtol, output_tangent = copy(ΔD2.diag), fdm = fdm)
            test_pullbacks_match(rng, eig_vals!, eig_vals, A, D.diag, ΔD2.diag, alg)
        else
            test_pullbacks_match(rng, eig_full!, eig_full, A, (nothing, nothing), (nothing, nothing), alg; ȳ = (ΔD2, ΔV))
            test_pullbacks_match(rng, eig_vals!, eig_vals, A, nothing, nothing, alg; ȳ = ΔD2.diag)
        end
    end
    @testset "eig_trunc reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        for r in 1:4:m
            truncalg = TruncatedAlgorithm(alg, truncrank(r; by = abs))
            ind = MatrixAlgebraKit.findtruncated(diagview(D), truncalg.trunc)
            Dtrunc = Diagonal(diagview(D)[ind])
            Vtrunc = V[:, ind]
            ΔDtrunc = Diagonal(diagview(ΔD2)[ind])
            ΔVtrunc = ΔV[:, ind]
            if T <: BlasFloat
                test_reverse(eig_trunc_no_error, RT, (A, TA), (truncalg, Const); atol, rtol, output_tangent = (ΔDtrunc, ΔVtrunc), fdm)
                test_pullbacks_match(rng, eig_trunc_no_error!, eig_trunc_no_error, A, (D, V), (ΔD2, ΔV), truncalg, ȳ = (ΔDtrunc, ΔVtrunc))
            else
                test_pullbacks_match(rng, eig_trunc_no_error!, eig_trunc_no_error, A, (nothing, nothing), (nothing, nothing), truncalg, ȳ = (ΔDtrunc, ΔVtrunc))
            end
        end
        truncalg = TruncatedAlgorithm(alg, truncrank(5; by = real))
        ind = MatrixAlgebraKit.findtruncated(Ddiag, truncalg.trunc)
        Dtrunc = Diagonal(Ddiag[ind])
        Vtrunc = V[:, ind]
        ΔDtrunc = Diagonal(diagview(ΔD2)[ind])
        ΔVtrunc = ΔV[:, ind]
        if T <: BlasFloat
            test_reverse(eig_trunc_no_error, RT, (A, TA), (truncalg, Const); atol, rtol, output_tangent = (ΔDtrunc, ΔVtrunc), fdm)
            test_pullbacks_match(rng, eig_trunc_no_error!, eig_trunc_no_error, A, (D, V), (ΔD2, ΔV), truncalg; ȳ = (ΔDtrunc, ΔVtrunc))
        else
            test_pullbacks_match(rng, eig_trunc_no_error!, eig_trunc_no_error, A, (nothing, nothing), (nothing, nothing), truncalg; ȳ = (ΔDtrunc, ΔVtrunc))
        end
    end
end


function copy_eigh_full(A, alg)
    A = (A + A') / 2
    return eigh_full(A, alg)
end

function copy_eigh_full!(A, DV::Tuple, alg::MatrixAlgebraKit.AbstractAlgorithm)
    A = (A + A') / 2
    return eigh_full!(A, DV, alg)
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

function copy_eigh_trunc_no_error(A, alg)
    A = (A + A') / 2
    return eigh_trunc_no_error(A, alg)
end

function copy_eigh_trunc_no_error!(A, DV, alg)
    A = (A + A') / 2
    return eigh_trunc_no_error!(A, DV, alg)
end

@timedtestset "EIGH AD Rules with eltype $T" for T in filter(T -> <:(T, BlasFloat), ETs)
    rng = StableRNG(12345)
    m = 19
    atol = rtol = m * m * precision(T)
    A = make_eigh_matrix(rng, T, m)
    D, V = eigh_full(A)
    D2 = Diagonal(D)
    ΔV = randn(rng, T, m, m)
    ΔV = remove_eighgauge_dependence!(ΔV, D, V; degeneracy_atol = atol)
    ΔD = randn(rng, real(T), m, m)
    ΔD2 = Diagonal(randn(rng, real(T), m))
    fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
    alg = MatrixAlgebraKit.default_eigh_algorithm(A)
    @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        T <: BlasFloat && test_reverse(copy_eigh_full, RT, (A, TA), (alg, Const); atol = atol, rtol = rtol, output_tangent = (copy(ΔD2), copy(ΔV)), fdm = fdm)
        T <: BlasFloat && test_reverse(copy_eigh_full!, RT, (copy(A), TA), ((D, V), TA), (alg, Const); atol = atol, rtol = rtol, output_tangent = (copy(ΔD2), copy(ΔV)), fdm = fdm)
        test_pullbacks_match(rng, copy_eigh_full!, copy_eigh_full, A, (D, V), (ΔD2, ΔV), alg)
        T <: BlasFloat && test_reverse(copy_eigh_vals, RT, (A, TA); fkwargs = (alg = alg,), atol = atol, rtol = rtol, output_tangent = copy(ΔD2.diag), fdm = fdm)
        test_pullbacks_match(rng, copy_eigh_vals!, copy_eigh_vals, A, D.diag, ΔD2.diag, alg)
    end
    @testset "eigh_trunc reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
        for r in 1:4:m
            Ddiag = diagview(D)
            truncalg = TruncatedAlgorithm(alg, truncrank(r; by = abs))
            ind = MatrixAlgebraKit.findtruncated(Ddiag, truncalg.trunc)
            Dtrunc = Diagonal(diagview(D)[ind])
            Vtrunc = V[:, ind]
            ΔDtrunc = Diagonal(diagview(ΔD2)[ind])
            ΔVtrunc = ΔV[:, ind]
            T <: BlasFloat && test_reverse(copy_eigh_trunc_no_error, RT, (A, TA), (truncalg, Const); atol, rtol, output_tangent = (ΔDtrunc, ΔVtrunc), fdm = fdm)
            test_pullbacks_match(rng, copy_eigh_trunc_no_error!, copy_eigh_trunc_no_error, A, (D, V), (ΔD2, ΔV), truncalg, ȳ = (ΔDtrunc, ΔVtrunc), return_act = RT)
        end
        Ddiag = diagview(D)
        truncalg = TruncatedAlgorithm(alg, trunctol(; atol = maximum(abs, Ddiag) / 2))
        ind = MatrixAlgebraKit.findtruncated(Ddiag, truncalg.trunc)
        Dtrunc = Diagonal(diagview(D)[ind])
        Vtrunc = V[:, ind]
        ΔDtrunc = Diagonal(diagview(ΔD2)[ind])
        ΔVtrunc = ΔV[:, ind]
        T <: BlasFloat && test_reverse(copy_eigh_trunc_no_error, RT, (A, TA), (truncalg, Const); atol = atol, rtol = rtol, output_tangent = (ΔDtrunc, ΔVtrunc), fdm = fdm)
        test_pullbacks_match(rng, copy_eigh_trunc_no_error!, copy_eigh_trunc_no_error, A, (D, V), (ΔD2, ΔV), truncalg, ȳ = (ΔDtrunc, ΔVtrunc), return_act = RT)
    end
end

@timedtestset "SVD AD Rules with eltype $T" for T in ETs
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        alg = MatrixAlgebraKit.default_svd_algorithm(A)
        minmn = min(m, n)
        fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
        @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
            @testset "svd_compact" begin
                U, S, Vᴴ = svd_compact(A)
                ΔU = randn(rng, T, m, minmn)
                ΔS = Diagonal(randn(rng, real(T), minmn))
                ΔVᴴ = randn(rng, T, minmn, n)
                ΔU, ΔVᴴ = remove_svdgauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ; degeneracy_atol = atol)
                if T <: BlasFloat
                    test_reverse(svd_compact, RT, (A, TA); fkwargs = (alg = alg,), atol, rtol, output_tangent = (ΔU, ΔS, ΔVᴴ), fdm = fdm)
                    test_pullbacks_match(rng, svd_compact!, svd_compact, A, (U, S, Vᴴ), (ΔU, ΔS, ΔVᴴ), alg)
                else
                    USVᴴ = MatrixAlgebraKit.initialize_output(svd_compact!, A, alg)
                    test_pullbacks_match(rng, svd_compact!, svd_compact, A, USVᴴ, (nothing, nothing, nothing), alg; ȳ = (ΔU, ΔS, ΔVᴴ))
                end
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
                if T <: BlasFloat
                    test_reverse(svd_full, RT, (A, TA); fkwargs = (alg = alg,), atol = atol, rtol = rtol, output_tangent = (ΔUfull, ΔSfull, ΔVᴴfull), fdm = fdm)
                    test_pullbacks_match(rng, svd_full!, svd_full, A, (U, S, Vᴴ), (ΔUfull, ΔSfull, ΔVᴴfull), alg)
                else
                    USVᴴ = MatrixAlgebraKit.initialize_output(svd_full!, A, alg)
                    test_pullbacks_match(rng, svd_full!, svd_full, A, USVᴴ, (nothing, nothing, nothing), alg; ȳ = (ΔUfull, ΔSfull, ΔVᴴfull))
                end
            end
            @testset "svd_vals" begin
                S = svd_vals(A)
                ΔS = randn(rng, real(T), minmn)
                if T <: BlasFloat
                    test_reverse(svd_vals, RT, (A, TA); atol = atol, rtol = rtol, fkwargs = (alg = alg,), output_tangent = ΔS, fdm = fdm)
                    test_pullbacks_match(rng, svd_vals!, svd_vals, A, S, ΔS, alg)
                else
                    S = MatrixAlgebraKit.initialize_output(svd_vals!, A, alg)
                    test_pullbacks_match(rng, svd_vals!, svd_vals, A, S, nothing, alg; ȳ = ΔS)
                end
            end
        end
        @testset "svd_trunc reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
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
                if T <: BlasFloat
                    test_reverse(svd_trunc_no_error, RT, (A, TA), (truncalg, Const); atol, rtol, output_tangent = (ΔUtrunc, ΔStrunc, ΔVᴴtrunc), fdm)
                    test_pullbacks_match(rng, svd_trunc_no_error!, svd_trunc_no_error, A, (U, S, Vᴴ), (ΔU, ΔS2, ΔVᴴ), truncalg, ȳ = (ΔUtrunc, ΔStrunc, ΔVᴴtrunc))
                else
                    test_pullbacks_match(rng, svd_trunc_no_error!, svd_trunc_no_error, A, (nothing, nothing, nothing), (nothing, nothing, nothing), truncalg, ȳ = (ΔUtrunc, ΔStrunc, ΔVᴴtrunc))
                end
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
            if T <: BlasFloat
                test_reverse(svd_trunc_no_error, RT, (A, TA), (truncalg, Const); atol, rtol, output_tangent = (ΔUtrunc, ΔStrunc, ΔVᴴtrunc), fdm)
                test_pullbacks_match(rng, svd_trunc_no_error!, svd_trunc_no_error, A, (U, S, Vᴴ), (ΔU, ΔS2, ΔVᴴ), truncalg, ȳ = (ΔUtrunc, ΔStrunc, ΔVᴴtrunc))
            else
                test_pullbacks_match(rng, svd_trunc_no_error!, svd_trunc_no_error, A, (nothing, nothing, nothing), (nothing, nothing, nothing), truncalg, ȳ = (ΔUtrunc, ΔStrunc, ΔVᴴtrunc))
            end
        end
    end
end

# GLA works with polar, but these tests
# segfault because of Sylvester + BigFloat
@timedtestset "Polar AD Rules with eltype $T" for T in filter(T -> <:(T, BlasFloat), ETs)
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        alg = MatrixAlgebraKit.default_polar_algorithm(A)
        @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
            if m >= n
                WP = left_polar(A; alg = alg)
                W, P = WP
                ΔWP = randn(rng, T, size(W)...), randn(rng, T, size(P)...)
                T <: BlasFloat && test_reverse(left_polar, RT, (A, TA), (alg, Const); atol = atol, rtol = rtol)
                test_pullbacks_match(rng, left_polar!, left_polar, A, WP, ΔWP, alg)
            elseif m <= n
                PWᴴ = right_polar(A; alg = alg)
                P, Wᴴ = PWᴴ
                ΔPWᴴ = randn(rng, T, size(P)...), randn(rng, T, size(Wᴴ)...)
                T <: BlasFloat && test_reverse(right_polar, RT, (A, TA), (alg, Const); atol = atol, rtol = rtol)
                test_pullbacks_match(rng, right_polar!, right_polar, A, PWᴴ, ΔPWᴴ, alg)
            end
        end
    end
end

# GLA not working with orthnull yet
@timedtestset "Orth and null with eltype $T" for T in filter(T -> <:(T, BlasFloat), ETs)
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
                    T <: BlasFloat && test_reverse(left_orth, RT, (A, TA); atol = atol, rtol = rtol, fkwargs = (alg = alg,), fdm = fdm)
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
                    T <: BlasFloat && test_reverse(right_orth, RT, (A, TA); atol = atol, rtol = rtol, fkwargs = (alg = alg,), fdm = fdm)
                    right_orth_alg!(A, CVᴴ) = right_orth!(A, CVᴴ; alg = alg)
                    right_orth_alg(A) = right_orth(A; alg = alg)
                    test_pullbacks_match(rng, right_orth_alg!, right_orth_alg, A, (C, Vᴴ), (ΔC, ΔVᴴ))
                end
            end
            @testset "left_null" begin
                ΔN = left_orth(A; alg = :qr)[1] * randn(rng, T, min(m, n), m - min(m, n))
                N = similar(ΔN)
                left_null_qr!(A, N) = left_null!(A, N; alg = :qr)
                left_null_qr(A) = left_null(A; alg = :qr)
                T <: BlasFloat && test_reverse(left_null_qr, RT, (A, TA); output_tangent = ΔN, atol = atol, rtol = rtol)
                test_pullbacks_match(rng, left_null_qr!, left_null_qr, A, N, ΔN)
            end
            @testset "right_null" begin
                ΔNᴴ = randn(rng, T, n - min(m, n), min(m, n)) * right_orth(A; alg = :lq)[2]
                Nᴴ = similar(ΔNᴴ)
                right_null_lq!(A, Nᴴ) = right_null!(A, Nᴴ; alg = :lq)
                right_null_lq(A) = right_null(A; alg = :lq)
                T <: BlasFloat && test_reverse(right_null_lq, RT, (A, TA); output_tangent = ΔNᴴ, atol = atol, rtol = rtol)
                test_pullbacks_match(rng, right_null_lq!, right_null_lq, A, Nᴴ, ΔNᴴ)
            end
        end
    end
end
