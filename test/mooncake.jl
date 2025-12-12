using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using Mooncake, Mooncake.TestUtils
using Mooncake: rrule!!
using MatrixAlgebraKit: diagview, TruncatedAlgorithm, PolarViaSVD, eigh_trunc
using LinearAlgebra: UpperTriangular, Diagonal, Hermitian, mul!

include("ad_utils.jl")

make_mooncake_tangent(ΔAelem::T) where {T <: Complex} = Mooncake.build_tangent(T, real(ΔAelem), imag(ΔAelem))
make_mooncake_tangent(ΔA::Matrix{<:Real}) = ΔA
make_mooncake_tangent(ΔA::Vector{<:Real}) = ΔA
make_mooncake_tangent(ΔA::Matrix{T}) where {T <: Complex} = map(make_mooncake_tangent, ΔA)
make_mooncake_tangent(ΔA::Vector{T}) where {T <: Complex} = map(make_mooncake_tangent, ΔA)
make_mooncake_tangent(ΔD::Diagonal{T}) where {T <: Real} = Mooncake.build_tangent(typeof(ΔD), diagview(ΔD))
make_mooncake_tangent(ΔD::Diagonal{T}) where {T <: Complex} = Mooncake.build_tangent(typeof(ΔD), map(make_mooncake_tangent, diagview(ΔD)))

make_mooncake_tangent(T::Tuple) = Mooncake.build_tangent(typeof(T), T...)

make_mooncake_fdata(x) = make_mooncake_tangent(x)
make_mooncake_fdata(x::Diagonal) = Mooncake.FData((diag = make_mooncake_tangent(x.diag),))

ETs = (Float32, ComplexF64)

# no `alg` argument
function _get_copying_derivative(f_c, rrule, A, ΔA, args, Δargs, ::Nothing, rdata)
    dA_copy = make_mooncake_tangent(copy(ΔA))
    A_copy = copy(A)
    dargs_copy = Δargs isa Tuple ? make_mooncake_fdata.(deepcopy(Δargs)) : make_mooncake_fdata(deepcopy(Δargs))
    copy_out, copy_pb!! = rrule(Mooncake.CoDual(f_c, Mooncake.NoFData()), Mooncake.CoDual(A_copy, dA_copy), Mooncake.CoDual(args, dargs_copy))
    copy_pb!!(rdata)
    return dA_copy
end

# `alg` argument
function _get_copying_derivative(f_c, rrule, A, ΔA, args, Δargs, alg, rdata)
    dA_copy = make_mooncake_tangent(copy(ΔA))
    A_copy = copy(A)
    dargs_copy = Δargs isa Tuple ? make_mooncake_fdata.(deepcopy(Δargs)) : make_mooncake_fdata(deepcopy(Δargs))
    copy_out, copy_pb!! = rrule(Mooncake.CoDual(f_c, Mooncake.NoFData()), Mooncake.CoDual(A_copy, dA_copy), Mooncake.CoDual(args, dargs_copy), Mooncake.CoDual(alg, Mooncake.NoFData()))
    copy_pb!!(rdata)
    return dA_copy
end

function _get_inplace_derivative(f!, A, ΔA, args, Δargs, ::Nothing, rdata)
    dA_inplace = make_mooncake_tangent(copy(ΔA))
    A_inplace = copy(A)
    dargs_inplace = Δargs isa Tuple ? make_mooncake_fdata.(deepcopy(Δargs)) : make_mooncake_fdata(deepcopy(Δargs))
    # not every f! has a handwritten rrule!!
    inplace_sig = Tuple{typeof(f!), typeof(A), typeof(args)}
    has_handwritten_rule = hasmethod(Mooncake.rrule!!, inplace_sig)
    if has_handwritten_rule
        inplace_out, inplace_pb!! = Mooncake.rrule!!(Mooncake.CoDual(f!, Mooncake.NoFData()), Mooncake.CoDual(A_inplace, dA_inplace), Mooncake.CoDual(args, dargs_inplace))
    else
        inplace_sig = Tuple{typeof(f!), typeof(A), typeof(args)}
        rvs_interp = Mooncake.get_interpreter(Mooncake.ReverseMode)
        inplace_rrule = Mooncake.build_rrule(rvs_interp, inplace_sig)
        inplace_out, inplace_pb!! = inplace_rrule(Mooncake.CoDual(f!, Mooncake.NoFData()), Mooncake.CoDual(A_inplace, dA_inplace), Mooncake.CoDual(args, dargs_inplace))
    end
    inplace_pb!!(rdata)
    return dA_inplace
end

function _get_inplace_derivative(f!, A, ΔA, args, Δargs, alg, rdata)
    dA_inplace = make_mooncake_tangent(copy(ΔA))
    A_inplace = copy(A)
    dargs_inplace = Δargs isa Tuple ? make_mooncake_fdata.(deepcopy(Δargs)) : make_mooncake_fdata(deepcopy(Δargs))
    # not every f! has a handwritten rrule!!
    inplace_sig = Tuple{typeof(f!), typeof(A), typeof(args), typeof(alg)}
    has_handwritten_rule = hasmethod(Mooncake.rrule!!, inplace_sig)
    if has_handwritten_rule
        inplace_out, inplace_pb!! = Mooncake.rrule!!(Mooncake.CoDual(f!, Mooncake.NoFData()), Mooncake.CoDual(A_inplace, dA_inplace), Mooncake.CoDual(args, dargs_inplace), Mooncake.CoDual(alg, Mooncake.NoFData()))
    else
        inplace_sig = Tuple{typeof(f!), typeof(A), typeof(args), typeof(alg)}
        rvs_interp = Mooncake.get_interpreter(Mooncake.ReverseMode)
        inplace_rrule = Mooncake.build_rrule(rvs_interp, inplace_sig)
        inplace_out, inplace_pb!! = inplace_rrule(Mooncake.CoDual(f!, Mooncake.NoFData()), Mooncake.CoDual(A_inplace, dA_inplace), Mooncake.CoDual(args, dargs_inplace), Mooncake.CoDual(alg, Mooncake.NoFData()))
    end
    inplace_pb!!(rdata)
    return dA_inplace
end

"""
    test_pullbacks_match(rng, f!, f, A, args, Δargs, alg = nothing; rdata = Mooncake.NoRData())

Compare the result of running the *in-place, mutating* function `f!`'s reverse rule
with the result of running its *non-mutating* partner function `f`'s reverse rule.
We must compare directly because many of the mutating functions modify `A` as a
scratch workspace, making testing `f!` against finite differences infeasible.

The arguments to this function are:
  - `f!` the mutating, in-place version of the function (accepts `args` for the function result)
  - `f` the non-mutating version of the function (does not accept `args` for the function result)
  - `A` the input matrix to factorize
  - `args` preallocated output for `f!` (e.g. `Q` and `R` matrices for `qr_compact!`)
  - `Δargs` precomputed derivatives of `args` for pullbacks of `f` and `f!`, to ensure they receive the same input
  - `alg` optional algorithm keyword argument
  - `rdata` Mooncake reverse data to supply to the pullback, in case `f` and `f!` return scalar results (as truncating functions do)
"""
function test_pullbacks_match(rng, f!, f, A, args, Δargs, alg = nothing; rdata = Mooncake.NoRData())
    f_c = isnothing(alg) ? (A, args) -> f!(MatrixAlgebraKit.copy_input(f, A), args) : (A, args, alg) -> f!(MatrixAlgebraKit.copy_input(f, A), args, alg)
    sig = isnothing(alg) ? Tuple{typeof(f_c), typeof(A), typeof(args)} : Tuple{typeof(f_c), typeof(A), typeof(args), typeof(alg)}
    rvs_interp = Mooncake.get_interpreter(Mooncake.ReverseMode)
    rrule = Mooncake.build_rrule(rvs_interp, sig)
    ΔA = randn(rng, eltype(A), size(A))

    dA_copy = _get_copying_derivative(f_c, rrule, A, ΔA, args, Δargs, alg, rdata)
    dA_inplace = _get_inplace_derivative(f!, A, ΔA, args, Δargs, alg, rdata)

    dA_inplace_ = Mooncake.arrayify(A, dA_inplace)[2]
    dA_copy_ = Mooncake.arrayify(A, dA_copy)[2]
    @test dA_inplace_ ≈ dA_copy_
    return
end

@timedtestset "QR AD Rules with eltype $T" for T in ETs
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        minmn = min(m, n)
        @testset for alg in (
                LAPACK_HouseholderQR(),
                LAPACK_HouseholderQR(; positive = true),
            )
            @testset "qr_compact" begin
                QR = qr_compact(A, alg)
                Q = randn(rng, T, m, minmn)
                R = randn(rng, T, minmn, n)
                Mooncake.TestUtils.test_rule(rng, qr_compact, A, alg; mode = Mooncake.ReverseMode, is_primitive = false, atol = atol, rtol = rtol)
                test_pullbacks_match(rng, qr_compact!, qr_compact, A, (Q, R), (randn(rng, T, m, minmn), randn(rng, T, minmn, n)), alg)
            end
            @testset "qr_null" begin
                Q, R = qr_compact(A, alg)
                ΔN = Q * randn(rng, T, minmn, max(0, m - minmn))
                N = qr_null(A, alg)
                dN = make_mooncake_tangent(copy(ΔN))
                Mooncake.TestUtils.test_rule(rng, qr_null, A, alg; mode = Mooncake.ReverseMode, output_tangent = dN, is_primitive = false, atol = atol, rtol = rtol)
                test_pullbacks_match(rng, qr_null!, qr_null, A, N, ΔN, alg)
            end
            @testset "qr_full" begin
                Q, R = qr_full(A, alg)
                Q1 = view(Q, 1:m, 1:minmn)
                ΔQ = randn(rng, T, m, m)
                ΔQ2 = view(ΔQ, :, (minmn + 1):m)
                mul!(ΔQ2, Q1, Q1' * ΔQ2)
                ΔR = randn(rng, T, m, n)
                dQ = make_mooncake_tangent(copy(ΔQ))
                dR = make_mooncake_tangent(copy(ΔR))
                dQR = Mooncake.build_tangent(typeof((ΔQ, ΔR)), dQ, dR)
                Mooncake.TestUtils.test_rule(rng, qr_full, A, alg; mode = Mooncake.ReverseMode, output_tangent = dQR, is_primitive = false, atol = atol, rtol = rtol)
                test_pullbacks_match(rng, qr_full!, qr_full, A, (Q, R), (ΔQ, ΔR), alg)
            end
            @testset "qr_compact - rank-deficient A" begin
                r = minmn - 5
                Ard = randn(rng, T, m, r) * randn(rng, T, r, n)
                Q, R = qr_compact(Ard, alg)
                QR = (Q, R)
                ΔQ = randn(rng, T, m, minmn)
                Q1 = view(Q, 1:m, 1:r)
                Q2 = view(Q, 1:m, (r + 1):minmn)
                ΔQ2 = view(ΔQ, 1:m, (r + 1):minmn)
                ΔQ2 .= 0
                ΔR = randn(rng, T, minmn, n)
                view(ΔR, (r + 1):minmn, :) .= 0
                dQ = make_mooncake_tangent(copy(ΔQ))
                dR = make_mooncake_tangent(copy(ΔR))
                dQR = Mooncake.build_tangent(typeof((ΔQ, ΔR)), dQ, dR)
                Mooncake.TestUtils.test_rule(rng, qr_compact, Ard, alg; mode = Mooncake.ReverseMode, output_tangent = dQR, is_primitive = false, atol = atol, rtol = rtol)
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
        A = randn(rng, T, m, n)
        minmn = min(m, n)
        @testset for alg in (
                LAPACK_HouseholderLQ(),
                LAPACK_HouseholderLQ(; positive = true),
            )
            @testset "lq_compact" begin
                L, Q = lq_compact(A, alg)
                Mooncake.TestUtils.test_rule(rng, lq_compact, A, alg; mode = Mooncake.ReverseMode, is_primitive = false, atol = atol, rtol = rtol)
                test_pullbacks_match(rng, lq_compact!, lq_compact, A, (L, Q), (randn(rng, T, m, minmn), randn(rng, T, minmn, n)), alg)
            end
            @testset "lq_null" begin
                L, Q = lq_compact(A, alg)
                ΔNᴴ = randn(rng, T, max(0, n - minmn), minmn) * Q
                Nᴴ = randn(rng, T, max(0, n - minmn), n)
                dNᴴ = make_mooncake_tangent(ΔNᴴ)
                Mooncake.TestUtils.test_rule(rng, lq_null, A, alg; mode = Mooncake.ReverseMode, output_tangent = dNᴴ, is_primitive = false, atol = atol, rtol = rtol)
                test_pullbacks_match(rng, lq_null!, lq_null, A, Nᴴ, ΔNᴴ, alg)
            end
            @testset "lq_full" begin
                L, Q = lq_full(A, alg)
                Q1 = view(Q, 1:minmn, 1:n)
                ΔQ = randn(rng, T, n, n)
                ΔQ2 = view(ΔQ, (minmn + 1):n, 1:n)
                mul!(ΔQ2, ΔQ2 * Q1', Q1)
                ΔL = randn(rng, T, m, n)
                dL = make_mooncake_tangent(ΔL)
                dQ = make_mooncake_tangent(ΔQ)
                dLQ = Mooncake.build_tangent(typeof((ΔL, ΔQ)), dL, dQ)
                Mooncake.TestUtils.test_rule(rng, lq_full, A, alg; mode = Mooncake.ReverseMode, output_tangent = dLQ, is_primitive = false, atol = atol, rtol = rtol)
                test_pullbacks_match(rng, lq_full!, lq_full, A, (L, Q), (ΔL, ΔQ), alg)
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
                dL = make_mooncake_tangent(ΔL)
                dQ = make_mooncake_tangent(ΔQ)
                dLQ = Mooncake.build_tangent(typeof((ΔL, ΔQ)), dL, dQ)
                Mooncake.TestUtils.test_rule(rng, lq_compact, Ard, alg; mode = Mooncake.ReverseMode, output_tangent = dLQ, is_primitive = false, atol = atol, rtol = rtol)
                test_pullbacks_match(rng, lq_compact!, lq_compact, Ard, (L, Q), (ΔL, ΔQ), alg)
            end
        end
    end
end

@timedtestset "EIG AD Rules with eltype $T" for T in ETs
    rng = StableRNG(12345)
    m = 19
    atol = rtol = m * m * precision(T)
    A = randn(rng, T, m, m)
    DV = eig_full(A)
    D, V = DV
    Ddiag = diagview(D)
    ΔV = randn(rng, complex(T), m, m)
    ΔV = remove_eiggauge_dependence!(ΔV, D, V; degeneracy_atol = atol)
    ΔD = randn(rng, complex(T), m, m)
    ΔD2 = Diagonal(randn(rng, complex(T), m))

    dD = make_mooncake_tangent(ΔD2)
    dV = make_mooncake_tangent(ΔV)
    dDV = Mooncake.build_tangent(typeof((ΔD2, ΔV)), dD, dV)
    # compute the dA corresponding to the above dD, dV
    @testset for alg in (
            LAPACK_Simple(),
            #LAPACK_Expert(), # expensive on CI
        )
        @testset "eig_full" begin
            Mooncake.TestUtils.test_rule(rng, eig_full, A, alg; mode = Mooncake.ReverseMode, output_tangent = dDV, is_primitive = false, atol = atol, rtol = rtol)
            test_pullbacks_match(rng, eig_full!, eig_full, A, (D, V), (ΔD2, ΔV), alg)
        end
        @testset "eig_vals" begin
            Mooncake.TestUtils.test_rule(rng, eig_vals, A, alg; mode = Mooncake.ReverseMode, atol = atol, rtol = rtol, is_primitive = false)
            test_pullbacks_match(rng, eig_vals!, eig_vals, A, D.diag, ΔD2.diag, alg)
        end
        @testset "eig_trunc" begin
            for r in 1:4:m
                truncalg = TruncatedAlgorithm(alg, truncrank(r; by = abs))
                ind = MatrixAlgebraKit.findtruncated(Ddiag, truncalg.trunc)
                Dtrunc = Diagonal(diagview(D)[ind])
                Vtrunc = V[:, ind]
                ΔDtrunc = Diagonal(diagview(ΔD2)[ind])
                ΔVtrunc = ΔV[:, ind]
                dDtrunc = make_mooncake_tangent(ΔDtrunc)
                dVtrunc = make_mooncake_tangent(ΔVtrunc)
                dDVtrunc = Mooncake.build_tangent(typeof((ΔDtrunc, ΔVtrunc, zero(real(T)))), dDtrunc, dVtrunc, zero(real(T)))
                Mooncake.TestUtils.test_rule(rng, eig_trunc, A, truncalg; mode = Mooncake.ReverseMode, output_tangent = dDVtrunc, atol = atol, rtol = rtol, is_primitive = false)
                test_pullbacks_match(rng, eig_trunc!, eig_trunc, A, (D, V), (ΔD2, ΔV), truncalg; rdata = (Mooncake.NoRData(), Mooncake.NoRData(), zero(real(T))))
            end
            truncalg = TruncatedAlgorithm(alg, truncrank(5; by = real))
            ind = MatrixAlgebraKit.findtruncated(Ddiag, truncalg.trunc)
            Dtrunc = Diagonal(diagview(D)[ind])
            Vtrunc = V[:, ind]
            ΔDtrunc = Diagonal(diagview(ΔD2)[ind])
            ΔVtrunc = ΔV[:, ind]
            dDtrunc = make_mooncake_tangent(ΔDtrunc)
            dVtrunc = make_mooncake_tangent(ΔVtrunc)
            dDVtrunc = Mooncake.build_tangent(typeof((ΔDtrunc, ΔVtrunc, zero(real(T)))), dDtrunc, dVtrunc, zero(real(T)))
            Mooncake.TestUtils.test_rule(rng, eig_trunc, A, truncalg; mode = Mooncake.ReverseMode, output_tangent = dDVtrunc, atol = atol, rtol = rtol, is_primitive = false)
            test_pullbacks_match(rng, eig_trunc!, eig_trunc, A, (D, V), (ΔD2, ΔV), truncalg; rdata = (Mooncake.NoRData(), Mooncake.NoRData(), zero(real(T))))
        end
    end
end

function copy_eigh_full(A, alg; kwargs...)
    A = (A + A') / 2
    return eigh_full(A, alg; kwargs...)
end

function copy_eigh_full!(A, DV, alg; kwargs...)
    A = (A + A') / 2
    return eigh_full!(A, DV, alg; kwargs...)
end

function copy_eigh_vals(A, alg; kwargs...)
    A = (A + A') / 2
    return eigh_vals(A, alg; kwargs...)
end

function copy_eigh_vals!(A, D, alg; kwargs...)
    A = (A + A') / 2
    return eigh_vals!(A, D, alg; kwargs...)
end

function copy_eigh_trunc(A, alg; kwargs...)
    A = (A + A') / 2
    return eigh_trunc(A, alg; kwargs...)
end

function copy_eigh_trunc!(A, DV, alg; kwargs...)
    A = (A + A') / 2
    return eigh_trunc!(A, DV, alg; kwargs...)
end

MatrixAlgebraKit.copy_input(::typeof(copy_eigh_full), A) = MatrixAlgebraKit.copy_input(eigh_full, A)
MatrixAlgebraKit.copy_input(::typeof(copy_eigh_vals), A) = MatrixAlgebraKit.copy_input(eigh_vals, A)
MatrixAlgebraKit.copy_input(::typeof(copy_eigh_trunc), A) = MatrixAlgebraKit.copy_input(eigh_trunc, A)

@timedtestset "EIGH AD Rules with eltype $T" for T in ETs
    rng = StableRNG(12345)
    m = 19
    atol = rtol = m * m * precision(T)
    A = randn(rng, T, m, m)
    A = A + A'
    D, V = eigh_full(A)
    ΔV = randn(rng, T, m, m)
    ΔV = remove_eighgauge_dependence!(ΔV, D, V; degeneracy_atol = atol)
    ΔD = randn(rng, real(T), m, m)
    ΔD2 = Diagonal(randn(rng, real(T), m))
    dD = make_mooncake_tangent(ΔD2)
    dV = make_mooncake_tangent(ΔV)
    dDV = Mooncake.build_tangent(typeof((ΔD2, ΔV)), dD, dV)
    Ddiag = diagview(D)
    @testset for alg in (
            LAPACK_QRIteration(),
            #LAPACK_DivideAndConquer(),
            #LAPACK_Bisection(),
            #LAPACK_MultipleRelativelyRobustRepresentations(), # expensive on CI
        )
        @testset "eigh_full" begin
            Mooncake.TestUtils.test_rule(rng, copy_eigh_full, A, alg; mode = Mooncake.ReverseMode, output_tangent = dDV, is_primitive = false, atol = atol, rtol = rtol)
            test_pullbacks_match(rng, copy_eigh_full!, copy_eigh_full, A, (D, V), (ΔD2, ΔV), alg)
        end
        @testset "eigh_vals" begin
            Mooncake.TestUtils.test_rule(rng, copy_eigh_vals, A, alg; mode = Mooncake.ReverseMode, is_primitive = false, atol = atol, rtol = rtol)
            test_pullbacks_match(rng, copy_eigh_vals!, copy_eigh_vals, A, D.diag, ΔD2.diag, alg)
        end
        @testset "eigh_trunc" begin
            for r in 1:4:m
                truncalg = TruncatedAlgorithm(alg, truncrank(r; by = abs))
                ind = MatrixAlgebraKit.findtruncated(Ddiag, truncalg.trunc)
                Dtrunc = Diagonal(diagview(D)[ind])
                Vtrunc = V[:, ind]
                ΔDtrunc = Diagonal(diagview(ΔD2)[ind])
                ΔVtrunc = ΔV[:, ind]
                dDtrunc = make_mooncake_tangent(ΔDtrunc)
                dVtrunc = make_mooncake_tangent(ΔVtrunc)
                dDVtrunc = Mooncake.build_tangent(typeof((ΔDtrunc, ΔVtrunc, zero(real(T)))), dDtrunc, dVtrunc, zero(real(T)))
                Mooncake.TestUtils.test_rule(rng, copy_eigh_trunc, A, truncalg; mode = Mooncake.ReverseMode, output_tangent = dDVtrunc, atol = atol, rtol = rtol, is_primitive = false)
                test_pullbacks_match(rng, copy_eigh_trunc!, copy_eigh_trunc, A, (D, V), (ΔD2, ΔV), truncalg; rdata = (Mooncake.NoRData(), Mooncake.NoRData(), zero(real(T))))
            end
            truncalg = TruncatedAlgorithm(alg, trunctol(; atol = maximum(abs, Ddiag) / 2))
            ind = MatrixAlgebraKit.findtruncated(Ddiag, truncalg.trunc)
            Dtrunc = Diagonal(diagview(D)[ind])
            Vtrunc = V[:, ind]
            ΔDtrunc = Diagonal(diagview(ΔD2)[ind])
            ΔVtrunc = ΔV[:, ind]
            dDtrunc = make_mooncake_tangent(ΔDtrunc)
            dVtrunc = make_mooncake_tangent(ΔVtrunc)
            dDVtrunc = Mooncake.build_tangent(typeof((ΔDtrunc, ΔVtrunc, zero(real(T)))), dDtrunc, dVtrunc, zero(real(T)))
            Mooncake.TestUtils.test_rule(rng, copy_eigh_trunc, A, truncalg; mode = Mooncake.ReverseMode, output_tangent = dDVtrunc, atol = atol, rtol = rtol, is_primitive = false)
            test_pullbacks_match(rng, copy_eigh_trunc!, copy_eigh_trunc, A, (D, V), (ΔD2, ΔV), truncalg; rdata = (Mooncake.NoRData(), Mooncake.NoRData(), zero(real(T))))
        end
    end
end

@timedtestset "SVD AD Rules with eltype $T" for T in ETs
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        minmn = min(m, n)
        @testset for alg in (
                LAPACK_QRIteration(),
                #LAPACK_DivideAndConquer(), # expensive on CI
            )
            @testset "svd_compact" begin
                ΔU = randn(rng, T, m, minmn)
                ΔS = randn(rng, real(T), minmn, minmn)
                ΔS2 = Diagonal(randn(rng, real(T), minmn))
                ΔVᴴ = randn(rng, T, minmn, n)
                U, S, Vᴴ = svd_compact(A)
                ΔU, ΔVᴴ = remove_svdgauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ; degeneracy_atol = atol)
                dS = make_mooncake_tangent(ΔS2)
                dU = make_mooncake_tangent(ΔU)
                dVᴴ = make_mooncake_tangent(ΔVᴴ)
                dUSVᴴ = Mooncake.build_tangent(typeof((ΔU, ΔS2, ΔVᴴ)), dU, dS, dVᴴ)
                Mooncake.TestUtils.test_rule(rng, svd_compact, A, alg; mode = Mooncake.ReverseMode, output_tangent = dUSVᴴ, atol = atol, rtol = rtol)
                test_pullbacks_match(rng, svd_compact!, svd_compact, A, (U, S, Vᴴ), (ΔU, ΔS2, ΔVᴴ), alg)
            end
            @testset "svd_full" begin
                ΔU = randn(rng, T, m, minmn)
                ΔS = randn(rng, real(T), minmn, minmn)
                ΔS2 = Diagonal(randn(rng, real(T), minmn))
                ΔVᴴ = randn(rng, T, minmn, n)
                U, S, Vᴴ = svd_compact(A)
                ΔU, ΔVᴴ = remove_svdgauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ; degeneracy_atol = atol)
                ΔUfull = zeros(T, m, m)
                ΔSfull = zeros(real(T), m, n)
                ΔVᴴfull = zeros(T, n, n)
                U, S, Vᴴ = svd_full(A)
                view(ΔUfull, :, 1:minmn) .= ΔU
                view(ΔVᴴfull, 1:minmn, :) .= ΔVᴴ
                diagview(ΔSfull)[1:minmn] .= diagview(ΔS2)
                dS = make_mooncake_tangent(ΔSfull)
                dU = make_mooncake_tangent(ΔUfull)
                dVᴴ = make_mooncake_tangent(ΔVᴴfull)
                dUSVᴴ = Mooncake.build_tangent(typeof((ΔUfull, ΔSfull, ΔVᴴfull)), dU, dS, dVᴴ)
                Mooncake.TestUtils.test_rule(rng, svd_full, A, alg; mode = Mooncake.ReverseMode, output_tangent = dUSVᴴ, atol = atol, rtol = rtol)
                test_pullbacks_match(rng, svd_full!, svd_full, A, (U, S, Vᴴ), (ΔUfull, ΔSfull, ΔVᴴfull), alg)
            end
            @testset "svd_vals" begin
                Mooncake.TestUtils.test_rule(rng, svd_vals, A, alg; mode = Mooncake.ReverseMode, atol = atol, rtol = rtol)
                S = svd_vals(A, alg)
                test_pullbacks_match(rng, svd_vals!, svd_vals, A, S, randn(rng, real(T), minmn), alg)
            end
            @testset "svd_trunc" begin
                @testset for r in 1:4:minmn
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
                    dStrunc = make_mooncake_tangent(ΔStrunc)
                    dUtrunc = make_mooncake_tangent(ΔUtrunc)
                    dVᴴtrunc = make_mooncake_tangent(ΔVᴴtrunc)
                    ϵ = zero(real(T))
                    dUSVᴴerr = Mooncake.build_tangent(typeof((ΔU, ΔS2, ΔVᴴ, ϵ)), dUtrunc, dStrunc, dVᴴtrunc, ϵ)
                    Mooncake.TestUtils.test_rule(rng, svd_trunc_with_err, A, truncalg; mode = Mooncake.ReverseMode, output_tangent = dUSVᴴerr, atol = atol, rtol = rtol)
                    test_pullbacks_match(rng, svd_trunc_with_err!, svd_trunc_with_err, A, (U, S, Vᴴ), (ΔU, ΔS2, ΔVᴴ), truncalg; rdata = (Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), zero(real(T))))
                    dUSVᴴ = Mooncake.build_tangent(typeof((ΔU, ΔS2, ΔVᴴ)), dUtrunc, dStrunc, dVᴴtrunc)
                    Mooncake.TestUtils.test_rule(rng, svd_trunc, A, truncalg; mode = Mooncake.ReverseMode, output_tangent = dUSVᴴ, atol = atol, rtol = rtol)
                    test_pullbacks_match(rng, svd_trunc!, svd_trunc, A, (U, S, Vᴴ), (ΔU, ΔS2, ΔVᴴ), truncalg)
                end
                @testset "trunctol" begin
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
                    dStrunc = make_mooncake_tangent(ΔStrunc)
                    dUtrunc = make_mooncake_tangent(ΔUtrunc)
                    dVᴴtrunc = make_mooncake_tangent(ΔVᴴtrunc)
                    ϵ = zero(real(T))
                    dUSVᴴerr = Mooncake.build_tangent(typeof((ΔU, ΔS2, ΔVᴴ, ϵ)), dUtrunc, dStrunc, dVᴴtrunc, ϵ)
                    Mooncake.TestUtils.test_rule(rng, svd_trunc_with_err, A, truncalg; mode = Mooncake.ReverseMode, output_tangent = dUSVᴴerr, atol = atol, rtol = rtol)
                    test_pullbacks_match(rng, svd_trunc_with_err!, svd_trunc_with_err, A, (U, S, Vᴴ), (ΔU, ΔS2, ΔVᴴ), truncalg; rdata = (Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), zero(real(T))))
                    dUSVᴴ = Mooncake.build_tangent(typeof((ΔU, ΔS2, ΔVᴴ)), dUtrunc, dStrunc, dVᴴtrunc)
                    Mooncake.TestUtils.test_rule(rng, svd_trunc, A, truncalg; mode = Mooncake.ReverseMode, output_tangent = dUSVᴴ, atol = atol, rtol = rtol)
                    test_pullbacks_match(rng, svd_trunc!, svd_trunc, A, (U, S, Vᴴ), (ΔU, ΔS2, ΔVᴴ), truncalg)
                end
            end
        end
    end
end

@timedtestset "Polar AD Rules with eltype $T" for T in ETs
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        @testset for alg in PolarViaSVD.(
                (
                    LAPACK_QRIteration(),
                    #LAPACK_DivideAndConquer(), # expensive on CI
                )
            )
            if m >= n
                WP = left_polar(A, alg)
                Mooncake.TestUtils.test_rule(rng, left_polar, A, alg; mode = Mooncake.ReverseMode, is_primitive = false, atol = atol, rtol = rtol)
                test_pullbacks_match(rng, left_polar!, left_polar, A, WP, (randn(rng, T, m, n), randn(rng, T, n, n)), alg)
            elseif m <= n
                PWᴴ = right_polar(A, alg)
                Mooncake.TestUtils.test_rule(rng, right_polar, A, alg; mode = Mooncake.ReverseMode, is_primitive = false, atol = atol, rtol = rtol)
                test_pullbacks_match(rng, right_polar!, right_polar, A, PWᴴ, (randn(rng, T, m, m), randn(rng, T, m, n)), alg)
            end
        end
    end
end

left_orth_qr(X) = left_orth(X; alg = :qr)
left_orth_polar(X) = left_orth(X; alg = :polar)
left_null_qr(X) = left_null(X; alg = :qr)
right_orth_lq(X) = right_orth(X; alg = :lq)
right_orth_polar(X) = right_orth(X; alg = :polar)
right_null_lq(X) = right_null(X; alg = :lq)

MatrixAlgebraKit.copy_input(::typeof(left_orth_qr), A) = MatrixAlgebraKit.copy_input(left_orth, A)
MatrixAlgebraKit.copy_input(::typeof(left_orth_polar), A) = MatrixAlgebraKit.copy_input(left_orth, A)
MatrixAlgebraKit.copy_input(::typeof(left_null_qr), A) = MatrixAlgebraKit.copy_input(left_null, A)
MatrixAlgebraKit.copy_input(::typeof(right_orth_lq), A) = MatrixAlgebraKit.copy_input(right_orth, A)
MatrixAlgebraKit.copy_input(::typeof(right_orth_polar), A) = MatrixAlgebraKit.copy_input(right_orth, A)
MatrixAlgebraKit.copy_input(::typeof(right_null_lq), A) = MatrixAlgebraKit.copy_input(right_null, A)

@timedtestset "Orth and null with eltype $T" for T in ETs
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        VC = left_orth(A)
        CVᴴ = right_orth(A)
        Mooncake.TestUtils.test_rule(rng, left_orth, A; mode = Mooncake.ReverseMode, atol = atol, rtol = rtol, is_primitive = false)
        test_pullbacks_match(rng, left_orth!, left_orth, A, VC, (randn(rng, T, size(VC[1])...), randn(rng, T, size(VC[2])...)))
        Mooncake.TestUtils.test_rule(rng, right_orth, A; mode = Mooncake.ReverseMode, atol = atol, rtol = rtol, is_primitive = false)
        test_pullbacks_match(rng, right_orth!, right_orth, A, CVᴴ, (randn(rng, T, size(CVᴴ[1])...), randn(rng, T, size(CVᴴ[2])...)))

        Mooncake.TestUtils.test_rule(rng, left_orth_qr, A; mode = Mooncake.ReverseMode, atol = atol, rtol = rtol, is_primitive = false)
        test_pullbacks_match(rng, ((X, VC) -> left_orth!(X, VC; alg = :qr)), left_orth_qr, A, VC, (randn(rng, T, size(VC[1])...), randn(rng, T, size(VC[2])...)))
        if m >= n
            Mooncake.TestUtils.test_rule(rng, left_orth_polar, A; mode = Mooncake.ReverseMode, atol = atol, rtol = rtol, is_primitive = false)
            test_pullbacks_match(rng, ((X, VC) -> left_orth!(X, VC; alg = :polar)), left_orth_polar, A, VC, (randn(rng, T, size(VC[1])...), randn(rng, T, size(VC[2])...)))
        end

        N = left_orth(A; alg = :qr)[1] * randn(rng, T, min(m, n), m - min(m, n))
        ΔN = left_orth(A; alg = :qr)[1] * randn(rng, T, min(m, n), m - min(m, n))
        dN = make_mooncake_tangent(ΔN)
        Mooncake.TestUtils.test_rule(rng, left_null_qr, A; mode = Mooncake.ReverseMode, atol = atol, rtol = rtol, is_primitive = false, output_tangent = dN)
        test_pullbacks_match(rng, ((X, N) -> left_null!(X, N; alg = :qr)), left_null_qr, A, N, ΔN)

        Mooncake.TestUtils.test_rule(rng, right_orth_lq, A; mode = Mooncake.ReverseMode, atol = atol, rtol = rtol, is_primitive = false)
        test_pullbacks_match(rng, ((X, CVᴴ) -> right_orth!(X, CVᴴ; alg = :lq)), right_orth_lq, A, CVᴴ, (randn(rng, T, size(CVᴴ[1])...), randn(rng, T, size(CVᴴ[2])...)))

        if m <= n
            Mooncake.TestUtils.test_rule(rng, right_orth_polar, A; mode = Mooncake.ReverseMode, atol = atol, rtol = rtol, is_primitive = false)
            test_pullbacks_match(rng, ((X, CVᴴ) -> right_orth!(X, CVᴴ; alg = :polar)), right_orth_polar, A, CVᴴ, (randn(rng, T, size(CVᴴ[1])...), randn(rng, T, size(CVᴴ[2])...)))
        end

        Nᴴ = randn(rng, T, n - min(m, n), min(m, n)) * right_orth(A; alg = :lq)[2]
        ΔNᴴ = randn(rng, T, n - min(m, n), min(m, n)) * right_orth(A; alg = :lq)[2]
        dNᴴ = make_mooncake_tangent(ΔNᴴ)
        Mooncake.TestUtils.test_rule(rng, right_null_lq, A; mode = Mooncake.ReverseMode, atol = atol, rtol = rtol, is_primitive = false, output_tangent = dNᴴ)
        test_pullbacks_match(rng, ((X, Nᴴ) -> right_null!(X, Nᴴ; alg = :lq)), right_null_lq, A, Nᴴ, ΔNᴴ)
    end
end
