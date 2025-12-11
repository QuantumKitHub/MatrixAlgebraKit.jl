using TestExtras
using MatrixAlgebraKit
using Mooncake, Mooncake.TestUtils
using Mooncake: rrule!!
using MatrixAlgebraKit: diagview, TruncatedAlgorithm, PolarViaSVD, eigh_trunc

make_mooncake_tangent(ΔAelem::T) where {T <: Real} = ΔAelem
make_mooncake_tangent(ΔAelem::T) where {T <: Complex} = Mooncake.build_tangent(T, real(ΔAelem), imag(ΔAelem))
make_mooncake_tangent(ΔA::AbstractMatrix{<:Real}) = ΔA
make_mooncake_tangent(ΔA::AbstractVector{<:Real}) = ΔA
make_mooncake_tangent(ΔA::AbstractMatrix{T}) where {T <: Complex} = map(make_mooncake_tangent, ΔA)
make_mooncake_tangent(ΔA::AbstractVector{T}) where {T <: Complex} = map(make_mooncake_tangent, ΔA)
make_mooncake_tangent(ΔD::Diagonal{T}) where {T <: Real} = Mooncake.build_tangent(typeof(ΔD), diagview(ΔD))
make_mooncake_tangent(ΔD::Diagonal{T}) where {T <: Complex} = Mooncake.build_tangent(typeof(ΔD), map(make_mooncake_tangent, diagview(ΔD)))

make_mooncake_tangent(T::Tuple) = Mooncake.build_tangent(typeof(T), make_mooncake_tangent.(T)...)

make_mooncake_fdata(x) = make_mooncake_tangent(x)
make_mooncake_fdata(x::Diagonal) = Mooncake.FData((diag = make_mooncake_tangent(x.diag),))

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
    test_pullbacks_match(f!, f, A, args, Δargs, alg = nothing; rdata = Mooncake.NoRData())

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
function test_pullbacks_match(f!, f, A, args, Δargs, alg = nothing; rdata = Mooncake.NoRData())
    f_c = isnothing(alg) ? (A, args) -> f!(MatrixAlgebraKit.copy_input(f, A), args) : (A, args, alg) -> f!(MatrixAlgebraKit.copy_input(f, A), args, alg)
    sig = isnothing(alg) ? Tuple{typeof(f_c), typeof(A), typeof(args)} : Tuple{typeof(f_c), typeof(A), typeof(args), typeof(alg)}
    rvs_interp = Mooncake.get_interpreter(Mooncake.ReverseMode)
    rrule = Mooncake.build_rrule(rvs_interp, sig)
    ΔA = randn!(similar(A))

    dA_copy = _get_copying_derivative(f_c, rrule, A, ΔA, args, Δargs, alg, rdata)
    dA_inplace = _get_inplace_derivative(f!, A, ΔA, args, Δargs, alg, rdata)

    dA_inplace_ = Mooncake.arrayify(A, dA_inplace)[2]
    dA_copy_ = Mooncake.arrayify(A, dA_copy)[2]
    @test dA_inplace_ ≈ dA_copy_
    return
end

function test_mooncake(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "Mooncake AD $summary_str" begin
        test_mooncake_qr(T, sz; kwargs...)
        test_mooncake_lq(T, sz; kwargs...)
        if length(sz) == 1 || sz[1] == sz[2]
            test_mooncake_eig(T, sz; kwargs...)
            test_mooncake_eigh(T, sz; kwargs...)
        end
        test_mooncake_svd(T, sz; kwargs...)
        test_mooncake_polar(T, sz; kwargs...)
        test_mooncake_orthnull(T, sz; kwargs...)
    end
end

function test_mooncake_qr(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "QR Mooncake AD rules $summary_str" begin
        A = instantiate_matrix(T, sz)
        @testset "qr_compact" begin
            QR, ΔQR = ad_qr_compact_setup(A)
            Mooncake.TestUtils.test_rule(rng, qr_compact, A; is_primitive = false, mode = Mooncake.ReverseMode, atol = atol, rtol = rtol)
            test_pullbacks_match(qr_compact!, qr_compact, A, QR, ΔQR)
        end
        @testset "qr_null" begin
            N, ΔN = ad_qr_null_setup(A)
            dN = make_mooncake_tangent(copy(ΔN))
            Mooncake.TestUtils.test_rule(rng, qr_null, A; is_primitive = false, mode = Mooncake.ReverseMode, output_tangent = dN, atol = atol, rtol = rtol)
            test_pullbacks_match(qr_null!, qr_null, A, N, ΔN)
        end
        @testset "qr_full" begin
            QR, ΔQR = ad_qr_full_setup(A)
            dQR = make_mooncake_tangent(ΔQR)
            Mooncake.TestUtils.test_rule(rng, qr_full, A; is_primitive = false, mode = Mooncake.ReverseMode, output_tangent = dQR, atol = atol, rtol = rtol)
            test_pullbacks_match(qr_full!, qr_full, A, QR, ΔQR)
        end
        @testset "qr_compact - rank-deficient A" begin
            m, n = size(A)
            r = min(m, n) - 5
            Ard = instantiate_matrix(T, (m, r)) * instantiate_matrix(T, (r, n))
            QR, ΔQR = ad_qr_rd_compact_setup(Ard)
            dQR = make_mooncake_tangent(ΔQR)
            Mooncake.TestUtils.test_rule(rng, qr_compact, Ard; is_primitive = false, mode = Mooncake.ReverseMode, output_tangent = dQR, atol = atol, rtol = rtol)
            test_pullbacks_match(qr_compact!, qr_compact, Ard, QR, ΔQR)
        end
    end
end

function test_mooncake_lq(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "LQ Mooncake AD rules $summary_str" begin
        A = instantiate_matrix(T, sz)
        @testset "lq_compact" begin
            LQ, ΔLQ = ad_lq_compact_setup(A)
            Mooncake.TestUtils.test_rule(rng, lq_compact, A; is_primitive = false, mode = Mooncake.ReverseMode, atol = atol, rtol = rtol)
            test_pullbacks_match(lq_compact!, lq_compact, A, LQ, ΔLQ)
        end
        @testset "lq_null" begin
            Nᴴ, ΔNᴴ = ad_lq_null_setup(A)
            dNᴴ = make_mooncake_tangent(ΔNᴴ)
            Mooncake.TestUtils.test_rule(rng, lq_null, A; is_primitive = false, mode = Mooncake.ReverseMode, output_tangent = dNᴴ, atol = atol, rtol = rtol)
            test_pullbacks_match(lq_null!, lq_null, A, Nᴴ, ΔNᴴ)
        end
        @testset "lq_full" begin
            LQ, ΔLQ = ad_lq_full_setup(A)
            dLQ = make_mooncake_tangent(ΔLQ)
            Mooncake.TestUtils.test_rule(rng, lq_full, A; is_primitive = false, mode = Mooncake.ReverseMode, output_tangent = dLQ, atol = atol, rtol = rtol)
            test_pullbacks_match(lq_full!, lq_full, A, LQ, ΔLQ)
        end
        @testset "lq_compact - rank-deficient A" begin
            m, n = size(A)
            r = min(m, n) - 5
            Ard = instantiate_matrix(T, (m, r)) * instantiate_matrix(T, (r, n))
            LQ, ΔLQ = ad_lq_rd_compact_setup(Ard)
            dLQ = make_mooncake_tangent(ΔLQ)
            Mooncake.TestUtils.test_rule(rng, lq_compact, Ard; is_primitive = false, mode = Mooncake.ReverseMode, output_tangent = dLQ, atol = atol, rtol = rtol)
            test_pullbacks_match(lq_compact!, lq_compact, Ard, LQ, ΔLQ)
        end
    end
end

function test_mooncake_eig(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "EIG Mooncake AD rules $summary_str" begin
        A = instantiate_matrix(T, sz)
        m = size(A, 1)
        @testset "eig_full" begin
            DV, ΔDV, ΔD2V = ad_eig_full_setup(A)
            dDV = make_mooncake_tangent(ΔD2V)
            Mooncake.TestUtils.test_rule(rng, eig_full, A; is_primitive = false, mode = Mooncake.ReverseMode, output_tangent = dDV, atol = atol, rtol = rtol)
            test_pullbacks_match(eig_full!, eig_full, A, DV, ΔD2V)
        end
        @testset "eig_vals" begin
            D, ΔD = ad_eig_vals_setup(A)
            dD = make_mooncake_tangent(ΔD)
            Mooncake.TestUtils.test_rule(rng, eig_vals, A; is_primitive = false, mode = Mooncake.ReverseMode, atol = atol, rtol = rtol)
            test_pullbacks_match(eig_vals!, eig_vals, A, D, ΔD)
        end
        @testset "eig_trunc" begin
            for r in 1:4:m
                truncalg = TruncatedAlgorithm(MatrixAlgebraKit.default_eig_algorithm(A), truncrank(r; by = abs))
                DV, _, ΔDV, ΔDVtrunc = ad_eig_trunc_setup(A, truncalg)
                ϵ = zero(real(T))
                dDVerr = make_mooncake_tangent((ΔDVtrunc..., ϵ))
                Mooncake.TestUtils.test_rule(rng, eig_trunc, A, truncalg; mode = Mooncake.ReverseMode, output_tangent = dDVerr, atol = atol, rtol = rtol)
                test_pullbacks_match(eig_trunc!, eig_trunc, A, DV, ΔDV, truncalg; rdata = (Mooncake.NoRData(), Mooncake.NoRData(), zero(real(T))))
            end
            truncalg = TruncatedAlgorithm(MatrixAlgebraKit.default_eig_algorithm(A), truncrank(5; by = real))
            DV, _, ΔDV, ΔDVtrunc = ad_eig_trunc_setup(A, truncalg)
            ϵ = zero(real(T))
            dDVerr = make_mooncake_tangent((ΔDVtrunc..., ϵ))
            Mooncake.TestUtils.test_rule(rng, eig_trunc, A, truncalg; mode = Mooncake.ReverseMode, output_tangent = dDVerr, atol = atol, rtol = rtol)
            test_pullbacks_match(eig_trunc!, eig_trunc, A, DV, ΔDV, truncalg; rdata = (Mooncake.NoRData(), Mooncake.NoRData(), zero(real(T))))
        end
    end
end

function test_mooncake_eigh(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "EIGH Mooncake AD rules $summary_str" begin
        A = instantiate_matrix(T, sz)
        A = A + A'
        m = size(A, 1)
        @testset "eigh_full" begin
            DV, ΔDV, ΔD2V = ad_eigh_full_setup(A)
            dDV = make_mooncake_tangent(ΔD2V)
            Mooncake.TestUtils.test_rule(rng, copy_eigh_full, A; mode = Mooncake.ReverseMode, output_tangent = dDV, is_primitive = false, atol = atol, rtol = rtol)
            test_pullbacks_match(copy_eigh_full!, copy_eigh_full, A, DV, ΔD2V)
        end
        @testset "eigh_vals" begin
            D, ΔD = ad_eigh_vals_setup(A)
            dD = make_mooncake_tangent(ΔD)
            Mooncake.TestUtils.test_rule(rng, copy_eigh_vals, A; mode = Mooncake.ReverseMode, output_tangent = dD, is_primitive = false, atol = atol, rtol = rtol)
            test_pullbacks_match(copy_eigh_vals!, copy_eigh_vals, A, D, ΔD)
        end
        @testset "eigh_trunc" begin
            for r in 1:4:m
                truncalg = TruncatedAlgorithm(MatrixAlgebraKit.default_eigh_algorithm(A), truncrank(r; by = abs))
                DV, _, ΔDV, ΔDVtrunc = ad_eigh_trunc_setup(A, truncalg)
                ϵ = zero(real(T))
                dDVerr = make_mooncake_tangent((ΔDVtrunc..., ϵ))
                Mooncake.TestUtils.test_rule(rng, copy_eigh_trunc, A, truncalg; mode = Mooncake.ReverseMode, output_tangent = dDVerr, atol = atol, rtol = rtol, is_primitive = false)
                test_pullbacks_match(copy_eigh_trunc!, copy_eigh_trunc, A, DV, ΔDV, truncalg; rdata = (Mooncake.NoRData(), Mooncake.NoRData(), zero(real(T))))
            end
            D = eigh_vals(A / 2)
            truncalg = TruncatedAlgorithm(MatrixAlgebraKit.default_eigh_algorithm(A), trunctol(; atol = maximum(abs, D) / 2))
            DV, _, ΔDV, ΔDVtrunc = ad_eigh_trunc_setup(A, truncalg)
            ϵ = zero(real(T))
            dDVerr = make_mooncake_tangent((ΔDVtrunc..., ϵ))
            Mooncake.TestUtils.test_rule(rng, copy_eigh_trunc, A, truncalg; mode = Mooncake.ReverseMode, output_tangent = dDVerr, atol = atol, rtol = rtol, is_primitive = false)
            test_pullbacks_match(copy_eigh_trunc!, copy_eigh_trunc, A, DV, ΔDV, truncalg; rdata = (Mooncake.NoRData(), Mooncake.NoRData(), zero(real(T))))
        end
    end
end

function test_mooncake_svd(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "SVD Mooncake AD rules $summary_str" begin
        A = instantiate_matrix(T, sz)
        minmn = min(size(A)...)
        @testset "svd_compact" begin
            USVᴴ, _, ΔUSVᴴ = ad_svd_compact_setup(A)
            dUSVᴴ = make_mooncake_tangent(ΔUSVᴴ)
            Mooncake.TestUtils.test_rule(rng, svd_compact, A; is_primitive = false, mode = Mooncake.ReverseMode, output_tangent = dUSVᴴ, atol = atol, rtol = rtol)
            test_pullbacks_match(svd_compact!, svd_compact, A, USVᴴ, ΔUSVᴴ)
        end
        @testset "svd_full" begin
            USVᴴ, ΔUSVᴴ = ad_svd_full_setup(A)
            dUSVᴴ = make_mooncake_tangent(ΔUSVᴴ)
            Mooncake.TestUtils.test_rule(rng, svd_full, A; is_primitive = false, mode = Mooncake.ReverseMode, output_tangent = dUSVᴴ, atol = atol, rtol = rtol)
            test_pullbacks_match(svd_full!, svd_full, A, USVᴴ, ΔUSVᴴ)
        end
        @testset "svd_vals" begin
            S, ΔS = ad_svd_vals_setup(A)
            Mooncake.TestUtils.test_rule(rng, svd_vals, A; is_primitive = false, mode = Mooncake.ReverseMode, atol = atol, rtol = rtol)
            test_pullbacks_match(svd_vals!, svd_vals, A, S, ΔS)
        end
        @testset "svd_trunc" begin
            S, ΔS = ad_svd_vals_setup(A)
            @testset for r in 1:4:minmn
                truncalg = TruncatedAlgorithm(MatrixAlgebraKit.default_svd_algorithm(A), truncrank(r))
                USVᴴ, ΔUSVᴴ, ΔUSVᴴtrunc = ad_svd_trunc_setup(A, truncalg)
                ϵ = zero(real(T))
                dUSVᴴerr = make_mooncake_tangent((ΔUSVᴴtrunc..., ϵ))
                Mooncake.TestUtils.test_rule(rng, svd_trunc, A, truncalg; mode = Mooncake.ReverseMode, output_tangent = dUSVᴴerr, atol = atol, rtol = rtol)
                test_pullbacks_match(svd_trunc!, svd_trunc, A, USVᴴ, ΔUSVᴴ, truncalg; rdata = (Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), zero(real(T))))
            end
            @testset "trunctol" begin
                truncalg = TruncatedAlgorithm(MatrixAlgebraKit.default_svd_algorithm(A), trunctol(atol = S[1, 1] / 2))
                USVᴴ, ΔUSVᴴ, ΔUSVᴴtrunc = ad_svd_trunc_setup(A, truncalg)
                ϵ = zero(real(T))
                dUSVᴴerr = make_mooncake_tangent((ΔUSVᴴtrunc..., ϵ))
                Mooncake.TestUtils.test_rule(rng, svd_trunc, A, truncalg; mode = Mooncake.ReverseMode, output_tangent = dUSVᴴerr, atol = atol, rtol = rtol)
                test_pullbacks_match(svd_trunc!, svd_trunc, A, USVᴴ, ΔUSVᴴ, truncalg; rdata = (Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), zero(real(T))))
            end
        end
    end
end

function test_mooncake_polar(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "Polar Mooncake AD rules $summary_str" begin
        A = instantiate_matrix(T, sz)
        m, n = size(A)
        @testset "left_polar" begin
            if m >= n
                WP, ΔWP = ad_left_polar_setup(A)
                Mooncake.TestUtils.test_rule(rng, left_polar, A; is_primitive = false, mode = Mooncake.ReverseMode, atol = atol, rtol = rtol)
                test_pullbacks_match(left_polar!, left_polar, A, WP, ΔWP)
            end
        end
        @testset "right_polar" begin
            if m <= n
                PWᴴ, ΔPWᴴ = ad_right_polar_setup(A)
                Mooncake.TestUtils.test_rule(rng, right_polar, A; is_primitive = false, mode = Mooncake.ReverseMode, atol = atol, rtol = rtol)
                test_pullbacks_match(right_polar!, right_polar, A, PWᴴ, ΔPWᴴ)
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

function test_mooncake_orthnull(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "Orthnull Mooncake AD rules $summary_str" begin
        A = instantiate_matrix(T, sz)
        m, n = size(A)
        VC, ΔVC = ad_left_orth_setup(A)
        CVᴴ, ΔCVᴴ = ad_right_orth_setup(A)
        Mooncake.TestUtils.test_rule(rng, left_orth, A; mode = Mooncake.ReverseMode, atol = atol, rtol = rtol, is_primitive = false)
        test_pullbacks_match(left_orth!, left_orth, A, VC, ΔVC)
        Mooncake.TestUtils.test_rule(rng, right_orth, A; mode = Mooncake.ReverseMode, atol = atol, rtol = rtol, is_primitive = false)
        test_pullbacks_match(right_orth!, right_orth, A, CVᴴ, ΔCVᴴ)

        Mooncake.TestUtils.test_rule(rng, left_orth_qr, A; mode = Mooncake.ReverseMode, atol = atol, rtol = rtol, is_primitive = false)
        test_pullbacks_match(((X, VC) -> left_orth!(X, VC; alg = :qr)), left_orth_qr, A, VC, ΔVC)
        if m >= n
            Mooncake.TestUtils.test_rule(rng, left_orth_polar, A; mode = Mooncake.ReverseMode, atol = atol, rtol = rtol, is_primitive = false)
            test_pullbacks_match(((X, VC) -> left_orth!(X, VC; alg = :polar)), left_orth_polar, A, VC, ΔVC)
        end

        N, ΔN = ad_left_null_setup(A)
        dN = make_mooncake_tangent(ΔN)
        Mooncake.TestUtils.test_rule(rng, left_null_qr, A; mode = Mooncake.ReverseMode, atol = atol, rtol = rtol, is_primitive = false, output_tangent = dN)
        test_pullbacks_match(((X, N) -> left_null!(X, N; alg = :qr)), left_null_qr, A, N, ΔN)

        Mooncake.TestUtils.test_rule(rng, right_orth_lq, A; mode = Mooncake.ReverseMode, atol = atol, rtol = rtol, is_primitive = false)
        test_pullbacks_match(((X, CVᴴ) -> right_orth!(X, CVᴴ; alg = :lq)), right_orth_lq, A, CVᴴ, ΔCVᴴ)

        if m <= n
            Mooncake.TestUtils.test_rule(rng, right_orth_polar, A; mode = Mooncake.ReverseMode, atol = atol, rtol = rtol, is_primitive = false)
            test_pullbacks_match(((X, CVᴴ) -> right_orth!(X, CVᴴ; alg = :polar)), right_orth_polar, A, CVᴴ, ΔCVᴴ)
        end

        Nᴴ, ΔNᴴ = ad_right_null_setup(A)
        dNᴴ = make_mooncake_tangent(ΔNᴴ)
        Mooncake.TestUtils.test_rule(rng, right_null_lq, A; mode = Mooncake.ReverseMode, atol = atol, rtol = rtol, is_primitive = false, output_tangent = dNᴴ)
        test_pullbacks_match(((X, Nᴴ) -> right_null!(X, Nᴴ; alg = :lq)), right_null_lq, A, Nᴴ, ΔNᴴ)
    end
end
