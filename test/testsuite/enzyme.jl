using TestExtras
using MatrixAlgebraKit
using Enzyme, EnzymeTestUtils
using MatrixAlgebraKit: diagview, TruncatedAlgorithm
using LinearAlgebra: Diagonal, Hermitian, mul!, BlasFloat
using GenericLinearAlgebra, GenericSchur

function enz_copy_eigh_full(A, alg)
    A = (A + A') / 2
    return eigh_full(A, alg)
end

function enz_copy_eigh_full!(A, DV::Tuple, alg::MatrixAlgebraKit.AbstractAlgorithm)
    A = (A + A') / 2
    return eigh_full!(A, DV, alg)
end

function enz_copy_eigh_vals(A; kwargs...)
    A = (A + A') / 2
    return eigh_vals(A; kwargs...)
end

function enz_copy_eigh_vals!(A, D; kwargs...)
    A = (A + A') / 2
    return eigh_vals!(A, D; kwargs...)
end

function enz_copy_eigh_vals(A, alg; kwargs...)
    A = (A + A') / 2
    return eigh_vals(A, alg; kwargs...)
end

function enz_copy_eigh_vals!(A, D, alg; kwargs...)
    A = (A + A') / 2
    return eigh_vals!(A, D, alg; kwargs...)
end

function enz_copy_eigh_trunc_no_error(A, alg)
    A = (A + A') / 2
    return eigh_trunc_no_error(A, alg)
end

function enz_copy_eigh_trunc_no_error!(A, DV, alg)
    A = (A + A') / 2
    return eigh_trunc_no_error!(A, DV, alg)
end

# necessary due to name conflict with Mooncake
function enz_test_pullbacks_match(rng, f!, f, A, args, Δargs, alg = nothing; ȳ = copy.(Δargs), return_act = Duplicated)
    ΔA = randn!(similar(A))
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

function test_enzyme(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "Enzyme AD $summary_str" begin
        test_enzyme_qr(T, sz; kwargs...)
        test_enzyme_lq(T, sz; kwargs...)
        if length(sz) == 1 || sz[1] == sz[2]
            test_enzyme_eig(T, sz; kwargs...)
            # missing Enzyme rule
            eltype(T) <: BlasFloat && test_enzyme_eigh(T, sz; kwargs...)
        end
        test_enzyme_svd(T, sz; kwargs...)
        if eltype(T) <: BlasFloat
            test_enzyme_polar(T, sz; kwargs...)
            test_enzyme_orthnull(T, sz; kwargs...)
        end
    end
end

is_cpu(A) = typeof(parent(A)) <: Array

function test_enzyme_qr(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "QR Enzyme AD rules $summary_str" begin
        A = instantiate_matrix(T, sz)
        fdm = T <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
        alg = MatrixAlgebraKit.default_qr_algorithm(A)
        @testset "qr_compact" begin
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                QR, ΔQR = ad_qr_compact_setup(A)
                eltype(T) <: BlasFloat && test_reverse(qr_compact, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔQR, fdm)
                is_cpu(A) && enz_test_pullbacks_match(rng, qr_compact!, qr_compact, A, QR, ΔQR, alg)
            end
        end
        @testset "qr_null" begin
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                N, ΔN = ad_qr_null_setup(A)
                eltype(T) <: BlasFloat && test_reverse(qr_null, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔN)
                is_cpu(A) && enz_test_pullbacks_match(rng, qr_null!, qr_null, A, N, ΔN, alg)
            end
        end
        @testset "qr_full" begin
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                QR, ΔQR = ad_qr_full_setup(A)
                eltype(T) <: BlasFloat && test_reverse(qr_full, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔQR, fdm)
                is_cpu(A) && enz_test_pullbacks_match(rng, qr_full!, qr_full, A, QR, ΔQR, alg)
            end
        end
        @testset "qr_compact - rank-deficient A" begin
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                m, n = size(A)
                r = min(m, n) - 5
                Ard = instantiate_matrix(T, (m, r)) * instantiate_matrix(T, (r, n))
                QR, ΔQR = ad_qr_compact_setup(Ard)
                eltype(T) <: BlasFloat && test_reverse(qr_compact, RT, (Ard, TA), (alg, Const); atol, rtol, output_tangent = ΔQR, fdm)
                is_cpu(A) && enz_test_pullbacks_match(rng, qr_compact!, qr_compact, Ard, QR, ΔQR, alg)
            end
        end
    end
end

function test_enzyme_lq(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "LQ Enzyme AD rules $summary_str" begin
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.default_lq_algorithm(A)
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
        @testset "lq_compact" begin
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                LQ, ΔLQ = ad_lq_compact_setup(A)
                eltype(T) <: BlasFloat && test_reverse(lq_compact, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔLQ, fdm)
                is_cpu(A) && enz_test_pullbacks_match(rng, lq_compact!, lq_compact, A, LQ, ΔLQ, alg)
            end
        end
        @testset "lq_null" begin
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                Nᴴ, ΔNᴴ = ad_lq_null_setup(A)
                eltype(T) <: BlasFloat && test_reverse(lq_null, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔNᴴ)
                is_cpu(A) && enz_test_pullbacks_match(rng, lq_null!, lq_null, A, Nᴴ, ΔNᴴ, alg)
            end
        end
        @testset "lq_full" begin
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                LQ, ΔLQ = ad_lq_full_setup(A)
                eltype(T) <: BlasFloat && test_reverse(lq_full, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔLQ, fdm)
                is_cpu(A) && enz_test_pullbacks_match(rng, lq_full!, lq_full, A, LQ, ΔLQ, alg)
            end
        end
        @testset "lq_compact -- rank-deficient A" begin
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                m, n = size(A)
                r = min(m, n) - 5
                Ard = instantiate_matrix(T, (m, r)) * instantiate_matrix(T, (r, n))
                LQ, ΔLQ = ad_lq_compact_setup(Ard)
                eltype(T) <: BlasFloat && test_reverse(lq_compact, RT, (Ard, TA), (alg, Const); atol, rtol, output_tangent = ΔLQ, fdm)
                is_cpu(A) && enz_test_pullbacks_match(rng, lq_compact!, lq_compact, Ard, LQ, ΔLQ, alg)
            end
        end
    end
end

function test_enzyme_eig(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "EIG Enzyme AD rules $summary_str" begin
        A = make_eig_matrix(T, sz)
        m = size(A, 1)
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
        alg = MatrixAlgebraKit.default_eig_algorithm(A)
        @testset "eig_full" begin
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                DV, ΔDV = ad_eig_full_setup(A)
                if eltype(T) <: BlasFloat
                    test_reverse(eig_full, RT, (A, TA); fkwargs = (alg = alg,), atol, rtol, output_tangent = ΔDV, fdm)
                    is_cpu(A) && enz_test_pullbacks_match(rng, eig_full!, eig_full, A, DV, ΔDV, alg)
                else
                    is_cpu(A) && enz_test_pullbacks_match(rng, eig_full!, eig_full, A, (nothing, nothing), (nothing, nothing), alg; ȳ = ΔDV)
                end
            end
        end
        @testset "eig_vals" begin
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                D, ΔD = ad_eig_vals_setup(A)
                if eltype(T) <: BlasFloat
                    test_reverse(eig_vals, RT, (A, TA); fkwargs = (alg = alg,), atol, rtol, output_tangent = ΔD, fdm)
                    is_cpu(A) && enz_test_pullbacks_match(rng, eig_vals!, eig_vals, A, D, ΔD, alg)
                else
                    is_cpu(A) && enz_test_pullbacks_match(rng, eig_vals!, eig_vals, A, nothing, nothing, alg; ȳ = ΔD)
                end
            end
        end
        @testset "eig_trunc" begin
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                for r in 1:4:m
                    truncalg = TruncatedAlgorithm(MatrixAlgebraKit.default_eig_algorithm(A), truncrank(r; by = abs))
                    DV, _, ΔDV, ΔDVtrunc = ad_eig_trunc_setup(A, truncalg)
                    if eltype(T) <: BlasFloat
                        test_reverse(eig_trunc_no_error, RT, (A, TA), (truncalg, Const); atol, rtol, output_tangent = ΔDVtrunc, fdm)
                        is_cpu(A) && enz_test_pullbacks_match(rng, eig_trunc_no_error!, eig_trunc_no_error, A, DV, ΔDV, truncalg, ȳ = ΔDVtrunc)
                    else
                        is_cpu(A) && enz_test_pullbacks_match(rng, eig_trunc_no_error!, eig_trunc_no_error, A, (nothing, nothing), (nothing, nothing), truncalg, ȳ = ΔDVtrunc)
                    end
                end
                truncalg = TruncatedAlgorithm(MatrixAlgebraKit.default_eig_algorithm(A), truncrank(5; by = real))
                DV, _, ΔDV, ΔDVtrunc = ad_eig_trunc_setup(A, truncalg)
                if eltype(T) <: BlasFloat
                    test_reverse(eig_trunc_no_error, RT, (A, TA), (truncalg, Const); atol, rtol, output_tangent = ΔDVtrunc, fdm)
                    is_cpu(A) && enz_test_pullbacks_match(rng, eig_trunc_no_error!, eig_trunc_no_error, A, DV, ΔDV, truncalg, ȳ = ΔDVtrunc)
                else
                    is_cpu(A) && enz_test_pullbacks_match(rng, eig_trunc_no_error!, eig_trunc_no_error, A, (nothing, nothing), (nothing, nothing), truncalg, ȳ = ΔDVtrunc)
                end
            end
        end
    end
end

function test_enzyme_eigh(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "EIGH Enzyme AD rules $summary_str" begin
        A = make_eigh_matrix(T, sz)
        m = size(A, 1)
        alg = MatrixAlgebraKit.default_eigh_algorithm(A)
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
        @testset "eigh_full" begin
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                DV, ΔDV = ad_eigh_full_setup(A)
                if eltype(T) <: BlasFloat
                    test_reverse(enz_copy_eigh_full, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔDV, fdm)
                    test_reverse(enz_copy_eigh_full!, RT, (A, TA), (DV, TA), (alg, Const); atol, rtol, output_tangent = ΔDV, fdm)
                end
                is_cpu(A) && enz_test_pullbacks_match(rng, enz_copy_eigh_full!, enz_copy_eigh_full, A, DV, ΔDV, alg)
            end
        end
        @testset "eigh_vals" begin
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                D, ΔD = ad_eigh_vals_setup(A)
                eltype(T) <: BlasFloat && test_reverse(enz_copy_eigh_vals, RT, (A, TA); fkwargs = (alg = alg,), atol, rtol, output_tangent = ΔD, fdm)
                is_cpu(A) && enz_test_pullbacks_match(rng, enz_copy_eigh_vals!, enz_copy_eigh_vals, A, D, ΔD, alg)
            end
        end
        @testset "eigh_trunc" begin
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                D = eigh_vals(A / 2)
                for r in 1:4:m
                    truncalg = TruncatedAlgorithm(alg, truncrank(r; by = abs))
                    DV, _, ΔDV, ΔDVtrunc = ad_eigh_trunc_setup(A, truncalg)
                    eltype(T) <: BlasFloat && test_reverse(enz_copy_eigh_trunc_no_error, RT, (A, TA), (truncalg, Const); atol, rtol, output_tangent = ΔDVtrunc, fdm)
                    is_cpu(A) && enz_test_pullbacks_match(rng, enz_copy_eigh_trunc_no_error!, enz_copy_eigh_trunc_no_error, A, DV, ΔDV, truncalg, ȳ = ΔDVtrunc, return_act = RT)
                end
                truncalg = TruncatedAlgorithm(alg, trunctol(; atol = maximum(abs, D) / 2))
                DV, _, ΔDV, ΔDVtrunc = ad_eigh_trunc_setup(A, truncalg)
                eltype(T) <: BlasFloat && test_reverse(enz_copy_eigh_trunc_no_error, RT, (A, TA), (truncalg, Const); atol, rtol, output_tangent = ΔDVtrunc, fdm)
                is_cpu(A) && enz_test_pullbacks_match(rng, enz_copy_eigh_trunc_no_error!, enz_copy_eigh_trunc_no_error, A, DV, ΔDV, truncalg, ȳ = ΔDVtrunc, return_act = RT)
            end
        end
    end
end

function test_enzyme_svd(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "SVD Enzyme AD rules $summary_str" begin
        A = instantiate_matrix(T, sz)
        minmn = min(size(A)...)
        alg = MatrixAlgebraKit.default_svd_algorithm(A)
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
        @testset "svd_compact" begin
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                USVᴴ, ΔUSVᴴ = ad_svd_compact_setup(A)
                if eltype(T) <: BlasFloat
                    test_reverse(svd_compact, RT, (A, TA); fkwargs = (alg = alg,), atol, rtol, output_tangent = ΔUSVᴴ, fdm)
                    is_cpu(A) && enz_test_pullbacks_match(rng, svd_compact!, svd_compact, A, USVᴴ, ΔUSVᴴ, alg)
                else
                    USVᴴ = MatrixAlgebraKit.initialize_output(svd_compact!, A, alg)
                    is_cpu(A) && enz_test_pullbacks_match(rng, svd_compact!, svd_compact, A, USVᴴ, (nothing, nothing, nothing), alg; ȳ = ΔUSVᴴ)
                end
            end
        end
        @testset "svd_full" begin
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                USVᴴ, ΔUSVᴴ = ad_svd_full_setup(A)
                if eltype(T) <: BlasFloat
                    test_reverse(svd_full, RT, (A, TA); fkwargs = (alg = alg,), atol, rtol, output_tangent = ΔUSVᴴ, fdm)
                    is_cpu(A) && enz_test_pullbacks_match(rng, svd_full!, svd_full, A, USVᴴ, ΔUSVᴴ, alg)
                else
                    USVᴴ = MatrixAlgebraKit.initialize_output(svd_full!, A, alg)
                    is_cpu(A) && enz_test_pullbacks_match(rng, svd_full!, svd_full, A, USVᴴ, (nothing, nothing, nothing), alg; ȳ = ΔUSVᴴ)
                end
            end
        end
        @testset "svd_vals" begin
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                S, ΔS = ad_svd_vals_setup(A)
                if eltype(T) <: BlasFloat
                    test_reverse(svd_vals, RT, (A, TA); atol, rtol, fkwargs = (alg = alg,), output_tangent = ΔS, fdm)
                    is_cpu(A) && enz_test_pullbacks_match(rng, svd_vals!, svd_vals, A, S, ΔS, alg)
                else
                    S = MatrixAlgebraKit.initialize_output(svd_vals!, A, alg)
                    is_cpu(A) && enz_test_pullbacks_match(rng, svd_vals!, svd_vals, A, S, nothing, alg; ȳ = ΔS)
                end
            end
        end
        @testset "svd_trunc" begin
            S, ΔS = ad_svd_vals_setup(A)
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                for r in 1:4:minmn
                    truncalg = TruncatedAlgorithm(MatrixAlgebraKit.default_svd_algorithm(A), truncrank(r))
                    USVᴴ, _, ΔUSVᴴ, ΔUSVᴴtrunc = ad_svd_trunc_setup(A, truncalg)
                    if eltype(T) <: BlasFloat
                        test_reverse(svd_trunc_no_error, RT, (A, TA), (truncalg, Const); atol, rtol, output_tangent = ΔUSVᴴtrunc, fdm)
                        is_cpu(A) && enz_test_pullbacks_match(rng, svd_trunc_no_error!, svd_trunc_no_error, A, USVᴴ, ΔUSVᴴ, truncalg, ȳ = ΔUSVᴴtrunc)
                    else
                        is_cpu(A) && enz_test_pullbacks_match(rng, svd_trunc_no_error!, svd_trunc_no_error, A, (nothing, nothing, nothing), (nothing, nothing, nothing), truncalg, ȳ = ΔUSVᴴtrunc)
                    end
                end
                truncalg = TruncatedAlgorithm(MatrixAlgebraKit.default_svd_algorithm(A), trunctol(atol = S[1, 1] / 2))
                USVᴴ, _, ΔUSVᴴ, ΔUSVᴴtrunc = ad_svd_trunc_setup(A, truncalg)
                if eltype(T) <: BlasFloat
                    test_reverse(svd_trunc_no_error, RT, (A, TA), (truncalg, Const); atol, rtol, output_tangent = ΔUSVᴴtrunc, fdm)
                    is_cpu(A) && enz_test_pullbacks_match(rng, svd_trunc_no_error!, svd_trunc_no_error, A, USVᴴ, ΔUSVᴴ, truncalg, ȳ = ΔUSVᴴtrunc)
                else
                    is_cpu(A) && enz_test_pullbacks_match(rng, svd_trunc_no_error!, svd_trunc_no_error, A, (nothing, nothing, nothing), (nothing, nothing, nothing), truncalg, ȳ = ΔUSVᴴtrunc)
                end
            end
        end
    end
end

# GLA works with polar, but these tests
# segfault because of Sylvester + BigFloat
function test_enzyme_polar(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "Polar Enzyme AD rules $summary_str" begin
        A = instantiate_matrix(T, sz)
        m, n = size(A)
        alg = MatrixAlgebraKit.default_polar_algorithm(A)
        @testset "left_polar" begin
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                if m >= n
                    WP, ΔWP = ad_left_polar_setup(A)
                    eltype(T) <: BlasFloat && test_reverse(left_polar, RT, (A, TA), (alg, Const); atol, rtol)
                    is_cpu(A) && enz_test_pullbacks_match(rng, left_polar!, left_polar, A, WP, ΔWP, alg)
                end
            end
        end
        @testset "right_polar" begin
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                if m <= n
                    PWᴴ, ΔPWᴴ = ad_right_polar_setup(A)
                    eltype(T) <: BlasFloat && test_reverse(right_polar, RT, (A, TA), (alg, Const); atol, rtol)
                    is_cpu(A) && enz_test_pullbacks_match(rng, right_polar!, right_polar, A, PWᴴ, ΔPWᴴ, alg)
                end
            end
        end
    end
end

function test_enzyme_orthnull(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "Orthnull Enzyme AD rules $summary_str" begin
        A = instantiate_matrix(T, sz)
        m, n = size(A)
        VC, ΔVC = ad_left_orth_setup(A)
        CVᴴ, ΔCVᴴ = ad_right_orth_setup(A)
        fdm = eltype(T) <: Union{Float32, ComplexF32} ? EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1, max_range = 1.0e-2) : EnzymeTestUtils.FiniteDifferences.central_fdm(5, 1)
        @testset "left_orth" begin
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                @testset for alg in (:polar, :qr)
                    n > m && alg == :polar && continue
                    eltype(T) <: BlasFloat && test_reverse(left_orth, RT, (A, TA); atol, rtol, fkwargs = (alg = alg,), fdm)
                    left_orth_alg!(A, VC) = left_orth!(A, VC; alg = alg)
                    left_orth_alg(A) = left_orth(A; alg = alg)
                    is_cpu(A) && enz_test_pullbacks_match(rng, left_orth_alg!, left_orth_alg, A, VC, ΔVC)
                end
            end
        end
        N, ΔN = ad_left_null_setup(A)
        @testset "left_null" begin
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                left_null_qr!(A, N) = left_null!(A, N; alg = :qr)
                left_null_qr(A) = left_null(A; alg = :qr)
                eltype(T) <: BlasFloat && test_reverse(left_null_qr, RT, (A, TA); output_tangent = ΔN, atol, rtol)
                is_cpu(A) && enz_test_pullbacks_match(rng, left_null_qr!, left_null_qr, A, N, ΔN)
            end
        end
        @testset "right_orth" begin
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                @testset for alg in (:polar, :lq)
                    n < m && alg == :polar && continue
                    eltype(T) <: BlasFloat && test_reverse(right_orth, RT, (A, TA); atol, rtol, fkwargs = (alg = alg,), fdm)
                    right_orth_alg!(A, CVᴴ) = right_orth!(A, CVᴴ; alg = alg)
                    right_orth_alg(A) = right_orth(A; alg = alg)
                    is_cpu(A) && enz_test_pullbacks_match(rng, right_orth_alg!, right_orth_alg, A, CVᴴ, ΔCVᴴ)
                end
            end
        end
        Nᴴ, ΔNᴴ = ad_right_null_setup(A)
        @testset "right_null" begin
            @testset "reverse: RT $RT, TA $TA" for RT in (Duplicated,), TA in (Duplicated,)
                right_null_lq!(A, Nᴴ) = right_null!(A, Nᴴ; alg = :lq)
                right_null_lq(A) = right_null(A; alg = :lq)
                eltype(T) <: BlasFloat && test_reverse(right_null_lq, RT, (A, TA); output_tangent = ΔNᴴ, atol, rtol)
                is_cpu(A) && enz_test_pullbacks_match(rng, right_null_lq!, right_null_lq, A, Nᴴ, ΔNᴴ)
            end
        end
    end
end
