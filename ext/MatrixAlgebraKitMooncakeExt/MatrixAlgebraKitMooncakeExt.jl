module MatrixAlgebraKitMooncakeExt

using Mooncake
using Mooncake: DefaultCtx, CoDual, Dual, NoRData, rrule!!, frule!!, arrayify, @is_primitive
using MatrixAlgebraKit
using MatrixAlgebraKit: inv_safe, diagview, copy_input
using MatrixAlgebraKit: qr_pullback!, lq_pullback!
using MatrixAlgebraKit: qr_null_pullback!, lq_null_pullback!
using MatrixAlgebraKit: eig_pullback!, eigh_pullback!, eig_trunc_pullback!, eigh_trunc_pullback!
using MatrixAlgebraKit: left_polar_pullback!, right_polar_pullback!
using MatrixAlgebraKit: svd_pullback!, svd_trunc_pullback!
using LinearAlgebra


Mooncake.tangent_type(::Type{<:MatrixAlgebraKit.AbstractAlgorithm}) = Mooncake.NoTangent

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(copy_input), Any, Any}
function Mooncake.rrule!!(::CoDual{typeof(copy_input)}, f_df::CoDual, A_dA::CoDual)
    Ac = copy_input(Mooncake.primal(f_df), Mooncake.primal(A_dA))
    dAc = Mooncake.zero_tangent(Ac)
    function copy_input_pb(::NoRData)
        Mooncake.increment!!(Mooncake.tangent(A_dA), dAc)
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(Ac, dAc), copy_input_pb
end

# two-argument in-place factorizations like LQ, QR, EIG
for (f!, f, pb, adj) in (
        (qr_full!, qr_full, qr_pullback!, :qr_adjoint),
        (lq_full!, lq_full, lq_pullback!, :lq_adjoint),
        (qr_compact!, qr_compact, qr_pullback!, :qr_adjoint),
        (lq_compact!, lq_compact, lq_pullback!, :lq_adjoint),
        (eig_full!, eig_full, eig_pullback!, :eig_adjoint),
        (eigh_full!, eigh_full, eigh_pullback!, :eigh_adjoint),
        (left_polar!, left_polar, left_polar_pullback!, :left_polar_adjoint),
        (right_polar!, right_polar, right_polar_pullback!, :right_polar_adjoint),
    )

    @eval begin
        @is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof($f!), Any, Tuple{<:Any, <:Any}, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(::CoDual{typeof($f!)}, A_dA::CoDual, args_dargs::CoDual, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm})
            A, dA = arrayify(A_dA)
            args = Mooncake.primal(args_dargs)
            dargs = Mooncake.tangent(args_dargs)
            arg1, darg1 = arrayify(args[1], dargs[1])
            arg2, darg2 = arrayify(args[2], dargs[2])
            Ac = copy(A)
            arg1c = copy(arg1)
            arg2c = copy(arg2)
            $f!(A, args, Mooncake.primal(alg_dalg))
            function $adj(::NoRData)
                copy!(A, Ac)
                $pb(dA, A, (arg1, arg2), (darg1, darg2))
                copy!(arg1, arg1c)
                copy!(arg2, arg2c)
                MatrixAlgebraKit.zero!(darg1)
                MatrixAlgebraKit.zero!(darg2)
                return NoRData(), NoRData(), NoRData(), NoRData()
            end
            return args_dargs, $adj
        end
        @is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof($f), Any, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(::CoDual{typeof($f)}, A_dA::CoDual, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm})
            A, dA = arrayify(A_dA)
            output = $f(A, Mooncake.primal(alg_dalg))
            # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
            # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
            # pass). For many types this is done automatically when the forward step returns, but
            # not for nested structs with various fields (like Diagonal{Complex})
            output_codual = CoDual(output, Mooncake.fdata(Mooncake.zero_tangent(output)))
            function $adj(::NoRData)
                arg1, arg2 = Mooncake.primal(output_codual)
                darg1_, darg2_ = Mooncake.tangent(output_codual)
                arg1, darg1 = arrayify(arg1, darg1_)
                arg2, darg2 = arrayify(arg2, darg2_)
                $pb(dA, A, (arg1, arg2), (darg1, darg2))
                MatrixAlgebraKit.zero!(darg1)
                MatrixAlgebraKit.zero!(darg2)
                return NoRData(), NoRData(), NoRData()
            end
            return output_codual, $adj
        end
    end
end

for (f!, f, pb, adj) in (
        (qr_null!, qr_null, qr_null_pullback!, :qr_null_adjoint),
        (lq_null!, lq_null, lq_null_pullback!, :lq_null_adjoint),
    )
    @eval begin
        @is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof($f!), Any, Any, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(f_df::CoDual{typeof($f!)}, A_dA::CoDual, arg_darg::CoDual, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm})
            A, dA = arrayify(A_dA)
            Ac = copy(A)
            arg, darg = arrayify(arg_darg)
            argc = copy(arg)
            $f!(A, arg, Mooncake.primal(alg_dalg))
            function $adj(::NoRData)
                copy!(A, Ac)
                $pb(dA, A, arg, darg)
                copy!(arg, argc)
                MatrixAlgebraKit.zero!(darg)
                return NoRData(), NoRData(), NoRData(), NoRData()
            end
            return arg_darg, $adj
        end
        @is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof($f), Any, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(f_df::CoDual{typeof($f)}, A_dA::CoDual, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm})
            A, dA = arrayify(A_dA)
            output = $f(A, Mooncake.primal(alg_dalg))
            output_codual = CoDual(output, Mooncake.zero_tangent(output))
            function $adj(::NoRData)
                arg, darg = arrayify(output_codual)
                $pb(dA, A, arg, darg)
                MatrixAlgebraKit.zero!(darg)
                return NoRData(), NoRData(), NoRData()
            end
            return output_codual, $adj
        end
    end
end

for (f!, f, f_full, pb, adj) in (
        (eig_vals!, eig_vals, eig_full, eig_pullback!, :eig_vals_adjoint),
        (eigh_vals!, eigh_vals, eigh_full, eigh_pullback!, :eigh_vals_adjoint),
    )
    @eval begin
        @is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof($f!), Any, Any, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(::CoDual{typeof($f!)}, A_dA::CoDual, D_dD::CoDual, alg_dalg::CoDual)
            # compute primal
            A, dA = arrayify(A_dA)
            D, dD = arrayify(D_dD)
            # update primal
            DV = $f_full(A, Mooncake.primal(alg_dalg))
            copy!(D, diagview(DV[1]))
            V = DV[2]
            function $adj(::NoRData)
                $pb(dA, A, (Diagonal(D), V), (Diagonal(dD), nothing))
                MatrixAlgebraKit.zero!(dD)
                return NoRData(), NoRData(), NoRData(), NoRData()
            end
            return D_dD, $adj
        end
        @is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof($f), Any, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(::CoDual{typeof($f)}, A_dA::CoDual, alg_dalg::CoDual)
            # compute primal
            A, dA = arrayify(A_dA)
            # update primal
            DV = $f_full(A, Mooncake.primal(alg_dalg))
            V = DV[2]
            output = diagview(DV[1])
            output_codual = CoDual(output, Mooncake.zero_tangent(output))
            function $adj(::NoRData)
                D, dD = arrayify(output_codual)
                $pb(dA, A, (Diagonal(D), V), (Diagonal(dD), nothing))
                MatrixAlgebraKit.zero!(dD)
                return NoRData(), NoRData(), NoRData()
            end
            return output_codual, $adj
        end
    end
end

for (f, pb, adj) in (
        (eig_trunc, eig_trunc_pullback!, :eig_trunc_adjoint),
        (eigh_trunc, eigh_trunc_pullback!, :eigh_trunc_adjoint),
    )
    @eval begin
        @is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof($f), Any, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(::CoDual{typeof($f)}, A_dA::CoDual, alg_dalg::CoDual)
            # compute primal
            A, dA = arrayify(A_dA)
            alg = Mooncake.primal(alg_dalg)
            output = $f(A, alg)
            # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
            # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
            # pass). For many types this is done automatically when the forward step returns, but
            # not for nested structs with various fields (like Diagonal{Complex})
            output_codual = CoDual(output, Mooncake.fdata(Mooncake.zero_tangent(output)))
            function $adj(dy::Tuple{NoRData, NoRData, T}) where {T <: Real}
                Dtrunc, Vtrunc, ϵ = Mooncake.primal(output_codual)
                dDtrunc_, dVtrunc_, dϵ = Mooncake.tangent(output_codual)
                abs(dy[3]) > MatrixAlgebraKit.defaulttol(dy[3]) && @warn "Pullback for $f does not yet support non-zero tangent for the truncation error"
                D, dD = arrayify(Dtrunc, dDtrunc_)
                V, dV = arrayify(Vtrunc, dVtrunc_)
                $pb(dA, A, (D, V), (dD, dV))
                MatrixAlgebraKit.zero!(dD)
                MatrixAlgebraKit.zero!(dV)
                return NoRData(), NoRData(), NoRData()
            end
            return output_codual, $adj
        end
    end
end

for (f!, f) in (
        (svd_full!, svd_full),
        (svd_compact!, svd_compact),
    )
    @eval begin
        @is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof($f!), Any, Tuple{<:Any, <:Any, <:Any}, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(::CoDual{typeof($f!)}, A_dA::CoDual, USVᴴ_dUSVᴴ::CoDual, alg_dalg::CoDual)
            A, dA = arrayify(A_dA)
            Ac = copy(A)
            USVᴴ = Mooncake.primal(USVᴴ_dUSVᴴ)
            dUSVᴴ = Mooncake.tangent(USVᴴ_dUSVᴴ)
            U, dU = arrayify(USVᴴ[1], dUSVᴴ[1])
            S, dS = arrayify(USVᴴ[2], dUSVᴴ[2])
            Vᴴ, dVᴴ = arrayify(USVᴴ[3], dUSVᴴ[3])
            output = $f!(A, Mooncake.primal(alg_dalg))
            function svd_adjoint(::NoRData)
                copy!(A, Ac)
                if $(f! == svd_compact!)
                    svd_pullback!(dA, A, (U, S, Vᴴ), (dU, dS, dVᴴ))
                else # full
                    minmn = min(size(A)...)
                    vU = view(U, :, 1:minmn)
                    vS = Diagonal(diagview(S)[1:minmn])
                    vVᴴ = view(Vᴴ, 1:minmn, :)
                    vdU = view(dU, :, 1:minmn)
                    vdS = Diagonal(diagview(dS)[1:minmn])
                    vdVᴴ = view(dVᴴ, 1:minmn, :)
                    svd_pullback!(dA, A, (vU, vS, vVᴴ), (vdU, vdS, vdVᴴ))
                end
                MatrixAlgebraKit.zero!(dU)
                MatrixAlgebraKit.zero!(dS)
                MatrixAlgebraKit.zero!(dVᴴ)
                return NoRData(), NoRData(), NoRData(), NoRData()
            end
            return CoDual(output, dUSVᴴ), svd_adjoint
        end
        @is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof($f), Any, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(::CoDual{typeof($f)}, A_dA::CoDual, alg_dalg::CoDual)
            A, dA = arrayify(A_dA)
            USVᴴ = $f(A, Mooncake.primal(alg_dalg))
            # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
            # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
            # pass). For many types this is done automatically when the forward step returns, but
            # not for nested structs with various fields (like Diagonal{Complex})
            USVᴴ_codual = CoDual(USVᴴ, Mooncake.fdata(Mooncake.zero_tangent(USVᴴ)))
            function svd_adjoint(::NoRData)
                U, S, Vᴴ = Mooncake.primal(USVᴴ_codual)
                dU_, dS_, dVᴴ_ = Mooncake.tangent(USVᴴ_codual)
                U, dU = arrayify(U, dU_)
                S, dS = arrayify(S, dS_)
                Vᴴ, dVᴴ = arrayify(Vᴴ, dVᴴ_)
                if $(f == svd_compact)
                    svd_pullback!(dA, A, (U, S, Vᴴ), (dU, dS, dVᴴ))
                else # full
                    minmn = min(size(A)...)
                    vU = view(U, :, 1:minmn)
                    vS = Diagonal(view(diagview(S), 1:minmn))
                    vVᴴ = view(Vᴴ, 1:minmn, :)
                    vdU = view(dU, :, 1:minmn)
                    vdS = Diagonal(view(diagview(dS), 1:minmn))
                    vdVᴴ = view(dVᴴ, 1:minmn, :)
                    svd_pullback!(dA, A, (vU, vS, vVᴴ), (vdU, vdS, vdVᴴ))
                end
                MatrixAlgebraKit.zero!(dU)
                MatrixAlgebraKit.zero!(dS)
                MatrixAlgebraKit.zero!(dVᴴ)
                return NoRData(), NoRData(), NoRData()
            end
            return USVᴴ_codual, svd_adjoint
        end
    end
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(MatrixAlgebraKit.svd_vals!), Any, Any, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(::CoDual{typeof(MatrixAlgebraKit.svd_vals!)}, A_dA::CoDual, S_dS::CoDual, alg_dalg::CoDual)
    # compute primal
    A, dA = arrayify(A_dA)
    S, dS = arrayify(S_dS)
    U, nS, Vᴴ = svd_compact(A, Mooncake.primal(alg_dalg))
    copy!(S, diagview(nS))
    function svd_vals_adjoint(::NoRData)
        svd_pullback!(dA, A, (U, Diagonal(S), Vᴴ), (nothing, Diagonal(dS), nothing))
        MatrixAlgebraKit.zero!(dS)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return S_dS, svd_vals_adjoint
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(MatrixAlgebraKit.svd_vals), Any, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(::CoDual{typeof(MatrixAlgebraKit.svd_vals)}, A_dA::CoDual, alg_dalg::CoDual)
    # compute primal
    A, dA = arrayify(A_dA)
    U, S, Vᴴ = svd_compact(A, Mooncake.primal(alg_dalg))
    # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
    # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
    # pass). For many types this is done automatically when the forward step returns, but
    # not for nested structs with various fields (like Diagonal{Complex})
    S_codual = CoDual(diagview(S), Mooncake.fdata(Mooncake.zero_tangent(diagview(S))))
    function svd_vals_adjoint(::NoRData)
        S, dS = arrayify(S_codual)
        svd_pullback!(dA, A, (U, Diagonal(S), Vᴴ), (nothing, Diagonal(dS), nothing))
        MatrixAlgebraKit.zero!(dS)
        return NoRData(), NoRData(), NoRData()
    end
    return S_codual, svd_vals_adjoint
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(MatrixAlgebraKit.svd_trunc), Any, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(::CoDual{typeof(MatrixAlgebraKit.svd_trunc)}, A_dA::CoDual, alg_dalg::CoDual)
    # compute primal
    A_ = Mooncake.primal(A_dA)
    dA_ = Mooncake.tangent(A_dA)
    A, dA = arrayify(A_, dA_)
    alg = Mooncake.primal(alg_dalg)
    output = svd_trunc(A, alg)
    # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
    # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
    # pass). For many types this is done automatically when the forward step returns, but
    # not for nested structs with various fields (like Diagonal{Complex})
    output_codual = CoDual(output, Mooncake.fdata(Mooncake.zero_tangent(output)))
    function svd_trunc_adjoint(dy::Tuple{NoRData, NoRData, NoRData, T}) where {T <: Real}
        Utrunc, Strunc, Vᴴtrunc, ϵ = Mooncake.primal(output_codual)
        dUtrunc_, dStrunc_, dVᴴtrunc_, dϵ = Mooncake.tangent(output_codual)
        abs(dy[4]) > MatrixAlgebraKit.defaulttol(dy[4]) && @warn "Pullback for svd_trunc! does not yet support non-zero tangent for the truncation error"
        U, dU = arrayify(Utrunc, dUtrunc_)
        S, dS = arrayify(Strunc, dStrunc_)
        Vᴴ, dVᴴ = arrayify(Vᴴtrunc, dVᴴtrunc_)
        svd_trunc_pullback!(dA, A, (U, S, Vᴴ), (dU, dS, dVᴴ))
        MatrixAlgebraKit.zero!(dU)
        MatrixAlgebraKit.zero!(dS)
        MatrixAlgebraKit.zero!(dVᴴ)
        return NoRData(), NoRData(), NoRData()
    end
    return output_codual, svd_trunc_adjoint
end

end
