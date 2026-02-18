module MatrixAlgebraKitMooncakeExt

using Mooncake
using Mooncake: DefaultCtx, CoDual, Dual, NoRData, rrule!!, frule!!, arrayify, @is_primitive
using MatrixAlgebraKit
using MatrixAlgebraKit: inv_safe, diagview, copy_input, initialize_output, zero!
using MatrixAlgebraKit: qr_pullback!, lq_pullback!
using MatrixAlgebraKit: qr_null_pullback!, lq_null_pullback!
using MatrixAlgebraKit: eig_pullback!, eigh_pullback!, eig_vals_pullback!
using MatrixAlgebraKit: eig_trunc_pullback!, eigh_trunc_pullback!, eigh_vals_pullback!
using MatrixAlgebraKit: left_polar_pullback!, right_polar_pullback!
using MatrixAlgebraKit: svd_pullback!, svd_trunc_pullback!, svd_vals_pullback!
using MatrixAlgebraKit: TruncatedAlgorithm
using LinearAlgebra

Mooncake.tangent_type(::Type{<:MatrixAlgebraKit.AbstractAlgorithm}) = Mooncake.NoTangent

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(copy_input), Any, Any}
function Mooncake.rrule!!(::CoDual{typeof(copy_input)}, f_df::CoDual, A_dA::CoDual)
    Ac = copy_input(Mooncake.primal(f_df), Mooncake.primal(A_dA))
    Ac_dAc = Mooncake.zero_fcodual(Ac)
    dAc = Mooncake.tangent(Ac_dAc)
    function copy_input_pb(::NoRData)
        Mooncake.increment!!(Mooncake.tangent(A_dA), dAc)
        return NoRData(), NoRData(), NoRData()
    end
    return Ac_dAc, copy_input_pb
end

Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(initialize_output), Any, Any, Any}
# two-argument in-place factorizations like LQ, QR, EIG
for (f!, f, pb, adj) in (
        (:qr_full!, :qr_full, :qr_pullback!, :qr_adjoint),
        (:lq_full!, :lq_full, :lq_pullback!, :lq_adjoint),
        (:qr_compact!, :qr_compact, :qr_pullback!, :qr_adjoint),
        (:lq_compact!, :lq_compact, :lq_pullback!, :lq_adjoint),
        (:eig_full!, :eig_full, :eig_pullback!, :eig_adjoint),
        (:eigh_full!, :eigh_full, :eigh_pullback!, :eigh_adjoint),
        (:left_polar!, :left_polar, :left_polar_pullback!, :left_polar_adjoint),
        (:right_polar!, :right_polar, :right_polar_pullback!, :right_polar_adjoint),
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
                zero!(darg1)
                zero!(darg2)
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
                zero!(darg1)
                zero!(darg2)
                return NoRData(), NoRData(), NoRData()
            end
            return output_codual, $adj
        end
    end
end

for (f!, f, pb, adj) in (
        (:qr_null!, :qr_null, :qr_null_pullback!, :qr_null_adjoint),
        (:lq_null!, :lq_null, :lq_null_pullback!, :lq_null_adjoint),
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
                zero!(darg)
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
                zero!(darg)
                return NoRData(), NoRData(), NoRData()
            end
            return output_codual, $adj
        end
    end
end

for (f!, f, f_full, pb, adj) in (
        (:eig_vals!, :eig_vals, :eig_full, :eig_vals_pullback!, :eig_vals_adjoint),
        (:eigh_vals!, :eigh_vals, :eigh_full, :eigh_vals_pullback!, :eigh_vals_adjoint),
    )
    @eval begin
        @is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof($f!), Any, Any, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(::CoDual{typeof($f!)}, A_dA::CoDual, D_dD::CoDual, alg_dalg::CoDual)
            # compute primal
            A, dA = arrayify(A_dA)
            D, dD = arrayify(D_dD)
            Dc = copy(D)
            # update primal
            DV = $f_full(A, Mooncake.primal(alg_dalg))
            copy!(D, diagview(DV[1]))
            V = DV[2]
            function $adj(::NoRData)
                $pb(dA, A, DV, dD)
                copy!(D, Dc)
                zero!(dD)
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
                $pb(dA, A, DV, dD)
                zero!(dD)
                return NoRData(), NoRData(), NoRData()
            end
            return output_codual, $adj
        end
    end
end

_warn_pullback_truncerror(dϵ::Real; tol = MatrixAlgebraKit.defaulttol(dϵ)) =
    abs(dϵ) ≤ tol || @warn "Pullback ignores non-zero tangents for truncation error"

for f in (:eig, :eigh)
    f_trunc = Symbol(f, :_trunc)
    f_trunc! = Symbol(f_trunc, :!)
    f_full = Symbol(f, :_full)
    f_full! = Symbol(f_full, :!)
    f_pullback! = Symbol(f, :_pullback!)
    f_trunc_pullback! = Symbol(f_trunc, :_pullback!)
    f_adjoint! = Symbol(f, :_adjoint!)
    f_trunc_no_error = Symbol(f_trunc, :_no_error)
    f_trunc_no_error! = Symbol(f_trunc_no_error, :!)

    @eval begin
        @is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof($f_trunc!), Any, Any, MatrixAlgebraKit.AbstractAlgorithm}
        @is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof($f_trunc), Any, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(::CoDual{typeof($f_trunc!)}, A_dA::CoDual, DV_dDV::CoDual, alg_dalg::CoDual)
            # compute primal
            A, dA = arrayify(A_dA)
            DV = Mooncake.primal(DV_dDV)
            dDV = Mooncake.tangent(DV_dDV)
            Ac = copy(A)
            DVc = copy.(DV)
            alg = Mooncake.primal(alg_dalg)
            output = $f_trunc!(A, DV, alg)
            # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
            # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
            # pass). For many types this is done automatically when the forward step returns, but
            # not for nested structs with various fields (like Diagonal{Complex})
            output_codual = Mooncake.zero_fcodual(output)
            function $f_adjoint!(dy::Tuple{NoRData, NoRData, <:Real})
                copy!(A, Ac)
                Dtrunc, Vtrunc, ϵ = Mooncake.primal(output_codual)
                dDtrunc_, dVtrunc_, dϵ = Mooncake.tangent(output_codual)
                _warn_pullback_truncerror(dy[3])
                D′, dD′ = arrayify(Dtrunc, dDtrunc_)
                V′, dV′ = arrayify(Vtrunc, dVtrunc_)
                $f_trunc_pullback!(dA, A, (D′, V′), (dD′, dV′))
                copy!(DV[1], DVc[1])
                copy!(DV[2], DVc[2])
                zero!(dD′)
                zero!(dV′)
                return NoRData(), NoRData(), NoRData(), NoRData()
            end
            return output_codual, $f_adjoint!
        end
        function Mooncake.rrule!!(::CoDual{typeof($f_trunc!)}, A_dA::CoDual, DV_dDV::CoDual, alg_dalg::CoDual{<:TruncatedAlgorithm})
            # unpack variables
            A, dA = arrayify(A_dA)
            DV_dDV_arr = arrayify.(Mooncake.primal(DV_dDV), Mooncake.tangent(DV_dDV))
            DV, dDV = first.(DV_dDV_arr), last.(DV_dDV_arr)
            alg = Mooncake.primal(alg_dalg)

            # store state prior to primal call
            Ac = copy(A)
            DVc = copy.(DV)

            # compute primal - capture full DV and ind
            DV = $f_full!(A, DV, alg.alg)
            DVtrunc, ind = MatrixAlgebraKit.truncate($f_trunc!, DV, alg.trunc)
            ϵ = MatrixAlgebraKit.truncation_error(diagview(DV[1]), ind)

            # pack output - note that we allocate new dDVtrunc because these aren't overwritten in the input
            DVtrunc_dDVtrunc = Mooncake.zero_fcodual((DVtrunc..., ϵ))

            # define pullback
            dDVtrunc = last.(arrayify.(DVtrunc, Base.front(Mooncake.tangent(DVtrunc_dDVtrunc))))
            function $f_adjoint!((_, _, dϵ)::Tuple{NoRData, NoRData, Real})
                _warn_pullback_truncerror(dϵ)

                # compute pullbacks
                $f_pullback!(dA, Ac, DV, dDVtrunc, ind)
                zero!.(dDVtrunc) # since this is allocated in this function this is probably not required

                # restore state
                copy!(A, Ac)
                copy!.(DV, DVc)

                return ntuple(Returns(NoRData()), 4)
            end

            return DVtrunc_dDVtrunc, $f_adjoint!
        end
        function Mooncake.rrule!!(::CoDual{typeof($f_trunc)}, A_dA::CoDual, alg_dalg::CoDual)
            # compute primal
            A, dA = arrayify(A_dA)
            alg = Mooncake.primal(alg_dalg)
            output = $f_trunc(A, alg)
            # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
            # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
            # pass). For many types this is done automatically when the forward step returns, but
            # not for nested structs with various fields (like Diagonal{Complex})
            output_codual = CoDual(output, Mooncake.fdata(Mooncake.zero_tangent(output)))
            function $f_adjoint!(dy::Tuple{NoRData, NoRData, T}) where {T <: Real}
                Dtrunc, Vtrunc, ϵ = Mooncake.primal(output_codual)
                dDtrunc_, dVtrunc_, dϵ = Mooncake.tangent(output_codual)
                _warn_pullback_truncerror(dy[3])
                D, dD = arrayify(Dtrunc, dDtrunc_)
                V, dV = arrayify(Vtrunc, dVtrunc_)
                $f_trunc_pullback!(dA, A, (D, V), (dD, dV))
                zero!(dD)
                zero!(dV)
                return NoRData(), NoRData(), NoRData()
            end
            return output_codual, $f_adjoint!
        end
        function Mooncake.rrule!!(::CoDual{typeof($f_trunc)}, A_dA::CoDual, alg_dalg::CoDual{<:TruncatedAlgorithm})
            # unpack variables
            A, dA = arrayify(A_dA)
            alg = Mooncake.primal(alg_dalg)

            # compute primal - capture full DV and ind
            DV = $f_full(A, alg.alg)
            DVtrunc, ind = MatrixAlgebraKit.truncate($f_trunc!, DV, alg.trunc)
            ϵ = MatrixAlgebraKit.truncation_error(diagview(DV[1]), ind)

            # pack output
            DVtrunc_dDVtrunc = Mooncake.zero_fcodual((DVtrunc..., ϵ))

            # define pullback
            dDVtrunc = last.(arrayify.(DVtrunc, Base.front(Mooncake.tangent(DVtrunc_dDVtrunc))))
            function $f_adjoint!((_, _, dϵ)::Tuple{NoRData, NoRData, Real})
                _warn_pullback_truncerror(dϵ)
                $f_pullback!(dA, A, DV, dDVtrunc, ind)
                zero!.(dDVtrunc) # since this is allocated in this function this is probably not required
                return ntuple(Returns(NoRData()), 3)
            end

            return DVtrunc_dDVtrunc, $f_adjoint!
        end
        @is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof($f_trunc_no_error!), Any, Any, MatrixAlgebraKit.AbstractAlgorithm}
        @is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof($f_trunc_no_error), Any, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(::CoDual{typeof($f_trunc_no_error!)}, A_dA::CoDual, DV_dDV::CoDual, alg_dalg::CoDual)
            # compute primal
            A, dA = arrayify(A_dA)
            alg = Mooncake.primal(alg_dalg)
            DV = Mooncake.primal(DV_dDV)
            dDV = Mooncake.tangent(DV_dDV)
            Ac = copy(A)
            DVc = copy.(DV)
            output = $f_trunc_no_error!(A, DV, alg)
            # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
            # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
            # pass). For many types this is done automatically when the forward step returns, but
            # not for nested structs with various fields (like Diagonal{Complex})
            output_codual = CoDual(output, Mooncake.fdata(Mooncake.zero_tangent(output)))
            function $f_adjoint!(::NoRData)
                copy!(A, Ac)
                Dtrunc, Vtrunc = Mooncake.primal(output_codual)
                dDtrunc_, dVtrunc_ = Mooncake.tangent(output_codual)
                D′, dD′ = arrayify(Dtrunc, dDtrunc_)
                V′, dV′ = arrayify(Vtrunc, dVtrunc_)
                $f_pullback!(dA, A, (D′, V′), (dD′, dV′))
                copy!(DV[1], DVc[1])
                copy!(DV[2], DVc[2])
                zero!(dD′)
                zero!(dV′)
                return NoRData(), NoRData(), NoRData(), NoRData()
            end
            return output_codual, $f_adjoint!
        end
        function Mooncake.rrule!!(::CoDual{typeof($f_trunc_no_error!)}, A_dA::CoDual, DV_dDV::CoDual, alg_dalg::CoDual{<:TruncatedAlgorithm})
            # unpack variables
            A, dA = arrayify(A_dA)
            DV_dDV_arr = arrayify.(Mooncake.primal(DV_dDV), Mooncake.tangent(DV_dDV))
            DV, dDV = first.(DV_dDV_arr), last.(DV_dDV_arr)
            alg = Mooncake.primal(alg_dalg)

            # store state prior to primal call
            Ac = copy(A)
            DVc = copy.(DV)

            # compute primal - capture full DV and ind
            DV = $f_full!(A, DV, alg.alg)
            DVtrunc, ind = MatrixAlgebraKit.truncate($f_trunc!, DV, alg.trunc)

            # pack output - note that we allocate new dDVtrunc because these aren't overwritten in the input
            DVtrunc_dDVtrunc = Mooncake.zero_fcodual(DVtrunc)

            # define pullback
            dDVtrunc = last.(arrayify.(DVtrunc, Mooncake.tangent(DVtrunc_dDVtrunc)))
            function $f_adjoint!(::NoRData)
                # compute pullbacks
                $f_pullback!(dA, Ac, DV, dDVtrunc, ind)
                zero!.(dDV)

                # restore state
                copy!(A, Ac)
                copy!.(DV, DVc)

                return ntuple(Returns(NoRData()), 4)
            end

            return DVtrunc_dDVtrunc, $f_adjoint!
        end
        function Mooncake.rrule!!(::CoDual{typeof($f_trunc_no_error)}, A_dA::CoDual, alg_dalg::CoDual)
            # compute primal
            A, dA = arrayify(A_dA)
            alg = Mooncake.primal(alg_dalg)
            output = $f_trunc_no_error(A, alg)
            # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
            # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
            # pass). For many types this is done automatically when the forward step returns, but
            # not for nested structs with various fields (like Diagonal{Complex})
            output_codual = CoDual(output, Mooncake.fdata(Mooncake.zero_tangent(output)))
            function $f_adjoint!(::NoRData)
                Dtrunc, Vtrunc = Mooncake.primal(output_codual)
                dDtrunc_, dVtrunc_ = Mooncake.tangent(output_codual)
                D, dD = arrayify(Dtrunc, dDtrunc_)
                V, dV = arrayify(Vtrunc, dVtrunc_)
                $f_trunc_pullback!(dA, A, (D, V), (dD, dV))
                zero!(dD)
                zero!(dV)
                return NoRData(), NoRData(), NoRData()
            end
            return output_codual, $f_adjoint!
        end
        function Mooncake.rrule!!(::CoDual{typeof($f_trunc_no_error)}, A_dA::CoDual, alg_dalg::CoDual{<:TruncatedAlgorithm})
            # unpack variables
            A, dA = arrayify(A_dA)
            alg = Mooncake.primal(alg_dalg)

            # compute primal - capture full DV and ind
            DV = $f_full(A, alg.alg)
            DVtrunc, ind = MatrixAlgebraKit.truncate($f_trunc!, DV, alg.trunc)

            # pack output
            DVtrunc_dDVtrunc = Mooncake.zero_fcodual(DVtrunc)

            # define pullback
            dDVtrunc = last.(arrayify.(DVtrunc, Mooncake.tangent(DVtrunc_dDVtrunc)))
            function $f_adjoint!(::NoRData)
                $f_pullback!(dA, A, DV, dDVtrunc, ind)
                zero!.(dDVtrunc) # since this is allocated in this function this is probably not required
                return ntuple(Returns(NoRData()), 3)
            end

            return DVtrunc_dDVtrunc, $f_adjoint!
        end
    end
end

for (f!, f) in (
        (:svd_full!, :svd_full),
        (:svd_compact!, :svd_compact),
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
            USVᴴc = copy.(USVᴴ)
            output = $f!(A, USVᴴ, Mooncake.primal(alg_dalg))
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
                copy!(U, USVᴴc[1])
                copy!(S, USVᴴc[2])
                copy!(Vᴴ, USVᴴc[3])
                zero!(dU)
                zero!(dS)
                zero!(dVᴴ)
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
                zero!(dU)
                zero!(dS)
                zero!(dVᴴ)
                return NoRData(), NoRData(), NoRData()
            end
            return USVᴴ_codual, svd_adjoint
        end
    end
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(svd_vals!), Any, Any, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(::CoDual{typeof(svd_vals!)}, A_dA::CoDual, S_dS::CoDual, alg_dalg::CoDual)
    # compute primal
    A, dA = arrayify(A_dA)
    S, dS = arrayify(S_dS)
    Sc = copy(S)
    USVᴴ = svd_compact(A, Mooncake.primal(alg_dalg))
    copy!(S, diagview(USVᴴ[2]))
    function svd_vals_adjoint(::NoRData)
        svd_vals_pullback!(dA, A, USVᴴ, dS)
        zero!(dS)
        copy!(S, Sc)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return S_dS, svd_vals_adjoint
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(svd_vals), Any, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(::CoDual{typeof(svd_vals)}, A_dA::CoDual, alg_dalg::CoDual)
    # compute primal
    A, dA = arrayify(A_dA)
    USVᴴ = svd_compact(A, Mooncake.primal(alg_dalg))
    # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
    # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
    # pass). For many types this is done automatically when the forward step returns, but
    # not for nested structs with various fields (like Diagonal{Complex})
    S = diagview(USVᴴ[2])
    S_codual = CoDual(S, Mooncake.fdata(Mooncake.zero_tangent(S)))
    function svd_vals_adjoint(::NoRData)
        S, dS = arrayify(S_codual)
        svd_vals_pullback!(dA, A, USVᴴ, dS)
        zero!(dS)
        return NoRData(), NoRData(), NoRData()
    end
    return S_codual, svd_vals_adjoint
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(svd_trunc!), Any, Any, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(::CoDual{typeof(svd_trunc!)}, A_dA::CoDual, USVᴴ_dUSVᴴ::CoDual, alg_dalg::CoDual)
    # compute primal
    A, dA = arrayify(A_dA)
    alg = Mooncake.primal(alg_dalg)
    Ac = copy(A)
    USVᴴ = Mooncake.primal(USVᴴ_dUSVᴴ)
    dUSVᴴ = Mooncake.tangent(USVᴴ_dUSVᴴ)
    U, dU = arrayify(USVᴴ[1], dUSVᴴ[1])
    S, dS = arrayify(USVᴴ[2], dUSVᴴ[2])
    Vᴴ, dVᴴ = arrayify(USVᴴ[3], dUSVᴴ[3])
    USVᴴc = copy.(USVᴴ)
    output = svd_trunc!(A, USVᴴ, alg)
    # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
    # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
    # pass). For many types this is done automatically when the forward step returns, but
    # not for nested structs with various fields (like Diagonal{Complex})
    output_codual = Mooncake.zero_fcodual(output)
    function svd_trunc_adjoint(dy::Tuple{NoRData, NoRData, NoRData, T}) where {T <: Real}
        copy!(A, Ac)
        Utrunc, Strunc, Vᴴtrunc, ϵ = Mooncake.primal(output_codual)
        dUtrunc_, dStrunc_, dVᴴtrunc_, dϵ = Mooncake.tangent(output_codual)
        _warn_pullback_truncerror(dy[4])
        U′, dU′ = arrayify(Utrunc, dUtrunc_)
        S′, dS′ = arrayify(Strunc, dStrunc_)
        Vᴴ′, dVᴴ′ = arrayify(Vᴴtrunc, dVᴴtrunc_)
        svd_trunc_pullback!(dA, A, (U′, S′, Vᴴ′), (dU′, dS′, dVᴴ′))
        copy!(U, USVᴴc[1])
        copy!(S, USVᴴc[2])
        copy!(Vᴴ, USVᴴc[3])
        zero!(dU)
        zero!(dS)
        zero!(dVᴴ)
        zero!(dU′)
        zero!(dS′)
        zero!(dVᴴ′)
        return NoRData(), NoRData(), NoRData()
    end
    return output_codual, svd_trunc_adjoint
end
function Mooncake.rrule!!(::CoDual{typeof(svd_trunc!)}, A_dA::CoDual, USVᴴ_dUSVᴴ::CoDual, alg_dalg::CoDual{<:TruncatedAlgorithm})
    # unpack variables
    A, dA = arrayify(A_dA)
    USVᴴ_dUSVᴴ_arr = arrayify.(Mooncake.primal(USVᴴ_dUSVᴴ), Mooncake.tangent(USVᴴ_dUSVᴴ))
    USVᴴ, dUSVᴴ = first.(USVᴴ_dUSVᴴ_arr), last.(USVᴴ_dUSVᴴ_arr)
    alg = Mooncake.primal(alg_dalg)

    # store state prior to primal call
    Ac = copy(A)
    USVᴴc = copy.(USVᴴ)

    # compute primal - capture full USVᴴ and ind
    USVᴴ = svd_compact!(A, USVᴴ, alg.alg)
    USVᴴtrunc, ind = MatrixAlgebraKit.truncate(svd_trunc!, USVᴴ, alg.trunc)
    ϵ = MatrixAlgebraKit.truncation_error(diagview(USVᴴ[2]), ind)

    # pack output - note that we allocate new dUSVᴴtrunc because these aren't actually
    # overwritten in the input!
    USVᴴtrunc_dUSVᴴtrunc = Mooncake.zero_fcodual((USVᴴtrunc..., ϵ))

    # define pullback
    dUSVᴴtrunc = last.(arrayify.(USVᴴtrunc, Base.front(Mooncake.tangent(USVᴴtrunc_dUSVᴴtrunc))))
    function svd_trunc_adjoint((_, _, _, dϵ)::Tuple{NoRData, NoRData, NoRData, Real})
        _warn_pullback_truncerror(dϵ)

        # compute pullbacks
        svd_pullback!(dA, Ac, USVᴴ, dUSVᴴtrunc, ind)
        zero!.(dUSVᴴtrunc) # since this is allocated in this function this is probably not required
        zero!.(dUSVᴴ)

        # restore state
        copy!(A, Ac)
        copy!.(USVᴴ, USVᴴc)

        return ntuple(Returns(NoRData()), 4)
    end

    return USVᴴtrunc_dUSVᴴtrunc, svd_trunc_adjoint
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(svd_trunc), Any, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(::CoDual{typeof(svd_trunc)}, A_dA::CoDual, alg_dalg::CoDual)
    # compute primal
    A, dA = arrayify(A_dA)
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
        _warn_pullback_truncerror(dy[4])
        U, dU = arrayify(Utrunc, dUtrunc_)
        S, dS = arrayify(Strunc, dStrunc_)
        Vᴴ, dVᴴ = arrayify(Vᴴtrunc, dVᴴtrunc_)
        svd_trunc_pullback!(dA, A, (U, S, Vᴴ), (dU, dS, dVᴴ))
        zero!(dU)
        zero!(dS)
        zero!(dVᴴ)
        return NoRData(), NoRData(), NoRData()
    end
    return output_codual, svd_trunc_adjoint
end
function Mooncake.rrule!!(::CoDual{typeof(svd_trunc)}, A_dA::CoDual, alg_dalg::CoDual{<:TruncatedAlgorithm})
    # unpack variables
    A, dA = arrayify(A_dA)
    alg = Mooncake.primal(alg_dalg)

    # compute primal - capture full USVᴴ and ind
    USVᴴ = svd_compact(A, alg.alg)
    USVᴴtrunc, ind = MatrixAlgebraKit.truncate(svd_trunc!, USVᴴ, alg.trunc)
    ϵ = MatrixAlgebraKit.truncation_error(diagview(USVᴴ[2]), ind)

    # pack output
    USVᴴtrunc_dUSVᴴtrunc = Mooncake.zero_fcodual((USVᴴtrunc..., ϵ))

    # define pullback
    dUSVᴴtrunc = last.(arrayify.(USVᴴtrunc, Base.front(Mooncake.tangent(USVᴴtrunc_dUSVᴴtrunc))))
    function svd_trunc_adjoint((_, _, _, dϵ)::Tuple{NoRData, NoRData, NoRData, Real})
        _warn_pullback_truncerror(dϵ)
        svd_pullback!(dA, A, USVᴴ, dUSVᴴtrunc, ind)
        zero!.(dUSVᴴtrunc) # since this is allocated in this function this is probably not required
        return ntuple(Returns(NoRData()), 3)
    end

    return USVᴴtrunc_dUSVᴴtrunc, svd_trunc_adjoint
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(svd_trunc_no_error!), Any, Any, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(::CoDual{typeof(svd_trunc_no_error!)}, A_dA::CoDual, USVᴴ_dUSVᴴ::CoDual, alg_dalg::CoDual)
    # compute primal
    A, dA = arrayify(A_dA)
    alg = Mooncake.primal(alg_dalg)
    Ac = copy(A)
    USVᴴ = Mooncake.primal(USVᴴ_dUSVᴴ)
    dUSVᴴ = Mooncake.tangent(USVᴴ_dUSVᴴ)
    U, dU = arrayify(USVᴴ[1], dUSVᴴ[1])
    S, dS = arrayify(USVᴴ[2], dUSVᴴ[2])
    Vᴴ, dVᴴ = arrayify(USVᴴ[3], dUSVᴴ[3])
    USVᴴc = copy.(USVᴴ)
    output = svd_trunc_no_error!(A, USVᴴ, alg)
    # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
    # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
    # pass). For many types this is done automatically when the forward step returns, but
    # not for nested structs with various fields (like Diagonal{Complex})
    output_codual = CoDual(output, Mooncake.fdata(Mooncake.zero_tangent(output)))
    function svd_trunc_adjoint(::NoRData)
        copy!(A, Ac)
        Utrunc, Strunc, Vᴴtrunc = Mooncake.primal(output_codual)
        dUtrunc_, dStrunc_, dVᴴtrunc_ = Mooncake.tangent(output_codual)
        U′, dU′ = arrayify(Utrunc, dUtrunc_)
        S′, dS′ = arrayify(Strunc, dStrunc_)
        Vᴴ′, dVᴴ′ = arrayify(Vᴴtrunc, dVᴴtrunc_)
        svd_trunc_pullback!(dA, A, (U′, S′, Vᴴ′), (dU′, dS′, dVᴴ′))
        copy!(U, USVᴴc[1])
        copy!(S, USVᴴc[2])
        copy!(Vᴴ, USVᴴc[3])
        zero!(dU)
        zero!(dS)
        zero!(dVᴴ)
        zero!(dU′)
        zero!(dS′)
        zero!(dVᴴ′)
        return NoRData(), NoRData(), NoRData()
    end
    return output_codual, svd_trunc_adjoint
end
function Mooncake.rrule!!(::CoDual{typeof(svd_trunc_no_error!)}, A_dA::CoDual, USVᴴ_dUSVᴴ::CoDual, alg_dalg::CoDual{<:TruncatedAlgorithm})
    # unpack variables
    A, dA = arrayify(A_dA)
    USVᴴ_dUSVᴴ_arr = arrayify.(Mooncake.primal(USVᴴ_dUSVᴴ), Mooncake.tangent(USVᴴ_dUSVᴴ))
    USVᴴ, dUSVᴴ = first.(USVᴴ_dUSVᴴ_arr), last.(USVᴴ_dUSVᴴ_arr)
    alg = Mooncake.primal(alg_dalg)

    # store state prior to primal call
    Ac = copy(A)
    USVᴴc = copy.(USVᴴ)

    # compute primal - capture full USVᴴ and ind
    USVᴴ = svd_compact!(A, USVᴴ, alg.alg)
    USVᴴtrunc, ind = MatrixAlgebraKit.truncate(svd_trunc!, USVᴴ, alg.trunc)

    # pack output - note that we allocate new dUSVᴴtrunc because these aren't actually
    # overwritten in the input!
    USVᴴtrunc_dUSVᴴtrunc = Mooncake.zero_fcodual(USVᴴtrunc)

    # define pullback
    dUSVᴴtrunc = last.(arrayify.(USVᴴtrunc, Mooncake.tangent(USVᴴtrunc_dUSVᴴtrunc)))
    function svd_trunc_adjoint(::NoRData)
        # compute pullbacks
        svd_pullback!(dA, Ac, USVᴴ, dUSVᴴtrunc, ind)
        zero!.(dUSVᴴ)

        # restore state
        copy!(A, Ac)
        copy!.(USVᴴ, USVᴴc)

        return ntuple(Returns(NoRData()), 4)
    end

    return USVᴴtrunc_dUSVᴴtrunc, svd_trunc_adjoint
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(svd_trunc_no_error), Any, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(::CoDual{typeof(svd_trunc_no_error)}, A_dA::CoDual, alg_dalg::CoDual)
    # compute primal
    A, dA = arrayify(A_dA)
    alg = Mooncake.primal(alg_dalg)
    output = svd_trunc_no_error(A, alg)
    # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
    # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
    # pass). For many types this is done automatically when the forward step returns, but
    # not for nested structs with various fields (like Diagonal{Complex})
    output_codual = CoDual(output, Mooncake.fdata(Mooncake.zero_tangent(output)))
    function svd_trunc_adjoint(::NoRData)
        Utrunc, Strunc, Vᴴtrunc = Mooncake.primal(output_codual)
        dUtrunc_, dStrunc_, dVᴴtrunc_ = Mooncake.tangent(output_codual)
        U, dU = arrayify(Utrunc, dUtrunc_)
        S, dS = arrayify(Strunc, dStrunc_)
        Vᴴ, dVᴴ = arrayify(Vᴴtrunc, dVᴴtrunc_)
        svd_trunc_pullback!(dA, A, (U, S, Vᴴ), (dU, dS, dVᴴ))
        zero!(dU)
        zero!(dS)
        zero!(dVᴴ)
        return NoRData(), NoRData(), NoRData()
    end
    return output_codual, svd_trunc_adjoint
end
function Mooncake.rrule!!(::CoDual{typeof(svd_trunc_no_error)}, A_dA::CoDual, alg_dalg::CoDual{<:TruncatedAlgorithm})
    # unpack variables
    A, dA = arrayify(A_dA)
    alg = Mooncake.primal(alg_dalg)

    # compute primal - capture full USVᴴ and ind
    USVᴴ = svd_compact(A, alg.alg)
    USVᴴtrunc, ind = MatrixAlgebraKit.truncate(svd_trunc!, USVᴴ, alg.trunc)

    # pack output
    USVᴴtrunc_dUSVᴴtrunc = Mooncake.zero_fcodual(USVᴴtrunc)

    # define pullback
    dUSVᴴtrunc = last.(arrayify.(USVᴴtrunc, Mooncake.tangent(USVᴴtrunc_dUSVᴴtrunc)))
    function svd_trunc_adjoint(::NoRData)
        svd_pullback!(dA, A, USVᴴ, dUSVᴴtrunc, ind)
        zero!.(dUSVᴴtrunc) # since this is allocated in this function this is probably not required
        return ntuple(Returns(NoRData()), 3)
    end

    return USVᴴtrunc_dUSVᴴtrunc, svd_trunc_adjoint
end

# single-output projections: project_hermitian!, project_antihermitian!
for (f!, f, adj) in (
        (:project_hermitian!, :project_hermitian, :project_hermitian_adjoint),
        (:project_antihermitian!, :project_antihermitian, :project_antihermitian_adjoint),
    )
    @eval begin
        @is_primitive DefaultCtx Mooncake.ReverseMode Tuple{typeof($f!), Any, Any, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(f_df::CoDual{typeof($f!)}, A_dA::CoDual, arg_darg::CoDual, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm})
            A, dA = arrayify(A_dA)
            arg, darg = A_dA === arg_darg ? (A, dA) : arrayify(arg_darg)
            argc = copy(arg)
            arg = $f!(A, arg, Mooncake.primal(alg_dalg))

            function $adj(::NoRData)
                $f!(darg)
                if dA !== darg
                    dA .+= darg
                    zero!(darg)
                end
                copy!(arg, argc)
                return ntuple(Returns(NoRData()), 4)
            end

            return arg_darg, $adj
        end

        @is_primitive DefaultCtx Mooncake.ReverseMode Tuple{typeof($f), Any, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(f_df::CoDual{typeof($f)}, A_dA::CoDual, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm})
            A, dA = arrayify(A_dA)
            output = $f(A, Mooncake.primal(alg_dalg))
            output_doutput = Mooncake.zero_fcodual(output)

            doutput = last(arrayify(output_doutput))
            function $adj(::NoRData)
                # TODO: need accumulating projection to avoid intermediate here
                dA .+= $f(doutput)
                zero!(doutput)
                return ntuple(Returns(NoRData()), 3)
            end

            return output_doutput, $adj
        end
    end
end

end
