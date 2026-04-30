module MatrixAlgebraKitMooncakeExt

using Mooncake
using Mooncake: CoDual, Dual, NoRData, arrayify, primal, tangent, zero_fcodual
using MatrixAlgebraKit
using MatrixAlgebraKit: MatrixAlgebraKit as MAK,
    diagview, zero!, AbstractAlgorithm, TruncatedAlgorithm
using LinearAlgebra

# Utility
# -------

"""
    @mark_primitive f(::A, ::B, ::C, ::D...)

Helper macro to mark a function as primitive, i.e. transforms the expression into:

    Mooncake.@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode, Tuple{typeof(f), A, B, C, Vararg{D}}
"""
macro mark_primitive(ex)
    @assert Meta.isexpr(ex, :call)
    args = []
    # function name
    push!(args, :(typeof($(ex.args[1]))))

    # arguments
    for i in 2:(length(ex.args) - 1)
        argex = ex.args[i]
        @assert Meta.isexpr(argex, Symbol("::"))
        push!(args, last(argex.args))
    end

    # (possible) vararg
    if length(ex.args) > 1
        argex = last(ex.args)
        if Meta.isexpr(argex, Symbol("..."))
            argex2 = only(argex.args)
            @assert Meta.isexpr(argex2, Symbol("::"))
            push!(args, :(Vararg{$(last(argex2.args))}))
        else
            @assert Meta.isexpr(argex, Symbol("::"))
            push!(args, last(argex.args))
        end
    end

    return :(Mooncake.@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{$(args...)})
end

_warn_pullback_truncerror(dϵ::Real; tol = MatrixAlgebraKit.defaulttol(dϵ)) =
    abs(dϵ) ≤ tol || @warn "Pullback ignores non-zero tangents for truncation error"

# No derivatives
# --------------
Mooncake.tangent_type(::Type{<:MatrixAlgebraKit.AbstractAlgorithm}) = Mooncake.NoTangent

Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(MAK.select_algorithm), Any, Any, Any}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(Core.kwcall), NamedTuple, typeof(MAK.select_algorithm), Any, Any, Any}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(MAK.initialize_output), Any, Any, Any}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(MAK.check_input), Any, Any, Any, Any}

# Factorizations
# --------------
#=
The general approach here is to define the functions in terms of the non-mutating versions first.
Since we are not guaranteeing that we will be mutating the input, nor that we will make
use of the provided output buffers, we can simplify our lives by calling the non-mutating
implementations instead of the mutating ones.

The main benefit here is that we do not have to guarantee that we will restore the state
after executing the pullback - ensuring that we don't have to keep as many copied objects
around. This being said, the total number of allocations does not become smaller because
of this, and in cases where the pullback would be used multiple times we now have to
allocate multiple times. On the other hand, we can also free these objects inbetween, so
this might also reduce the total GC pressure...
=#

# Multi-output functions
for (f, pullback!, adjoint) in [
        (:qr_full, :qr_pullback!, :qr_adjoint),
        (:qr_compact, :qr_pullback!, :qr_adjoint),
        (:lq_full, :lq_pullback!, :lq_adjoint),
        (:lq_compact, :lq_pullback!, :lq_adjoint),
        (:eig_full, :eig_pullback!, :eig_adjoint!),
        (:eig_trunc_no_error, :eig_trunc_pullback!, :eig_adjoint),
        (:eigh_full, :eigh_pullback!, :eigh_adjoint),
        (:eigh_trunc_no_error, :eigh_trunc_pullback!, :eigh_adjoint),
        (:left_polar, :left_polar_pullback!, :left_polar_adjoint),
        (:right_polar, :right_polar_pullback!, :right_polar_adjoint),
        (:svd_compact, :svd_pullback!, :svd_adjoint),
        (:svd_full, :svd_pullback!, :svd_adjoint),
        (:svd_trunc_no_error, :svd_trunc_pullback!, :svd_adjoint),
    ]
    f! = Symbol(f, :!)
    @eval begin
        # first handle the non-inplace version
        @mark_primitive $f(::Any, ::AbstractAlgorithm)
        function Mooncake.rrule!!(::CoDual{typeof($f)}, A_dA::CoDual, alg_dalg::CoDual{<:AbstractAlgorithm})
            # unpack variables
            A, dA = arrayify(A_dA)
            alg = primal(alg_dalg)

            # compute primal and pack output
            args = $f(A, alg)
            args_dargs = zero_fcodual(args)

            # define pullback
            dargs = last.(arrayify.(args, tangent(args_dargs)))
            function $adjoint(::NoRData)
                MAK.$pullback!(dA, A, args, dargs)
                return ntuple(Returns(NoRData()), 3)
            end

            return args_dargs, $adjoint
        end

        # then handle the inplace version
        @mark_primitive $f!(::Any, ::Tuple, ::AbstractAlgorithm)
        function Mooncake.rrule!!(::CoDual{typeof($f!)}, A_dA::CoDual, args_dargs::CoDual, alg_dalg::CoDual{<:AbstractAlgorithm})
            args_dargs, pb! = Mooncake.rrule!!(zero_fcodual($f), A_dA, alg_dalg)
            function $adjoint(x::NoRData)
                pb!(x)
                return ntuple(Returns(NoRData()), 4)
            end
            return args_dargs, $adjoint
        end
    end
end

# Single-output functions
for (f, pullback!, adjoint) in [
        (:qr_null, :qr_null_pullback!, :qr_null_adjoint!),
        (:lq_null, :lq_null_pullback!, :lq_null_adjoint!),
    ]
    f! = Symbol(f, :!)
    @eval begin
        # first handle the non-inplace version
        @mark_primitive $f(::Any, ::AbstractAlgorithm)
        function Mooncake.rrule!!(::CoDual{typeof($f)}, A_dA::CoDual, alg_dalg::CoDual{<:AbstractAlgorithm})
            # unpack variables
            A, dA = arrayify(A_dA)
            alg = primal(alg_dalg)

            # compute primal and pack output
            arg = $f(A, alg)
            arg_darg = zero_fcodual(arg)

            # define pullback
            darg = last(arrayify(arg, tangent(arg_darg)))
            function $adjoint(::NoRData)
                MAK.$pullback!(dA, A, arg, darg)
                return ntuple(Returns(NoRData()), 3)
            end

            return arg_darg, $adjoint
        end

        # then handle the inplace version
        @mark_primitive $f!(::Any, ::Any, ::AbstractAlgorithm)
        function Mooncake.rrule!!(::CoDual{typeof($f!)}, A_dA::CoDual, arg_darg::CoDual, alg_dalg::CoDual{<:AbstractAlgorithm})
            arg_darg, pb! = Mooncake.rrule!!(zero_fcodual($f), A_dA, alg_dalg)
            function $adjoint(x::NoRData)
                pb!(x)
                return ntuple(Returns(NoRData()), 4)
            end
            return arg_darg, $adjoint
        end
    end
end

for f in [:eig, :eigh, :svd]
    f_vals = Symbol(f, :_vals)
    f_vals! = Symbol(f_vals, :!)
    vals_pullback! = Symbol(f, :_vals_pullback!)

    f_full = f === :svd ? Symbol(f, :_compact) : Symbol(f, :_full)
    adjoint = Symbol(f, :_adjoint)

    # Values functions
    @eval begin
        # first handle the non-inplace version
        @mark_primitive $f_vals(::Any, ::AbstractAlgorithm)
        function Mooncake.rrule!!(::CoDual{typeof($f_vals)}, A_dA::CoDual, alg_dalg::CoDual)
            # unpack variables
            A, dA = arrayify(A_dA)
            alg = primal(alg_dalg)

            # compute primal and pack output - store full decomposition for pullback
            F = $f_full(A, alg)
            vals = diagview(F[$(f === :svd ? 2 : 1)])
            vals_dvals = zero_fcodual(vals)

            # define pullback
            dvals = last(arrayify(vals, tangent(vals_dvals)))
            function $adjoint(::NoRData)
                MAK.$vals_pullback!(dA, A, F, dvals)
                return ntuple(Returns(NoRData()), 3)
            end

            return vals_dvals, $adjoint
        end

        # then handle the inplace version
        @mark_primitive $f_vals!(::Any, ::Any, ::AbstractAlgorithm)
        function Mooncake.rrule!!(::CoDual{typeof($f_vals!)}, A_dA::CoDual, vals_dvals::CoDual, alg_dalg::CoDual)
            vals_dvals, pb! = Mooncake.rrule!!(zero_fcodual($f_vals), A_dA, alg_dalg)
            function $adjoint(x::NoRData)
                pb!(x)
                return ntuple(Returns(NoRData()), 4)
            end
            return vals_dvals, $adjoint
        end
    end

    # Truncated decompositions
    f_trunc = Symbol(f, :_trunc)
    f_trunc! = Symbol(f_trunc, :!)
    f_trunc_no_error = Symbol(f_trunc, :_no_error)
    f_trunc_no_error! = Symbol(f_trunc_no_error, :!)
    pullback! = Symbol(f, :_pullback!)
    trunc_pullback! = Symbol(f_trunc, :_pullback!)

    @eval begin
        # By default we use `f_trunc_pullback`
        @mark_primitive $f_trunc(::Any, ::AbstractAlgorithm)
        function Mooncake.rrule!!(::CoDual{typeof($f_trunc)}, A_dA::CoDual, alg_dalg::CoDual)
            # unpack variables
            A, dA = arrayify(A_dA)
            alg = primal(alg_dalg)

            # compute primal and pack output
            argsϵ = $f_trunc(A, alg)
            argsϵ_dargsϵ = zero_fcodual(argsϵ)

            # define pullback
            args = Base.front(args)
            dargs = last.(arrayify.(args, Base.front(tangent(argsϵ_dargsϵ))))
            function $adjoint(dy)
                _warn_pullback_truncerror(last(dy))
                MAK.$trunc_pullback!(dA, A, args, dargs)
                return ntuple(_nordata, 3)
            end

            return argsϵ_dargsϵ, $adjoint
        end

        # TruncatedAlgorithm computes f_full, so use `f_pullback` + `inds`
        function Mooncake.rrule!!(::CoDual{typeof($f_trunc)}, A_dA::CoDual, alg_dalg::CoDual{<:TruncatedAlgorithm})
            # unpack variables
            A, dA = arrayify(A_dA)
            alg = primal(alg_dalg)

            # compute primal and pack output - capture full args and ind
            args_full = $f_full(A, alg.alg)
            args, ind = MAK.truncate($f_trunc!, args_full, alg.trunc)
            ϵ = MAK.truncation_error(diagview(args_full[$(f === :svd ? 2 : 1)]), ind)
            argsϵ = (args..., ϵ)
            argsϵ_dargsϵ = zero_fcodual(argsϵ)

            # define pullback
            dargs = last.(arrayify.(args, Base.front(tangent(argsϵ_dargsϵ))))
            function $adjoint(dy)
                _warn_pullback_truncerror(last(dy))
                MAK.$pullback!(dA, A, args_full, dargs, ind)
                return ntuple(Returns(NoRData()), 3)
            end

            return argsϵ_dargsϵ, $adjoint
        end

        @mark_primitive $f_trunc!(::Any, ::Any, ::AbstractAlgorithm)
        function Mooncake.rrule!!(::CoDual{typeof($f_trunc!)}, A_dA::CoDual, args_dargs::CoDual, alg_dalg::CoDual)
            args_dargs, pb! = Mooncake.rrule!!(zero_fcodual($f_trunc), A_dA, alg_dalg)
            function $adjoint(x)
                pb!(x)
                return ntuple(Returns(NoRData()), 4)
            end
            return args_dargs, $adjoint
        end

        # no_error versions are already handled above, but
        # still need specialized implementation for <:TruncatedAlgorithm
        function Mooncake.rrule!!(::CoDual{typeof($f_trunc_no_error)}, A_dA::CoDual, alg_dalg::CoDual{<:TruncatedAlgorithm})
            # unpack variables
            A, dA = arrayify(A_dA)
            alg = primal(alg_dalg)

            # compute primal and pack output - capture full DV and ind
            args_full = $f_full(A, alg.alg)
            args, ind = MAK.truncate($f_trunc!, args_full, alg.trunc)
            args_dargs = zero_fcodual(args)

            # define pullback
            dargs = last.(arrayify.(args, tangent(args_dargs)))
            function $adjoint(::NoRData)
                MAK.$pullback!(dA, A, args_full, dargs, ind)
                return ntuple(Returns(NoRData()), 3)
            end

            return args_dargs, $adjoint
        end
    end
end
#
# # two-argument in-place factorizations like LQ, QR, EIG
# for (f!, f, pb, adj) in (
#         (:qr_full!, :qr_full, :qr_pullback!, :qr_adjoint),
#         (:lq_full!, :lq_full, :lq_pullback!, :lq_adjoint),
#         (:qr_compact!, :qr_compact, :qr_pullback!, :qr_adjoint),
#         (:lq_compact!, :lq_compact, :lq_pullback!, :lq_adjoint),
#         (:eig_full!, :eig_full, :eig_pullback!, :eig_adjoint),
#         (:eigh_full!, :eigh_full, :eigh_pullback!, :eigh_adjoint),
#         (:left_polar!, :left_polar, :left_polar_pullback!, :left_polar_adjoint),
#         (:right_polar!, :right_polar, :right_polar_pullback!, :right_polar_adjoint),
#     )
#
#     @eval begin
#         @mark_primitive $f!(::Any, ::Tuple{<:Any, <:Any}, ::MatrixAlgebraKit.AbstractAlgorithm)
#         function Mooncake.rrule!!(::CoDual{typeof($f!)}, A_dA::CoDual, args_dargs::CoDual, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm})
#             A, dA = arrayify(A_dA)
#             args = Mooncake.primal(args_dargs)
#             dargs = Mooncake.tangent(args_dargs)
#             arg1, darg1 = arrayify(args[1], dargs[1])
#             arg2, darg2 = arrayify(args[2], dargs[2])
#             Ac = copy(A)
#             arg1c = copy(arg1)
#             arg2c = copy(arg2)
#             $f!(A, args, Mooncake.primal(alg_dalg))
#             function $adj(::NoRData)
#                 # DON'T copy Ac to A if A === one
#                 # of the output args -- this can
#                 # mess up the pullback because
#                 # generally the args are used there
#                 if !(A === arg1 || A === arg2)
#                     copy!(A, Ac)
#                     $pb(dA, A, (arg1, arg2), (darg1, darg2))
#                 else
#                     ΔA = zero(A)
#                     $pb(ΔA, A, (arg1, arg2), (darg1, darg2))
#                     dA .= ΔA
#                 end
#                 if A === arg1
#                     zero!(darg2)
#                 elseif A === arg2
#                     zero!(darg1)
#                 else
#                     zero!(darg1)
#                     zero!(darg2)
#                 end
#                 copy!(arg2, arg2c)
#                 copy!(arg1, arg1c)
#                 return ntuple(Returns(NoRData()), 4)
#             end
#             return args_dargs, $adj
#         end
#         @mark_primitive $f(::Any, ::MatrixAlgebraKit.AbstractAlgorithm)
#         function Mooncake.rrule!!(::CoDual{typeof($f)}, A_dA::CoDual, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm})
#             A, dA = arrayify(A_dA)
#             output = $f(A, Mooncake.primal(alg_dalg))
#             # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
#             # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
#             # pass). For many types this is done automatically when the forward step returns, but
#             # not for nested structs with various fields (like Diagonal{Complex})
#             output_codual = CoDual(output, Mooncake.fdata(Mooncake.zero_tangent(output)))
#             function $adj(::NoRData)
#                 arg1, arg2 = Mooncake.primal(output_codual)
#                 darg1_, darg2_ = Mooncake.tangent(output_codual)
#                 arg1, darg1 = arrayify(arg1, darg1_)
#                 arg2, darg2 = arrayify(arg2, darg2_)
#                 $pb(dA, A, (arg1, arg2), (darg1, darg2))
#                 zero!(darg1)
#                 zero!(darg2)
#                 return NoRData(), NoRData(), NoRData()
#             end
#             return output_codual, $adj
#         end
#     end
# end
#
# for (f!, f, pb, adj) in (
#         (:qr_null!, :qr_null, :qr_null_pullback!, :qr_null_adjoint),
#         (:lq_null!, :lq_null, :lq_null_pullback!, :lq_null_adjoint),
#     )
#     @eval begin
#         @mark_primitive $f!(::Any, ::Any, ::MatrixAlgebraKit.AbstractAlgorithm)
#         function Mooncake.rrule!!(f_df::CoDual{typeof($f!)}, A_dA::CoDual, arg_darg::CoDual, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm})
#             A, dA = arrayify(A_dA)
#             Ac = copy(A)
#             arg, darg = arrayify(arg_darg)
#             argc = copy(arg)
#             $f!(A, arg, Mooncake.primal(alg_dalg))
#             function $adj(::NoRData)
#                 copy!(A, Ac)
#                 $pb(dA, A, arg, darg)
#                 copy!(arg, argc)
#                 zero!(darg)
#                 return NoRData(), NoRData(), NoRData(), NoRData()
#             end
#             return arg_darg, $adj
#         end
#         @mark_primitive $f(::Any, ::MatrixAlgebraKit.AbstractAlgorithm)
#         function Mooncake.rrule!!(f_df::CoDual{typeof($f)}, A_dA::CoDual, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm})
#             A, dA = arrayify(A_dA)
#             output = $f(A, Mooncake.primal(alg_dalg))
#             output_codual = CoDual(output, Mooncake.zero_tangent(output))
#             function $adj(::NoRData)
#                 arg, darg = arrayify(output_codual)
#                 $pb(dA, A, arg, darg)
#                 zero!(darg)
#                 return NoRData(), NoRData(), NoRData()
#             end
#             return output_codual, $adj
#         end
#     end
# end
#
# for (f!, f, f_full, pb, adj) in (
#         (:eig_vals!, :eig_vals, :eig_full, :eig_vals_pullback!, :eig_vals_adjoint),
#         (:eigh_vals!, :eigh_vals, :eigh_full, :eigh_vals_pullback!, :eigh_vals_adjoint),
#     )
#     @eval begin
#         @mark_primitive $f!(::Any, ::Any, ::MatrixAlgebraKit.AbstractAlgorithm)
#         function Mooncake.rrule!!(::CoDual{typeof($f!)}, A_dA::CoDual, D_dD::CoDual, alg_dalg::CoDual)
#             # compute primal
#             A, dA = arrayify(A_dA)
#             D, dD = arrayify(D_dD)
#             Dc = copy(D)
#             # update primal
#             DV = $f_full(A, Mooncake.primal(alg_dalg))
#             copy!(D, diagview(DV[1]))
#             V = DV[2]
#             function $adj(::NoRData)
#                 if A !== D
#                     $pb(dA, A, DV, dD)
#                 else
#                     ΔA = zero(A)
#                     $pb(ΔA, A, DV, dD)
#                     dA .= A
#                 end
#                 if A !== D
#                     zero!(dD)
#                     copy!(D, Dc)
#                 else
#                     copy!(A, Ac)
#                 end
#                 return NoRData(), NoRData(), NoRData(), NoRData()
#             end
#             return D_dD, $adj
#         end
#         @mark_primitive $f(::Any, ::MatrixAlgebraKit.AbstractAlgorithm)
#         function Mooncake.rrule!!(::CoDual{typeof($f)}, A_dA::CoDual, alg_dalg::CoDual)
#             # compute primal
#             A, dA = arrayify(A_dA)
#             # update primal
#             DV = $f_full(A, Mooncake.primal(alg_dalg))
#             V = DV[2]
#             output = diagview(DV[1])
#             output_codual = CoDual(output, Mooncake.zero_tangent(output))
#             function $adj(::NoRData)
#                 D, dD = arrayify(output_codual)
#                 $pb(dA, A, DV, dD)
#                 zero!(dD)
#                 return NoRData(), NoRData(), NoRData()
#             end
#             return output_codual, $adj
#         end
#     end
# end
#
# _warn_pullback_truncerror(dϵ::Real; tol = MatrixAlgebraKit.defaulttol(dϵ)) =
#     abs(dϵ) ≤ tol || @warn "Pullback ignores non-zero tangents for truncation error"
#
# for f in (:eig, :eigh)
#     f_trunc = Symbol(f, :_trunc)
#     f_trunc! = Symbol(f_trunc, :!)
#     f_full = Symbol(f, :_full)
#     f_full! = Symbol(f_full, :!)
#     f_pullback! = Symbol(f, :_pullback!)
#     f_trunc_pullback! = Symbol(f_trunc, :_pullback!)
#     f_adjoint! = Symbol(f, :_adjoint!)
#     f_trunc_no_error = Symbol(f_trunc, :_no_error)
#     f_trunc_no_error! = Symbol(f_trunc_no_error, :!)
#
#     @eval begin
#         @mark_primitive $f_trunc!(::Any, ::Any, ::MatrixAlgebraKit.AbstractAlgorithm)
#         @mark_primitive $f_trunc(::Any, ::MatrixAlgebraKit.AbstractAlgorithm)
#         function Mooncake.rrule!!(::CoDual{typeof($f_trunc!)}, A_dA::CoDual, DV_dDV::CoDual, alg_dalg::CoDual)
#             # compute primal
#             A, dA = arrayify(A_dA)
#             DV = Mooncake.primal(DV_dDV)
#             dDV = Mooncake.tangent(DV_dDV)
#             Ac = copy(A)
#             DVc = copy.(DV)
#             alg = Mooncake.primal(alg_dalg)
#             output = $f_trunc!(A, DV, alg)
#             # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
#             # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
#             # pass). For many types this is done automatically when the forward step returns, but
#             # not for nested structs with various fields (like Diagonal{Complex})
#             output_codual = Mooncake.zero_fcodual(output)
#             function $f_adjoint!(dy::Tuple{NoRData, NoRData, <:Real})
#                 Dtrunc, Vtrunc, ϵ = Mooncake.primal(output_codual)
#                 dDtrunc_, dVtrunc_, dϵ = Mooncake.tangent(output_codual)
#                 _warn_pullback_truncerror(dy[3])
#                 D′, dD′ = arrayify(Dtrunc, dDtrunc_)
#                 V′, dV′ = arrayify(Vtrunc, dVtrunc_)
#                 D, dD = arrayify(DV[1], dDV[1])
#                 V, dV = arrayify(DV[2], dDV[2])
#                 copy!(A, Ac)
#                 if !(A === D || A === V)
#                     $f_trunc_pullback!(dA, A, (D′, V′), (dD′, dV′))
#                 else
#                     ΔA = zero(A)
#                     $f_trunc_pullback!(ΔA, A, (D′, V′), (dD′, dV′))
#                     dA .= ΔA
#                 end
#                 if A === D
#                     copy!(DV[2], DVc[2])
#                 else
#                     copy!(DV[1], DVc[1])
#                     copy!(DV[2], DVc[2])
#                 end
#                 zero!(dD′)
#                 zero!(dV′)
#                 return NoRData(), NoRData(), NoRData(), NoRData()
#             end
#             return output_codual, $f_adjoint!
#         end
#         function Mooncake.rrule!!(::CoDual{typeof($f_trunc!)}, A_dA::CoDual, DV_dDV::CoDual, alg_dalg::CoDual{<:TruncatedAlgorithm})
#             # unpack variables
#             A, dA = arrayify(A_dA)
#             DV_dDV_arr = arrayify.(Mooncake.primal(DV_dDV), Mooncake.tangent(DV_dDV))
#             DV, dDV = first.(DV_dDV_arr), last.(DV_dDV_arr)
#             alg = Mooncake.primal(alg_dalg)
#
#             # store state prior to primal call
#             Ac = copy(A)
#             DVc = copy.(DV)
#
#             # compute primal - capture full DV and ind
#             DV = $f_full!(A, DV, alg.alg)
#             DVtrunc, ind = MatrixAlgebraKit.truncate($f_trunc!, DV, alg.trunc)
#             ϵ = MatrixAlgebraKit.truncation_error(diagview(DV[1]), ind)
#
#             # pack output - note that we allocate new dDVtrunc because these aren't overwritten in the input
#             DVtrunc_dDVtrunc = Mooncake.zero_fcodual((DVtrunc..., ϵ))
#
#             # define pullback
#             dDVtrunc = last.(arrayify.(DVtrunc, Base.front(Mooncake.tangent(DVtrunc_dDVtrunc))))
#             function $f_adjoint!((_, _, dϵ)::Tuple{NoRData, NoRData, Real})
#                 _warn_pullback_truncerror(dϵ)
#
#                 # compute pullbacks
#                 if !(A === DV[1] || A === DV[2])
#                     $f_pullback!(dA, Ac, DV, dDVtrunc, ind)
#                 else
#                     ΔA = zero(A)
#                     $f_pullback!(ΔA, Ac, DV, dDVtrunc, ind)
#                     dA .= ΔA
#                 end
#                 # restore state
#                 copy!(A, Ac)
#                 if A === DV[1]
#                     copy!(DV[2], DVc[2])
#                     zero!(dDV[2])
#                 else
#                     copy!.(DV, DVc)
#                     zero!.(dDV)
#                 end
#
#                 return ntuple(Returns(NoRData()), 4)
#             end
#
#             return DVtrunc_dDVtrunc, $f_adjoint!
#         end
#         function Mooncake.rrule!!(::CoDual{typeof($f_trunc)}, A_dA::CoDual, alg_dalg::CoDual)
#             # compute primal
#             A, dA = arrayify(A_dA)
#             alg = Mooncake.primal(alg_dalg)
#             output = $f_trunc(A, alg)
#             # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
#             # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
#             # pass). For many types this is done automatically when the forward step returns, but
#             # not for nested structs with various fields (like Diagonal{Complex})
#             output_codual = CoDual(output, Mooncake.fdata(Mooncake.zero_tangent(output)))
#             function $f_adjoint!(dy::Tuple{NoRData, NoRData, T}) where {T <: Real}
#                 Dtrunc, Vtrunc, ϵ = Mooncake.primal(output_codual)
#                 dDtrunc_, dVtrunc_, dϵ = Mooncake.tangent(output_codual)
#                 _warn_pullback_truncerror(dy[3])
#                 D, dD = arrayify(Dtrunc, dDtrunc_)
#                 V, dV = arrayify(Vtrunc, dVtrunc_)
#                 $f_trunc_pullback!(dA, A, (D, V), (dD, dV))
#                 zero!(dD)
#                 zero!(dV)
#                 return NoRData(), NoRData(), NoRData()
#             end
#             return output_codual, $f_adjoint!
#         end
#         function Mooncake.rrule!!(::CoDual{typeof($f_trunc)}, A_dA::CoDual, alg_dalg::CoDual{<:TruncatedAlgorithm})
#             # unpack variables
#             A, dA = arrayify(A_dA)
#             alg = Mooncake.primal(alg_dalg)
#
#             # compute primal - capture full DV and ind
#             DV = $f_full(A, alg.alg)
#             DVtrunc, ind = MatrixAlgebraKit.truncate($f_trunc!, DV, alg.trunc)
#             ϵ = MatrixAlgebraKit.truncation_error(diagview(DV[1]), ind)
#
#             # pack output
#             DVtrunc_dDVtrunc = Mooncake.zero_fcodual((DVtrunc..., ϵ))
#
#             # define pullback
#             dDVtrunc = last.(arrayify.(DVtrunc, Base.front(Mooncake.tangent(DVtrunc_dDVtrunc))))
#             function $f_adjoint!((_, _, dϵ)::Tuple{NoRData, NoRData, Real})
#                 _warn_pullback_truncerror(dϵ)
#                 $f_pullback!(dA, A, DV, dDVtrunc, ind)
#                 zero!.(dDVtrunc) # since this is allocated in this function this is probably not required
#                 return ntuple(Returns(NoRData()), 3)
#             end
#
#             return DVtrunc_dDVtrunc, $f_adjoint!
#         end
#         @mark_primitive $f_trunc_no_error!(::Any, ::Any, ::MatrixAlgebraKit.AbstractAlgorithm)
#         @mark_primitive $f_trunc_no_error(::Any, ::MatrixAlgebraKit.AbstractAlgorithm)
#         function Mooncake.rrule!!(::CoDual{typeof($f_trunc_no_error!)}, A_dA::CoDual, DV_dDV::CoDual, alg_dalg::CoDual)
#             # compute primal
#             A, dA = arrayify(A_dA)
#             alg = Mooncake.primal(alg_dalg)
#             DV = Mooncake.primal(DV_dDV)
#             dDV = Mooncake.tangent(DV_dDV)
#             Ac = copy(A)
#             DVc = copy.(DV)
#             output = $f_trunc_no_error!(A, DV, alg)
#             # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
#             # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
#             # pass). For many types this is done automatically when the forward step returns, but
#             # not for nested structs with various fields (like Diagonal{Complex})
#             output_codual = CoDual(output, Mooncake.fdata(Mooncake.zero_tangent(output)))
#             function $f_adjoint!(::NoRData)
#                 copy!(A, Ac)
#                 Dtrunc, Vtrunc = Mooncake.primal(output_codual)
#                 dDtrunc_, dVtrunc_ = Mooncake.tangent(output_codual)
#                 D′, dD′ = arrayify(Dtrunc, dDtrunc_)
#                 V′, dV′ = arrayify(Vtrunc, dVtrunc_)
#                 $f_pullback!(dA, A, (D′, V′), (dD′, dV′))
#                 copy!(DV[1], DVc[1])
#                 copy!(DV[2], DVc[2])
#                 zero!(dD′)
#                 zero!(dV′)
#                 return NoRData(), NoRData(), NoRData(), NoRData()
#             end
#             return output_codual, $f_adjoint!
#         end
#         function Mooncake.rrule!!(::CoDual{typeof($f_trunc_no_error!)}, A_dA::CoDual, DV_dDV::CoDual, alg_dalg::CoDual{<:TruncatedAlgorithm})
#             # unpack variables
#             A, dA = arrayify(A_dA)
#             DV_dDV_arr = arrayify.(Mooncake.primal(DV_dDV), Mooncake.tangent(DV_dDV))
#             DV, dDV = first.(DV_dDV_arr), last.(DV_dDV_arr)
#             alg = Mooncake.primal(alg_dalg)
#
#             # store state prior to primal call
#             Ac = copy(A)
#             DVc = copy.(DV)
#
#             # compute primal - capture full DV and ind
#             DV = $f_full!(A, DV, alg.alg)
#             DVtrunc, ind = MatrixAlgebraKit.truncate($f_trunc!, DV, alg.trunc)
#
#             # pack output - note that we allocate new dDVtrunc because these aren't overwritten in the input
#             DVtrunc_dDVtrunc = Mooncake.zero_fcodual(DVtrunc)
#
#             # define pullback
#             dDVtrunc = last.(arrayify.(DVtrunc, Mooncake.tangent(DVtrunc_dDVtrunc)))
#             function $f_adjoint!(::NoRData)
#                 # compute pullbacks
#                 if !(A === DV[1] || A === DV[2])
#                     $f_pullback!(dA, Ac, DV, dDVtrunc, ind)
#                 else
#                     ΔA = zero(A)
#                     $f_pullback!(ΔA, Ac, DV, dDVtrunc, ind)
#                     dA .= ΔA
#                 end
#
#                 # restore state
#                 copy!(A, Ac)
#                 if A === DV[1]
#                     copy!(DV[2], DVc[2])
#                     zero!(dDV[2])
#                 else
#                     copy!.(DV, DVc)
#                     zero!.(dDV)
#                 end
#
#                 return ntuple(Returns(NoRData()), 4)
#             end
#
#             return DVtrunc_dDVtrunc, $f_adjoint!
#         end
#         function Mooncake.rrule!!(::CoDual{typeof($f_trunc_no_error)}, A_dA::CoDual, alg_dalg::CoDual)
#             # compute primal
#             A, dA = arrayify(A_dA)
#             alg = Mooncake.primal(alg_dalg)
#             output = $f_trunc_no_error(A, alg)
#             # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
#             # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
#             # pass). For many types this is done automatically when the forward step returns, but
#             # not for nested structs with various fields (like Diagonal{Complex})
#             output_codual = CoDual(output, Mooncake.fdata(Mooncake.zero_tangent(output)))
#             function $f_adjoint!(::NoRData)
#                 Dtrunc, Vtrunc = Mooncake.primal(output_codual)
#                 dDtrunc_, dVtrunc_ = Mooncake.tangent(output_codual)
#                 D, dD = arrayify(Dtrunc, dDtrunc_)
#                 V, dV = arrayify(Vtrunc, dVtrunc_)
#                 $f_trunc_pullback!(dA, A, (D, V), (dD, dV))
#                 zero!(dD)
#                 zero!(dV)
#                 return NoRData(), NoRData(), NoRData()
#             end
#             return output_codual, $f_adjoint!
#         end
#         function Mooncake.rrule!!(::CoDual{typeof($f_trunc_no_error)}, A_dA::CoDual, alg_dalg::CoDual{<:TruncatedAlgorithm})
#             # unpack variables
#             A, dA = arrayify(A_dA)
#             alg = Mooncake.primal(alg_dalg)
#
#             # compute primal - capture full DV and ind
#             DV = $f_full(A, alg.alg)
#             DVtrunc, ind = MatrixAlgebraKit.truncate($f_trunc!, DV, alg.trunc)
#
#             # pack output
#             DVtrunc_dDVtrunc = Mooncake.zero_fcodual(DVtrunc)
#
#             # define pullback
#             dDVtrunc = last.(arrayify.(DVtrunc, Mooncake.tangent(DVtrunc_dDVtrunc)))
#             function $f_adjoint!(::NoRData)
#                 $f_pullback!(dA, A, DV, dDVtrunc, ind)
#                 zero!.(dDVtrunc) # since this is allocated in this function this is probably not required
#                 return ntuple(Returns(NoRData()), 3)
#             end
#
#             return DVtrunc_dDVtrunc, $f_adjoint!
#         end
#     end
# end
#
# for (f!, f) in (
#         (:svd_full!, :svd_full),
#         (:svd_compact!, :svd_compact),
#     )
#     @eval begin
#         @mark_primitive $f!(::Any, ::Tuple{<:Any, <:Any, <:Any}, ::MatrixAlgebraKit.AbstractAlgorithm)
#         function Mooncake.rrule!!(::CoDual{typeof($f!)}, A_dA::CoDual, USVᴴ_dUSVᴴ::CoDual, alg_dalg::CoDual)
#             A, dA = arrayify(A_dA)
#             USVᴴ = Mooncake.primal(USVᴴ_dUSVᴴ)
#             dUSVᴴ = Mooncake.tangent(USVᴴ_dUSVᴴ)
#             U, dU = arrayify(USVᴴ[1], dUSVᴴ[1])
#             S, dS = arrayify(USVᴴ[2], dUSVᴴ[2])
#             Vᴴ, dVᴴ = arrayify(USVᴴ[3], dUSVᴴ[3])
#             Ac = copy(A)
#             USVᴴc = copy.(USVᴴ)
#             output = $f!(A, USVᴴ, Mooncake.primal(alg_dalg))
#             function svd_adjoint(::NoRData)
#                 copy!(A, Ac)
#                 svd_pullback!(dA, A, (U, S, Vᴴ), (dU, dS, dVᴴ))
#                 copy!(U, USVᴴc[1])
#                 copy!(S, USVᴴc[2])
#                 copy!(Vᴴ, USVᴴc[3])
#                 zero!(dU)
#                 zero!(dS)
#                 zero!(dVᴴ)
#                 return NoRData(), NoRData(), NoRData(), NoRData()
#             end
#             return USVᴴ_dUSVᴴ, svd_adjoint
#         end
#         @mark_primitive $f(::Any, ::MatrixAlgebraKit.AbstractAlgorithm)
#         function Mooncake.rrule!!(::CoDual{typeof($f)}, A_dA::CoDual, alg_dalg::CoDual)
#             A, dA = arrayify(A_dA)
#             USVᴴ = $f(A, Mooncake.primal(alg_dalg))
#             # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
#             # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
#             # pass). For many types this is done automatically when the forward step returns, but
#             # not for nested structs with various fields (like Diagonal{Complex})
#             USVᴴ_codual = CoDual(USVᴴ, Mooncake.fdata(Mooncake.zero_tangent(USVᴴ)))
#             function svd_adjoint(::NoRData)
#                 U, S, Vᴴ = Mooncake.primal(USVᴴ_codual)
#                 dU_, dS_, dVᴴ_ = Mooncake.tangent(USVᴴ_codual)
#                 U, dU = arrayify(U, dU_)
#                 S, dS = arrayify(S, dS_)
#                 Vᴴ, dVᴴ = arrayify(Vᴴ, dVᴴ_)
#                 svd_pullback!(dA, A, (U, S, Vᴴ), (dU, dS, dVᴴ))
#                 zero!(dU)
#                 zero!(dS)
#                 zero!(dVᴴ)
#                 return NoRData(), NoRData(), NoRData()
#             end
#             return USVᴴ_codual, svd_adjoint
#         end
#     end
# end
#
# @mark_primitive svd_vals!(::Any, ::Any, ::MatrixAlgebraKit.AbstractAlgorithm)
# function Mooncake.rrule!!(::CoDual{typeof(svd_vals!)}, A_dA::CoDual, S_dS::CoDual, alg_dalg::CoDual)
#     # compute primal
#     A, dA = arrayify(A_dA)
#     S, dS = arrayify(S_dS)
#     Sc = copy(S)
#     USVᴴ = svd_compact(A, Mooncake.primal(alg_dalg))
#     copy!(S, diagview(USVᴴ[2]))
#     function svd_vals_adjoint(::NoRData)
#         svd_vals_pullback!(dA, A, USVᴴ, dS)
#         zero!(dS)
#         copy!(S, Sc)
#         return NoRData(), NoRData(), NoRData(), NoRData()
#     end
#     return S_dS, svd_vals_adjoint
# end
#
# @mark_primitive svd_vals(::Any, ::MatrixAlgebraKit.AbstractAlgorithm)
# function Mooncake.rrule!!(::CoDual{typeof(svd_vals)}, A_dA::CoDual, alg_dalg::CoDual)
#     # compute primal
#     A, dA = arrayify(A_dA)
#     USVᴴ = svd_compact(A, Mooncake.primal(alg_dalg))
#     # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
#     # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
#     # pass). For many types this is done automatically when the forward step returns, but
#     # not for nested structs with various fields (like Diagonal{Complex})
#     S = diagview(USVᴴ[2])
#     S_codual = CoDual(S, Mooncake.fdata(Mooncake.zero_tangent(S)))
#     function svd_vals_adjoint(::NoRData)
#         S, dS = arrayify(S_codual)
#         svd_vals_pullback!(dA, A, USVᴴ, dS)
#         zero!(dS)
#         return NoRData(), NoRData(), NoRData()
#     end
#     return S_codual, svd_vals_adjoint
# end
#
# @mark_primitive svd_trunc!(::Any, ::Any, ::MatrixAlgebraKit.AbstractAlgorithm)
# function Mooncake.rrule!!(::CoDual{typeof(svd_trunc!)}, A_dA::CoDual, USVᴴ_dUSVᴴ::CoDual, alg_dalg::CoDual)
#     # compute primal
#     A, dA = arrayify(A_dA)
#     alg = Mooncake.primal(alg_dalg)
#     Ac = copy(A)
#     USVᴴ = Mooncake.primal(USVᴴ_dUSVᴴ)
#     dUSVᴴ = Mooncake.tangent(USVᴴ_dUSVᴴ)
#     U, dU = arrayify(USVᴴ[1], dUSVᴴ[1])
#     S, dS = arrayify(USVᴴ[2], dUSVᴴ[2])
#     Vᴴ, dVᴴ = arrayify(USVᴴ[3], dUSVᴴ[3])
#     USVᴴc = copy.(USVᴴ)
#     output = svd_trunc!(A, USVᴴ, alg)
#     # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
#     # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
#     # pass). For many types this is done automatically when the forward step returns, but
#     # not for nested structs with various fields (like Diagonal{Complex})
#     output_codual = Mooncake.zero_fcodual(output)
#     function svd_trunc_adjoint(dy::Tuple{NoRData, NoRData, NoRData, T}) where {T <: Real}
#         copy!(A, Ac)
#         Utrunc, Strunc, Vᴴtrunc, ϵ = Mooncake.primal(output_codual)
#         dUtrunc_, dStrunc_, dVᴴtrunc_, dϵ = Mooncake.tangent(output_codual)
#         _warn_pullback_truncerror(dy[4])
#         U′, dU′ = arrayify(Utrunc, dUtrunc_)
#         S′, dS′ = arrayify(Strunc, dStrunc_)
#         Vᴴ′, dVᴴ′ = arrayify(Vᴴtrunc, dVᴴtrunc_)
#         svd_trunc_pullback!(dA, A, (U′, S′, Vᴴ′), (dU′, dS′, dVᴴ′))
#         copy!(U, USVᴴc[1])
#         copy!(S, USVᴴc[2])
#         copy!(Vᴴ, USVᴴc[3])
#         zero!(dU)
#         zero!(dS)
#         zero!(dVᴴ)
#         zero!(dU′)
#         zero!(dS′)
#         zero!(dVᴴ′)
#         return NoRData(), NoRData(), NoRData()
#     end
#     return output_codual, svd_trunc_adjoint
# end
# function Mooncake.rrule!!(::CoDual{typeof(svd_trunc!)}, A_dA::CoDual, USVᴴ_dUSVᴴ::CoDual, alg_dalg::CoDual{<:TruncatedAlgorithm})
#     # unpack variables
#     A, dA = arrayify(A_dA)
#     USVᴴ_dUSVᴴ_arr = arrayify.(Mooncake.primal(USVᴴ_dUSVᴴ), Mooncake.tangent(USVᴴ_dUSVᴴ))
#     USVᴴ, dUSVᴴ = first.(USVᴴ_dUSVᴴ_arr), last.(USVᴴ_dUSVᴴ_arr)
#     alg = Mooncake.primal(alg_dalg)
#
#     # store state prior to primal call
#     Ac = copy(A)
#     USVᴴc = copy.(USVᴴ)
#
#     # compute primal - capture full USVᴴ and ind
#     USVᴴ = svd_compact!(A, USVᴴ, alg.alg)
#     USVᴴtrunc, ind = MatrixAlgebraKit.truncate(svd_trunc!, USVᴴ, alg.trunc)
#     ϵ = MatrixAlgebraKit.truncation_error(diagview(USVᴴ[2]), ind)
#
#     # pack output - note that we allocate new dUSVᴴtrunc because these aren't actually
#     # overwritten in the input!
#     USVᴴtrunc_dUSVᴴtrunc = Mooncake.zero_fcodual((USVᴴtrunc..., ϵ))
#
#     # define pullback
#     dUSVᴴtrunc = last.(arrayify.(USVᴴtrunc, Base.front(Mooncake.tangent(USVᴴtrunc_dUSVᴴtrunc))))
#     function svd_trunc_adjoint((_, _, _, dϵ)::Tuple{NoRData, NoRData, NoRData, Real})
#         _warn_pullback_truncerror(dϵ)
#
#         # compute pullbacks
#         svd_pullback!(dA, Ac, USVᴴ, dUSVᴴtrunc, ind)
#         zero!.(dUSVᴴtrunc) # since this is allocated in this function this is probably not required
#         zero!.(dUSVᴴ)
#
#         # restore state
#         copy!(A, Ac)
#         copy!.(USVᴴ, USVᴴc)
#
#         return ntuple(Returns(NoRData()), 4)
#     end
#
#     return USVᴴtrunc_dUSVᴴtrunc, svd_trunc_adjoint
# end
#
# @mark_primitive svd_trunc(::Any, ::MatrixAlgebraKit.AbstractAlgorithm)
# function Mooncake.rrule!!(::CoDual{typeof(svd_trunc)}, A_dA::CoDual, alg_dalg::CoDual)
#     # compute primal
#     A, dA = arrayify(A_dA)
#     alg = Mooncake.primal(alg_dalg)
#     output = svd_trunc(A, alg)
#     # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
#     # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
#     # pass). For many types this is done automatically when the forward step returns, but
#     # not for nested structs with various fields (like Diagonal{Complex})
#     output_codual = CoDual(output, Mooncake.fdata(Mooncake.zero_tangent(output)))
#     function svd_trunc_adjoint(dy::Tuple{NoRData, NoRData, NoRData, T}) where {T <: Real}
#         Utrunc, Strunc, Vᴴtrunc, ϵ = Mooncake.primal(output_codual)
#         dUtrunc_, dStrunc_, dVᴴtrunc_, dϵ = Mooncake.tangent(output_codual)
#         _warn_pullback_truncerror(dy[4])
#         U, dU = arrayify(Utrunc, dUtrunc_)
#         S, dS = arrayify(Strunc, dStrunc_)
#         Vᴴ, dVᴴ = arrayify(Vᴴtrunc, dVᴴtrunc_)
#         svd_trunc_pullback!(dA, A, (U, S, Vᴴ), (dU, dS, dVᴴ))
#         zero!(dU)
#         zero!(dS)
#         zero!(dVᴴ)
#         return NoRData(), NoRData(), NoRData()
#     end
#     return output_codual, svd_trunc_adjoint
# end
# function Mooncake.rrule!!(::CoDual{typeof(svd_trunc)}, A_dA::CoDual, alg_dalg::CoDual{<:TruncatedAlgorithm})
#     # unpack variables
#     A, dA = arrayify(A_dA)
#     alg = Mooncake.primal(alg_dalg)
#
#     # compute primal - capture full USVᴴ and ind
#     USVᴴ = svd_compact(A, alg.alg)
#     USVᴴtrunc, ind = MatrixAlgebraKit.truncate(svd_trunc!, USVᴴ, alg.trunc)
#     ϵ = MatrixAlgebraKit.truncation_error(diagview(USVᴴ[2]), ind)
#
#     # pack output
#     USVᴴtrunc_dUSVᴴtrunc = Mooncake.zero_fcodual((USVᴴtrunc..., ϵ))
#
#     # define pullback
#     dUSVᴴtrunc = last.(arrayify.(USVᴴtrunc, Base.front(Mooncake.tangent(USVᴴtrunc_dUSVᴴtrunc))))
#     function svd_trunc_adjoint((_, _, _, dϵ)::Tuple{NoRData, NoRData, NoRData, Real})
#         _warn_pullback_truncerror(dϵ)
#         svd_pullback!(dA, A, USVᴴ, dUSVᴴtrunc, ind)
#         zero!.(dUSVᴴtrunc) # since this is allocated in this function this is probably not required
#         return ntuple(Returns(NoRData()), 3)
#     end
#
#     return USVᴴtrunc_dUSVᴴtrunc, svd_trunc_adjoint
# end
#
# @mark_primitive svd_trunc_no_error!(::Any, ::Any, ::MatrixAlgebraKit.AbstractAlgorithm)
# function Mooncake.rrule!!(::CoDual{typeof(svd_trunc_no_error!)}, A_dA::CoDual, USVᴴ_dUSVᴴ::CoDual, alg_dalg::CoDual)
#     # compute primal
#     A, dA = arrayify(A_dA)
#     alg = Mooncake.primal(alg_dalg)
#     Ac = copy(A)
#     USVᴴ = Mooncake.primal(USVᴴ_dUSVᴴ)
#     dUSVᴴ = Mooncake.tangent(USVᴴ_dUSVᴴ)
#     U, dU = arrayify(USVᴴ[1], dUSVᴴ[1])
#     S, dS = arrayify(USVᴴ[2], dUSVᴴ[2])
#     Vᴴ, dVᴴ = arrayify(USVᴴ[3], dUSVᴴ[3])
#     USVᴴc = copy.(USVᴴ)
#     output = svd_trunc_no_error!(A, USVᴴ, alg)
#     # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
#     # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
#     # pass). For many types this is done automatically when the forward step returns, but
#     # not for nested structs with various fields (like Diagonal{Complex})
#     output_codual = CoDual(output, Mooncake.fdata(Mooncake.zero_tangent(output)))
#     function svd_trunc_adjoint(::NoRData)
#         copy!(A, Ac)
#         Utrunc, Strunc, Vᴴtrunc = Mooncake.primal(output_codual)
#         dUtrunc_, dStrunc_, dVᴴtrunc_ = Mooncake.tangent(output_codual)
#         U′, dU′ = arrayify(Utrunc, dUtrunc_)
#         S′, dS′ = arrayify(Strunc, dStrunc_)
#         Vᴴ′, dVᴴ′ = arrayify(Vᴴtrunc, dVᴴtrunc_)
#         svd_trunc_pullback!(dA, A, (U′, S′, Vᴴ′), (dU′, dS′, dVᴴ′))
#         copy!(U, USVᴴc[1])
#         copy!(S, USVᴴc[2])
#         copy!(Vᴴ, USVᴴc[3])
#         zero!(dU)
#         zero!(dS)
#         zero!(dVᴴ)
#         zero!(dU′)
#         zero!(dS′)
#         zero!(dVᴴ′)
#         return NoRData(), NoRData(), NoRData()
#     end
#     return output_codual, svd_trunc_adjoint
# end
# function Mooncake.rrule!!(::CoDual{typeof(svd_trunc_no_error!)}, A_dA::CoDual, USVᴴ_dUSVᴴ::CoDual, alg_dalg::CoDual{<:TruncatedAlgorithm})
#     # unpack variables
#     A, dA = arrayify(A_dA)
#     USVᴴ_dUSVᴴ_arr = arrayify.(Mooncake.primal(USVᴴ_dUSVᴴ), Mooncake.tangent(USVᴴ_dUSVᴴ))
#     USVᴴ, dUSVᴴ = first.(USVᴴ_dUSVᴴ_arr), last.(USVᴴ_dUSVᴴ_arr)
#     alg = Mooncake.primal(alg_dalg)
#
#     # store state prior to primal call
#     Ac = copy(A)
#     USVᴴc = copy.(USVᴴ)
#
#     # compute primal - capture full USVᴴ and ind
#     USVᴴ = svd_compact!(A, USVᴴ, alg.alg)
#     USVᴴtrunc, ind = MatrixAlgebraKit.truncate(svd_trunc!, USVᴴ, alg.trunc)
#
#     # pack output - note that we allocate new dUSVᴴtrunc because these aren't actually
#     # overwritten in the input!
#     USVᴴtrunc_dUSVᴴtrunc = Mooncake.zero_fcodual(USVᴴtrunc)
#
#     # define pullback
#     dUSVᴴtrunc = last.(arrayify.(USVᴴtrunc, Mooncake.tangent(USVᴴtrunc_dUSVᴴtrunc)))
#     function svd_trunc_adjoint(::NoRData)
#         # compute pullbacks
#         svd_pullback!(dA, Ac, USVᴴ, dUSVᴴtrunc, ind)
#         zero!.(dUSVᴴ)
#
#         # restore state
#         copy!(A, Ac)
#         copy!.(USVᴴ, USVᴴc)
#
#         return ntuple(Returns(NoRData()), 4)
#     end
#
#     return USVᴴtrunc_dUSVᴴtrunc, svd_trunc_adjoint
# end
#
# @mark_primitive svd_trunc_no_error(::Any, ::MatrixAlgebraKit.AbstractAlgorithm)
# function Mooncake.rrule!!(::CoDual{typeof(svd_trunc_no_error)}, A_dA::CoDual, alg_dalg::CoDual)
#     # compute primal
#     A, dA = arrayify(A_dA)
#     alg = Mooncake.primal(alg_dalg)
#     output = svd_trunc_no_error(A, alg)
#     # fdata call here is necessary to convert complicated Tangent type (e.g. of a Diagonal
#     # of ComplexF32) into the correct **forwards** data type (since we are now in the forward
#     # pass). For many types this is done automatically when the forward step returns, but
#     # not for nested structs with various fields (like Diagonal{Complex})
#     output_codual = CoDual(output, Mooncake.fdata(Mooncake.zero_tangent(output)))
#     function svd_trunc_adjoint(::NoRData)
#         Utrunc, Strunc, Vᴴtrunc = Mooncake.primal(output_codual)
#         dUtrunc_, dStrunc_, dVᴴtrunc_ = Mooncake.tangent(output_codual)
#         U, dU = arrayify(Utrunc, dUtrunc_)
#         S, dS = arrayify(Strunc, dStrunc_)
#         Vᴴ, dVᴴ = arrayify(Vᴴtrunc, dVᴴtrunc_)
#         svd_trunc_pullback!(dA, A, (U, S, Vᴴ), (dU, dS, dVᴴ))
#         zero!(dU)
#         zero!(dS)
#         zero!(dVᴴ)
#         return NoRData(), NoRData(), NoRData()
#     end
#     return output_codual, svd_trunc_adjoint
# end
# function Mooncake.rrule!!(::CoDual{typeof(svd_trunc_no_error)}, A_dA::CoDual, alg_dalg::CoDual{<:TruncatedAlgorithm})
#     # unpack variables
#     A, dA = arrayify(A_dA)
#     alg = Mooncake.primal(alg_dalg)
#
#     # compute primal - capture full USVᴴ and ind
#     USVᴴ = svd_compact(A, alg.alg)
#     USVᴴtrunc, ind = MatrixAlgebraKit.truncate(svd_trunc!, USVᴴ, alg.trunc)
#
#     # pack output
#     USVᴴtrunc_dUSVᴴtrunc = Mooncake.zero_fcodual(USVᴴtrunc)
#
#     # define pullback
#     dUSVᴴtrunc = last.(arrayify.(USVᴴtrunc, Mooncake.tangent(USVᴴtrunc_dUSVᴴtrunc)))
#     function svd_trunc_adjoint(::NoRData)
#         svd_pullback!(dA, A, USVᴴ, dUSVᴴtrunc, ind)
#         zero!.(dUSVᴴtrunc) # since this is allocated in this function this is probably not required
#         return ntuple(Returns(NoRData()), 3)
#     end
#
#     return USVᴴtrunc_dUSVᴴtrunc, svd_trunc_adjoint
# end

# single-output projections: project_hermitian!, project_antihermitian!
for (f!, f, adj) in (
        (:project_hermitian!, :project_hermitian, :project_hermitian_adjoint),
        (:project_antihermitian!, :project_antihermitian, :project_antihermitian_adjoint),
    )
    @eval begin
        @mark_primitive $f!(::Any, ::Any, ::MatrixAlgebraKit.AbstractAlgorithm)
        function Mooncake.rrule!!(f_df::CoDual{typeof($f!)}, A_dA::CoDual, arg_darg::CoDual, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm})
            A, dA = arrayify(A_dA)
            arg, darg = A_dA === arg_darg ? (A, dA) : arrayify(arg_darg)

            # don't need to copy/restore A since projections don't mutate input
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

        @mark_primitive $f(::Any, ::MatrixAlgebraKit.AbstractAlgorithm)
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
