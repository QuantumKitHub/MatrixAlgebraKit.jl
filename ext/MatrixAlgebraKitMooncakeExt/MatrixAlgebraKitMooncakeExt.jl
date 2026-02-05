module MatrixAlgebraKitMooncakeExt

using Mooncake
using Mooncake: CoDual, Dual, NoRData, arrayify, primal, tangent, zero_fcodual
import Mooncake: rrule!!
using MatrixAlgebraKit
using MatrixAlgebraKit: MatrixAlgebraKit as MAK, diagview, zero!, AbstractAlgorithm, TruncatedAlgorithm
using LinearAlgebra


# Utility
# -------
# convenience helper for marking DefaultCtx ReverseMode signature as primitive
macro is_rev_primitive(sig)
    return esc(:(Mooncake.@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode $sig))
end
_warn_pullback_truncerror(dϵ::Real; tol = MatrixAlgebraKit.defaulttol(dϵ)) =
    abs(dϵ) ≤ tol || @warn "Pullback ignores non-zero tangents for truncation error"

# No derivatives
# --------------
Mooncake.tangent_type(::Type{<:AbstractAlgorithm}) = Mooncake.NoTangent

Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(MAK.initialize_output), Vararg{Any}}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(MAK.check_input), Vararg{Any}}


@is_rev_primitive Tuple{typeof(copy_input), Any, Any}
function rrule!!(::CoDual{typeof(copy_input)}, f_df::CoDual, A_dA::CoDual)
    Ac = copy_input(primal(f_df), primal(A_dA))
    Ac_dAc = zero_fcodual(Ac)
    dAc = tangent(Ac_dAc)
    function copy_input_pb(::NoRData)
        Mooncake.increment!!(tangent(A_dA), dAc)
        return NoRData()
    end
    return Ac_dAc, copy_input_pb
end

# Factorizations
# --------------
for (f, pullback!, adjoint) in (
        (:qr_full, :qr_pullback!, :qr_adjoint),
        (:lq_full, :lq_pullback!, :lq_adjoint),
        (:qr_compact, :qr_pullback!, :qr_adjoint),
        (:lq_compact, :lq_pullback!, :lq_adjoint),
        (:eig_full, :eig_pullback!, :eig_adjoint),
        (:eig_trunc_no_error, :eig_trunc_pullback!, :eig_adjoint),
        (:eigh_full, :eigh_pullback!, :eigh_adjoint),
        (:eigh_trunc_no_error, :eigh_trunc_pullback!, :eigh_adjoint),
        (:left_polar, :left_polar_pullback!, :left_polar_adjoint),
        (:right_polar, :right_polar_pullback!, :right_polar_adjoint),
        (:svd_compact, :svd_pullback!, :svd_adjoint),
        (:svd_full, :svd_pullback!, :svd_adjoint),
        (:svd_trunc_no_error, :svd_trunc_pullback!, :svd_adjoint),
    )
    f! = Symbol(f, :!)

    @eval begin
        @is_rev_primitive Tuple{typeof($f), Any, AbstractAlgorithm}
        function rrule!!(::CoDual{typeof($f)}, A_dA::CoDual, alg_dalg::CoDual{<:AbstractAlgorithm})
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
                return NoRData()
            end

            return args, $adjoint
        end

        @is_rev_primitive Tuple{typeof($f!), Any, Tuple, AbstractAlgorithm}
        rrule!!(::CoDual{typeof($f!)}, A_dA::CoDual, args_dargs::CoDual, alg_dalg::CoDual{<:AbstractAlgorithm}) =
            rrule!!(zero_fcodual($f), A_dA, alg_dalg)
    end
end

# Nullspaces
# ----------
for (f, pullback, adjoint) in (
        (:qr_null, :qr_null_pullback!, :qr_null_adjoint),
        (:lq_null, :lq_null_pullback!, :lq_null_adjoint),
    )
    f! = Symbol(f, :!)

    @eval begin
        @is_rev_primitive Tuple{typeof($f), Any, AbstractAlgorithm}
        function rrule!!(::CoDual{typeof($f)}, A_dA::CoDual, alg_dalg::CoDual{<:AbstractAlgorithm})
            # unpack variables
            A, dA = arrayify(A_dA)
            alg = primal(alg_dalg)

            # compute primal and pack output
            N = $f(A, alg)
            N_dN = zero_fcodual(N)

            # define pullback
            dN = last(arrayify(N, tangent(N_dN)))
            function $adjoint(::NoRData)
                MAK.$pullback!(dA, A, N, dN)
                return NoRData()
            end

            return N, $adjoint
        end

        @is_rev_primitive Tuple{typeof($f!), Any, Any, AbstractAlgorithm}
        rrule!!(::CoDual{typeof($f!)}, A_dA::CoDual, N_dN::CoDual, alg_dalg::CoDual{<:AbstractAlgorithm}) =
            rrule!!(zero_fcodual($f), A_dA, alg_dalg)
    end
end

for f in (:eig, :eigh, :svd)
    f_vals = Symbol(f, :_vals)
    f_vals! = Symbol(f_vals, :!)
    f_full = Symbol(f, :_full)
    vals_pulback! = Symbol(f, :_vals_pullback!)
    adjoint! = Symbol(f, :_adjoint)

    # f_values
    # --------
    @eval begin
        @is_rev_primitive Tuple{typeof($f_vals), Any, AbstractAlgorithm}
        function rrule!!(::CoDual{typeof($f_vals)}, A_dA::CoDual, alg_dalg::CoDual)
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
                return NoRData()
            end

            return vals_dvals, $adjoint
        end

        @is_rev_primitive Tuple{typeof($f_vals!), Any, Any, AbstractAlgorithm}
        rrule!!(::CoDual{typeof($f_vals!)}, A_dA::CoDual, D_dD::CoDual, alg_dalg::CoDual) =
            rrule!!(zero_fcodual($f_vals), A_dA, alg_dalg)
    end


    # Truncated decompositions
    # ------------------------
    f_trunc = Symbol(f, :_trunc)
    f_trunc! = Symbol(f_trunc, :!)
    pullback! = Symbol(f, :_pullback!)
    trunc_pullback! = Symbol(f_trunc, :_pullback!)

    @eval begin
        @is_rev_primitive Tuple{typeof($f_trunc), Any, AbstractAlgorithm}
        function rrule!!(::CoDual{typeof($f_trunc)}, A_dA::CoDual, alg_dalg::CoDual)
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
                return NoRData()
            end

            return argsϵ_dargsϵ, $adjoint
        end
        function rrule!!(::CoDual{typeof($f_trunc)}, A_dA::CoDual, alg_dalg::CoDual{<:TruncatedAlgorithm})
            # unpack variables
            A, dA = arrayify(A_dA)
            alg = Mooncake.primal(alg_dalg)

            # compute primal and pack output - capture full DV and ind
            args_full = $f_full(A, alg.alg)
            args, ind = MAK.truncate($f_trunc!, args_full, alg.trunc)
            ϵ = MAK.truncation_error(diagview(args[1]), ind)
            argsϵ = (args..., ϵ)
            argsϵ_dargsϵ = zero_fcodual(argsϵ)

            # define pullback
            dargs = last.(arrayify.(args, Base.front(tangent(argsϵ_dargsϵ))))
            function $f_adjoint!(dy)
                _warn_pullback_truncerror(last(dy))
                MAK.$pullback!(dA, A, args_full, dargs, ind)
                return NoRData()
            end

            return DVtrunc_dDVtrunc, $f_adjoint!
        end
        @is_rev_primitive Tuple{typeof($f_trunc!), Any, Any, AbstractAlgorithm}
        rrule!!(::CoDual{typeof($f_trunc!)}, A_dA::CoDual, args_dargs::CoDual, alg_dalg::CoDual) =
            rrule!!(zero_fcodual($f_trunc), A_dA, alg_dalg)
    end

    # Truncated decompositions - no error
    # -----------------------------------
    f_trunc_no_error = Symbol(f_trunc, :_no_error)
    f_trunc_no_error! = Symbol(f_trunc_no_error, :!)

    @eval begin
        @is_rev_primitive Tuple{typeof($f_trunc_no_error), Any, AbstractAlgorithm}
        function rrule!!(::CoDual{typeof($f_trunc_no_error)}, A_dA::CoDual, alg_dalg::CoDual)
            # unpack variables
            A, dA = arrayify(A_dA)
            alg = primal(alg_dalg)

            # compute primal and pack output
            args = $f_trunc(A, alg)
            args_dargs = zero_fcodual(args)

            # define pullback
            dargs = last.(arrayify.(args, tangent(args_dargs)))
            function $adjoint(::NoRData)
                MAK.$trunc_pullback!(dA, A, args, dargs)
                return NoRData()
            end

            return args_dargs, $adjoint
        end
        function rrule!!(::CoDual{typeof($f_trunc_no_error)}, A_dA::CoDual, alg_dalg::CoDual{<:TruncatedAlgorithm})
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
                return NoRData()
            end

            return args_dargs, $adjoint
        end

        @is_rev_primitive Tuple{typeof($f_trunc_no_error!), Any, Any, AbstractAlgorithm}
        rrule!!(::CoDual{typeof($f_trunc_no_error!)}, A_dA::CoDual, args_dargs::CoDual, alg_dalg::CoDual) =
            rrule!!(zero_fcodual($f_trunc_no_error), A_dA, alg_dalg)
    end
end

end
