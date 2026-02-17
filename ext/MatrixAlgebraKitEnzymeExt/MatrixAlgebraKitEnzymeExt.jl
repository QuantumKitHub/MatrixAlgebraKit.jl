module MatrixAlgebraKitEnzymeExt

using MatrixAlgebraKit
using MatrixAlgebraKit: copy_input, initialize_output, zero!
using MatrixAlgebraKit: diagview, inv_safe, truncate
using MatrixAlgebraKit: qr_pullback!, lq_pullback!
using MatrixAlgebraKit: qr_null_pullback!, lq_null_pullback!
using MatrixAlgebraKit: eig_pullback!, eigh_pullback!, eig_vals_pullback!, eigh_vals_pullback!
using MatrixAlgebraKit: svd_pullback!, svd_vals_pullback!
using MatrixAlgebraKit: left_polar_pullback!, right_polar_pullback!
using Enzyme
using Enzyme.EnzymeCore
using Enzyme.EnzymeCore: EnzymeRules
using LinearAlgebra

@inline EnzymeRules.inactive_type(::Type{Alg}) where {Alg <: MatrixAlgebraKit.AbstractAlgorithm} = true
@inline EnzymeRules.inactive_type(::Type{TS}) where {TS <: MatrixAlgebraKit.TruncationStrategy} = true
@inline EnzymeRules.inactive(f::typeof(MatrixAlgebraKit.select_algorithm), func::F, A::AbstractMatrix, alg::Alg) where {F, Alg} = true
@inline EnzymeRules.inactive(f::typeof(MatrixAlgebraKit.default_algorithm), func::F, A::AbstractMatrix) where {F} = true
@inline EnzymeRules.inactive(f::typeof(MatrixAlgebraKit.check_input), func::F, A::AbstractMatrix, alg::Alg) where {F, Alg} = true
@inline EnzymeRules.inactive(::typeof(MatrixAlgebraKit.defaulttol), ::Any) = true
@inline EnzymeRules.inactive(::typeof(MatrixAlgebraKit.default_pullback_gauge_atol), ::Any) = true
@inline EnzymeRules.inactive(::typeof(MatrixAlgebraKit.default_pullback_gauge_atol), ::Any, ::Any...) = true
@inline EnzymeRules.inactive(::typeof(MatrixAlgebraKit.default_pullback_degeneracy_atol), ::Any) = true
@inline EnzymeRules.inactive(::typeof(MatrixAlgebraKit.default_pullback_rank_atol), ::Any) = true
@inline EnzymeRules.inactive(::typeof(MatrixAlgebraKit.default_hermitian_tol), ::Any) = true

#----------- NOTE about derivatives ---------
# Each Enzyme augmented_return + reverse pair
# has a "tape" or "cache" -- we can place
# variables on this tape that can be accessed
# in the return pass *after* they have been
# "filled in" with accumulated derivatives.
# For many of the rules here, we may create a
# placeholder (usually called `dret`) for
# variables which may be instantiated, then,
# earlier in the reverse pass, this `dret` is
# filled in with accumulated derivatives for
# the created variable. It can then be used
# to update the derivative of `A` or any
# other provided input variable.
#--------------------------------------------

# this rule is necessary for now as without it,
# a segfault occurs both on 1.10 and 1.12 -- likely
# a deeper internal bug
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(copy_input)},
        ::Type{RT},
        f::Annotation,
        A::Annotation
    ) where {RT}
    ret = func.val(f.val, A.val)
    primal = EnzymeRules.needs_primal(config) ? ret : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(A.dval) : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, shadow)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(copy_input)},
        ::Type{RT},
        cache,
        f::Annotation,
        A::Annotation
    ) where {RT}
    copy_shadow = cache
    if !isa(A, Const) && !isnothing(copy_shadow)
        A.dval .+= copy_shadow
    end
    return (nothing, nothing)
end

# two-argument factorizations like LQ, QR, EIG
for (f, pb) in (
        (qr_full!, qr_pullback!),
        (lq_full!, lq_pullback!),
        (qr_compact!, qr_pullback!),
        (lq_compact!, lq_pullback!),
        (eig_full!, eig_pullback!),
        (eigh_full!, eigh_pullback!),
        (left_polar!, left_polar_pullback!),
        (right_polar!, right_polar_pullback!),
    )
    @eval begin
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f)},
                ::Type{RT},
                A::Annotation,
                arg::Annotation{Tuple{TA, TB}},
                alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
            ) where {RT, TA, TB}
            # A is overwritten in the primal, but NOT used in the pullback,
            # so we do not need to cache it. This may change if future pullbacks
            # depend directly on A!
            ret = func.val(A.val, arg.val, alg.val)
            # if arg.val == ret, the annotation must be Duplicated or DuplicatedNoNeed
            # if arg isa Const, ret may still be modified further down the call graph so we should
            # copy it to protect ourselves
            cache_arg = (arg.val !== ret) || EnzymeRules.overwritten(config)[3] ? copy.(ret) : nothing
            dret = if EnzymeRules.needs_shadow(config)
                (TA == Nothing && TB == Nothing) || isa(arg, Const) ? zero.(ret) : arg.dval
            else
                nothing
            end
            primal = EnzymeRules.needs_primal(config) ? ret : nothing
            return EnzymeRules.AugmentedReturn(primal, dret, (cache_arg, dret))
        end
        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f)},
                ::Type{RT},
                cache,
                A::Annotation,
                arg::Annotation{Tuple{TA, TB}},
                alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
            ) where {RT, TA, TB}
            cache_arg, darg = cache
            # A is  NOT used in the pullback, so we assign Aval = nothing
            # to trigger an error in case the pullback is modified to directly
            # use A (so that whoever does this is forced to handle caching A
            # appropriately here)
            Aval = nothing
            argval = something(cache_arg, arg.val)
            if !isa(A, Const)
                $pb(A.dval, Aval, argval, darg)
            end
            !isa(arg, Const) && make_zero!(arg.dval)
            return (nothing, nothing, nothing)
        end
    end
end

for (f, pb) in (
        (qr_null!, qr_null_pullback!),
        (lq_null!, lq_null_pullback!),
    )
    @eval begin
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f)},
                ::Type{RT},
                A::Annotation,
                arg::Annotation{TA},
                alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
            ) where {RT, TA}
            # here, A IS directly used in the pullback, and overwritten
            # in the forward primal call, so we MUST cache its value
            cache_A = copy(A.val)
            ret = func.val(A.val, arg.val, alg.val)
            # if arg.val == ret, the annotation must be Duplicated or DuplicatedNoNeed
            # if arg isa Const, ret may still be modified further down the call graph so we should
            # copy it to protect ourselves
            cache_arg = (arg.val !== ret) || EnzymeRules.overwritten(config)[3] ? copy(ret) : nothing
            # on 1.10, Enzyme can get confused about whether it needs the shadow
            if EnzymeRules.needs_shadow(config)
                dret = (TA == Nothing || isa(arg, Const)) ? zero(ret) : arg.dval
            else
                dret = nothing
            end
            primal = EnzymeRules.needs_primal(config) ? ret : nothing
            return EnzymeRules.AugmentedReturn(primal, dret, (cache_A, cache_arg, dret))
        end
        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f)},
                ::Type{RT},
                cache,
                A::Annotation,
                arg::Annotation,
                alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
            ) where {RT}
            cache_A, cache_arg, darg = cache
            Aval = cache_A
            argval = something(cache_arg, arg.val)
            # on 1.10, Enzyme can get confused about whether it needs the shadow
            # replace the dret with the arg.dval in the case it's nothing
            argdval = something(darg, arg.dval)
            if !isa(A, Const)
                $pb(A.dval, Aval, argval, argdval)
            end
            !isa(arg, Const) && make_zero!(arg.dval)
            return (nothing, nothing, nothing)
        end
    end
end

for f in (:svd_compact!, :svd_full!)
    @eval begin
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f)},
                ::Type{RT},
                A::Annotation,
                USVᴴ::Annotation{Tuple{TU, TS, TVᴴ}},
                alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
            ) where {RT, TU, TS, TVᴴ}
            # A is overwritten in the primal, but NOT used in the pullback,
            # so we do not need to cache it. This may change if future pullbacks
            # depend directly on A!
            ret = func.val(A.val, USVᴴ.val, alg.val)
            # if USVᴴ.val == ret, the annotation must be Duplicated or DuplicatedNoNeed
            # if USVᴴ isa Const, ret may still be modified further down the call graph so we should
            # copy it to protect ourselves
            cache_USVᴴ = (USVᴴ.val !== ret) || EnzymeRules.overwritten(config)[3] ? copy.(ret) : nothing
            # the USVᴴ may be nothing for eltypes handled by GenericLinearAlgebra
            dret = if EnzymeRules.needs_shadow(config) && ((TU == TS == TVᴴ == Nothing) || isa(USVᴴ, Const))
                dU = zero(ret[1])
                # special casing `Diagonal` seems to be necessary due to Enzyme's type analysis
                dS = $(f == svd_compact!) ? Diagonal(zero(ret[2].diag)) : zero(ret[2])
                dVᴴ = zero(ret[3])
                (dU, dS, dVᴴ)
            elseif EnzymeRules.needs_shadow(config)
                USVᴴ.dval
            else
                nothing
            end
            primal = EnzymeRules.needs_primal(config) ? ret : nothing
            return EnzymeRules.AugmentedReturn(primal, dret, (cache_USVᴴ, dret))
        end
        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f)},
                ::Type{RT},
                cache,
                A::Annotation,
                USVᴴ::Annotation,
                alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
            ) where {RT}
            cache_USVᴴ, dUSVᴴ = cache
            # A is  NOT used in the pullback, so we assign Aval = nothing
            # to trigger an error in case the pullback is modified to directly
            # use A (so that whoever does this is forced to handle caching A
            # appropriately here)
            Aval = nothing
            USVᴴval = something(cache_USVᴴ, USVᴴ.val)
            if !isa(A, Const)
                minmn = min(size(A.val)...)
                if $(f == svd_compact!) # compact
                    svd_pullback!(A.dval, Aval, USVᴴval, dUSVᴴ)
                else # full
                    # TODO: revisit this once `svd_pullback` supports `svd_full` output and adjoints
                    U, S, Vᴴ = USVᴴval
                    vU = view(U, :, 1:minmn)
                    vS = Diagonal(view(diagview(S), 1:minmn))
                    vVᴴ = view(Vᴴ, 1:minmn, :)
                    vdU = view(dUSVᴴ[1], :, 1:minmn)
                    vdS = Diagonal(view(diagview(dUSVᴴ[2]), 1:minmn))
                    vdVᴴ = view(dUSVᴴ[3], 1:minmn, :)
                    svd_pullback!(A.dval, Aval, (vU, vS, vVᴴ), (vdU, vdS, vdVᴴ))
                end
            end
            !isa(USVᴴ, Const) && make_zero!(USVᴴ.dval)
            return (nothing, nothing, nothing)
        end
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(svd_trunc_no_error!)},
        ::Type{RT},
        A::Annotation,
        USVᴴ::Annotation{Tuple{TU, TS, TVᴴ}},
        alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
    ) where {RT, TU, TS, TVᴴ}
    # A is overwritten in the primal, but NOT used in the pullback,
    # so we do not need to cache it. This may change if future pullbacks
    # depend directly on A!
    ret = svd_compact!(A.val, USVᴴ.val, alg.val.alg)
    cache_USVᴴ = (USVᴴ.val !== ret) || EnzymeRules.overwritten(config)[3] ? copy.(ret) : nothing
    USVᴴ′, ind = MatrixAlgebraKit.truncate(svd_trunc!, ret, alg.val.trunc)
    primal = EnzymeRules.needs_primal(config) ? USVᴴ′ : nothing
    # This creates new output shadow matrices, we use USVᴴ′ to ensure the
    # eltypes and dimensions are correct.
    # These new shadow matrices are "filled in" with the accumulated
    # results from earlier in reverse-mode AD after this function exits
    # and before `reverse` is called.
    dret = if EnzymeRules.needs_shadow(config)
        (zero(USVᴴ′[1]), Diagonal(zero(USVᴴ′[2].diag)), zero(USVᴴ′[3]))
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn(primal, dret, (cache_USVᴴ, dret, ind))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(svd_trunc_no_error!)},
        ::Type{RT},
        cache,
        A::Annotation,
        USVᴴ::Annotation,
        alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
    ) where {RT}
    cache_USVᴴ, dUSVᴴ, ind = cache
    # A is  NOT used in the pullback, so we assign Aval = nothing
    # to trigger an error in case the pullback is modified to directly
    # use A (so that whoever does this is forced to handle caching A
    # appropriately here)
    Aval = nothing
    USVᴴval = something(cache_USVᴴ, USVᴴ.val)
    if !isa(A, Const)
        svd_pullback!(A.dval, Aval, USVᴴval, dUSVᴴ, ind)
    end
    !isa(USVᴴ, Const) && make_zero!(USVᴴ.dval)
    return (nothing, nothing, nothing)
end

for (f, trunc_f, full_f, pb) in (
        (:eigh_trunc_no_error!, :eigh_trunc!, :eigh_full!, :eigh_pullback!),
        (:eig_trunc_no_error!, :eig_trunc!, :eig_full!, :eig_pullback!),
    )
    @eval function EnzymeRules.augmented_primal(
            config::EnzymeRules.RevConfigWidth{1},
            func::Const{typeof($f)},
            ::Type{RT},
            A::Annotation,
            DV::Annotation{Tuple{TA, TB}},
            alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
        ) where {RT, TA, TB}
        # A is overwritten in the primal, but NOT used in the pullback,
        # so we do not need to cache it. This may change if future pullbacks
        # depend directly on A!
        ret = $full_f(A.val, DV.val, alg.val.alg)
        cache_DV = (DV.val !== ret) || EnzymeRules.overwritten(config)[3] ? copy.(ret) : nothing
        DV′, ind = truncate($trunc_f, ret, alg.val.trunc)
        primal = EnzymeRules.needs_primal(config) ? DV′ : nothing
        dret = if EnzymeRules.needs_shadow(config)
            (Diagonal(zero(diagview(DV′[1]))), zero(DV′[2]))
        else
            nothing
        end
        return EnzymeRules.AugmentedReturn(primal, dret, (cache_DV, dret, ind))
    end
    @eval function EnzymeRules.reverse(
            config::EnzymeRules.RevConfigWidth{1},
            func::Const{typeof($f)},
            ::Type{RT},
            cache,
            A::Annotation,
            DV::Annotation,
            alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
        ) where {RT}
        cache_DV, dDVtrunc, ind = cache
        # A is  NOT used in the pullback, so we assign Aval = nothing
        # to trigger an error in case the pullback is modified to directly
        # use A (so that whoever does this is forced to handle caching A
        # appropriately here)
        Aval = nothing
        DVval = something(cache_DV, DV.val)
        if !isa(A, Const)
            $pb(A.dval, Aval, DVval, dDVtrunc, ind)
        end
        !isa(DV, Const) && make_zero!(DV.dval)
        return (nothing, nothing, nothing)
    end
end

for (f!, f_full!, pb!) in (
        (eig_vals!, eig_full!, eig_vals_pullback!),
        (eigh_vals!, eigh_full!, eigh_vals_pullback!),
    )
    @eval begin
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f!)},
                ::Type{RT},
                A::Annotation,
                D::Annotation{TD},
                alg::Annotation{<:MatrixAlgebraKit.AbstractAlgorithm},
            ) where {RT, TD}
            # A is overwritten in the primal, but NOT used in the pullback,
            # so we do not need to cache it. This may change if future pullbacks
            # depend directly on A!
            nD, V = $f_full!(A.val, alg.val)
            ret = TD == Nothing ? diagview(nD) : copy!(D.val, diagview(nD))
            cache_D = (D.val !== ret) || EnzymeRules.overwritten(config)[3] ? copy(ret) : nothing
            primal = EnzymeRules.needs_primal(config) ? ret : nothing
            # on 1.10, Enzyme can get confused about whether it needs the shadow
            # create dret no matter what to account for this
            dret = TD == Nothing || isa(D, Const) ? zero(ret) : D.dval
            shadow = EnzymeRules.needs_shadow(config) ? dret : nothing
            return EnzymeRules.AugmentedReturn(primal, shadow, (cache_D, dret, V))
        end
        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f!)},
                ::Type{RT},
                cache,
                A::Annotation,
                D::Annotation,
                alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
            ) where {RT}
            cache_D, dD, V = cache
            Dval = something(cache_D, D.val)
            # A is  NOT used in the pullback, so we assign Aval = nothing
            # to trigger an error in case the pullback is modified to directly
            # use A (so that whoever does this is forced to handle caching A
            # appropriately here)
            Aval = nothing
            if !isa(A, Const)
                $pb!(A.dval, Aval, (Diagonal(Dval), V), dD)
            end
            !isa(D, Const) && make_zero!(D.dval)
            return (nothing, nothing, nothing)
        end
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(svd_vals!)},
        ::Type{RT},
        A::Annotation,
        S::Annotation{TS},
        alg::Annotation{<:MatrixAlgebraKit.AbstractAlgorithm},
    ) where {RT, TS}
    # A is overwritten in the primal, but NOT used in the pullback,
    # so we do not need to cache it. This may change if future pullbacks
    # depend directly on A!
    U, nS, Vᴴ = svd_compact!(A.val, alg.val)
    ret = TS == Nothing ? diagview(nS) : copy!(S.val, diagview(nS))
    cache_S = (S.val !== ret) || EnzymeRules.overwritten(config)[3] ? copy(ret) : nothing
    primal = EnzymeRules.needs_primal(config) ? ret : nothing
    # on 1.10, Enzyme can get confused about whether it needs the shadow
    # create dret no matter what to account for this
    dret = TS == Nothing || isa(S, Const) ? zero(ret) : S.dval
    shadow = EnzymeRules.needs_shadow(config) ? dret : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_S, dret, U, Vᴴ))
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(svd_vals!)},
        ::Type{RT},
        cache,
        A::Annotation,
        S::Annotation,
        alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
    ) where {RT}
    cache_S, dS, U, Vᴴ = cache
    # A is  NOT used in the pullback, so we assign Aval = nothing
    # to trigger an error in case the pullback is modified to directly
    # use A (so that whoever does this is forced to handle caching A
    # appropriately here)
    Aval = nothing
    Sval = something(cache_S, S.val)
    if !isa(A, Const)
        svd_vals_pullback!(A.dval, Aval, (U, Diagonal(Sval), Vᴴ), dS)
    end
    !isa(S, Const) && make_zero!(S.dval)
    return (nothing, nothing, nothing)
end

end
