module MatrixAlgebraKitEnzymeExt

using MatrixAlgebraKit
using MatrixAlgebraKit: copy_input
using MatrixAlgebraKit: diagview, inv_safe, eig_trunc!, eigh_trunc!
using MatrixAlgebraKit: qr_pullback!, lq_pullback!
using MatrixAlgebraKit: qr_null_pullback!, lq_null_pullback!
using MatrixAlgebraKit: eig_pullback!, eigh_pullback!, eig_trunc_pullback!, eigh_trunc_pullback!
using MatrixAlgebraKit: svd_pullback!
using MatrixAlgebraKit: left_polar_pullback!, right_polar_pullback!
using Enzyme
using Enzyme.EnzymeCore
using Enzyme.EnzymeCore: EnzymeRules
using LinearAlgebra

@inline EnzymeRules.inactive_type(v::Type{<:MatrixAlgebraKit.AbstractAlgorithm}) = true

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(copy_input)},
        ::Type{RT},
        f::Annotation,
        A::Annotation
    ) where {RT}
    func.val(f.val, A.val)
    primal = EnzymeRules.needs_primal(config) ? copy(A.val) : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(A.dval) : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, shadow)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(copy_input)},
        dret::Type{RT},
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
            cache_arg = nothing
            # form cache if needed
            cache_A = !(typeof(arg) <: Const) ? copy(A.val) : nothing
            func.val(A.val, arg.val, alg.val)
            primal = EnzymeRules.needs_primal(config) ? arg.val : nothing
            shadow = EnzymeRules.needs_shadow(config) ? arg.dval : nothing
            return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_arg))
        end
        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f)},
                dret::Type{RT},
                cache,
                A::Annotation,
                arg::Annotation{Tuple{TA, TB}},
                alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
            ) where {RT, TA, TB}
            cache_A, cache_arg = cache
            argval = arg.val
            Aval = !isnothing(cache_A) ? cache_A : A.val
            ∂arg = isa(arg, Const) ? nothing : arg.dval
            if !isa(A, Const) && !isa(arg, Const)
                $pb(A.dval, Aval, argval, ∂arg)
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
                arg::Annotation,
                alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
            ) where {RT}
            cache_A = copy(A.val)
            func.val(A.val, arg.val, alg.val)
            primal = EnzymeRules.needs_primal(config) ? arg.val : nothing
            shadow = EnzymeRules.needs_shadow(config) ? arg.dval : nothing
            return EnzymeRules.AugmentedReturn(primal, shadow, cache_A)
        end

        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f)},
                dret::Type{RT},
                cache,
                A::Annotation,
                arg::Annotation,
                alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
            ) where {RT}
            cache_A = cache
            Aval = isnothing(cache_A) ? A.val : cache_A
            if !isa(A, Const) && !isa(arg, Const)
                $pb(A.dval, Aval, arg.val, arg.dval)
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
                USVᴴ::Annotation,
                alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
            ) where {RT}
            # form cache if needed
            cache_USVᴴ = (EnzymeRules.overwritten(config)[3] && !(typeof(USVᴴ) <: Const)) ? copy(USVᴴ.val) : nothing
            cache_A = !(typeof(A) <: Const) ? copy(A.val) : nothing
            func.val(A.val, USVᴴ.val, alg.val)
            primal = EnzymeRules.needs_primal(config) ? USVᴴ.val : nothing
            shadow = EnzymeRules.needs_shadow(config) ? USVᴴ.dval : nothing
            return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_USVᴴ))
        end
        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f)},
                dret::Type{RT},
                cache,
                A::Annotation,
                USVᴴ::Annotation,
                alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
            ) where {RT}
            cache_A, cache_USVᴴ = cache
            Aval = isnothing(cache_A) ? A.val : cache_A
            USVᴴval = !isnothing(cache_USVᴴ) ? cache_USVᴴ : USVᴴ.val
            U, S, Vᴴ = USVᴴval
            ∂USVᴴ = isa(USVᴴ, Const) ? nothing : USVᴴ.dval
            if !isa(A, Const) && !isa(USVᴴ, Const)
                minmn = min(size(A.val)...)
                if $(f == svd_compact!) # compact
                    svd_pullback!(A.dval, Aval, USVᴴval, ∂USVᴴ)
                else # full
                    vU = view(U, :, 1:minmn)
                    vS = Diagonal(diagview(S)[1:minmn])
                    vVᴴ = view(Vᴴ, 1:minmn, :)
                    vdU = view(∂USVᴴ[1], :, 1:minmn)
                    vdS = Diagonal(diagview(∂USVᴴ[2])[1:minmn])
                    vdVᴴ = view(∂USVᴴ[3], 1:minmn, :)
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
        func::Const{typeof(svd_trunc!)},
        ::Type{RT},
        A::Annotation,
        USVᴴ::Annotation,
        alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
    ) where {RT}
    # form cache if needed
    cache_A = copy(A.val)
    svd_compact!(A.val, USVᴴ.val, alg.val.alg)
    cache_USVᴴ = copy.(USVᴴ.val)
    USVᴴ′, ind = MatrixAlgebraKit.truncate(svd_trunc!, USVᴴ.val, alg.val.trunc)
    ϵ.val      = MatrixAlgebraKit.truncation_error!(diagview(USVᴴ.val[2]), ind)
    primal     = EnzymeRules.needs_primal(config) ? (USVᴴ′..., ϵ.val) : nothing
    shadow_USVᴴ = if !isa(A, Const) && !isa(USVᴴ, Const)
        dU, dS, dVᴴ = USVᴴ.dval
        # This creates new output shadow matrices, we do this slicing
        # to ensure they have the correct eltype and dimensions.
        # These new shadow matrices are "filled in" with the accumulated
        # results from earlier in reverse-mode AD after this function exits
        # and before `reverse` is called.
        dStrunc  = Diagonal(diagview(dS)[ind])
        dUtrunc  = dU[:, ind]
        dVᴴtrunc = dVᴴ[ind, :]
        (dUtrunc, dStrunc, dVᴴtrunc)
    else
        (nothing, nothing, nothing)
    end
    shadow = EnzymeRules.needs_shadow(config) ? (shadow_USVᴴ..., ϵ.dval) : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_USVᴴ, shadow_USVᴴ, ind))
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(svd_trunc!)},
        dret::Type{RT},
        cache,
        A::Annotation,
        USVᴴ::Annotation,
        alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
    ) where {RT}
    cache_A, cache_USVᴴ, shadow_USVᴴ, ind = cache
    U, S, Vᴴ    = cache_USVᴴ
    dU, dS, dVᴴ = shadow_USVᴴ
    Aval        = isnothing(cache_A) ? A.val : cache_A
    if !isa(A, Const) && !isa(USVᴴ, Const)
        svd_pullback!(A.dval, Aval, (U, S, Vᴴ), shadow_USVᴴ, ind)
    end
    !isa(USVᴴ, Const) && make_zero!(USVᴴ.dval)
    !isa(ϵ, Const) && make_zero!(ϵ.dval)
    return (nothing, nothing, nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(svd_trunc)},
        ::Type{MixedDuplicated},
        A::Annotation,
        alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
    )
    # form cache if needed
    cache_A     = copy(A.val)
    U, S, Vᴴ, ϵ = svd_trunc(A.val, USVᴴ.val, alg.val.alg)
    primal      = EnzymeRules.needs_primal(config) ? (U, S, Vᴴ, ϵ) : nothing
    dU  = zero(U)
    dS  = zero(S)
    dVᴴ = zero(Vᴴ)
    dϵ  = zero(ϵ)
    shadow      = EnzymeRules.needs_shadow(config) ? (dU, dS, dVᴴ, dϵ) : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, (U, S, Vᴴ), (dU, dS, dVᴴ)))
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(svd_trunc)},
        dret::Type{MixedDuplicated},
        cache,
        A::Annotation,
        alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
    )
    cache_A, cache_USVᴴ, shadow_USVᴴ = cache
    U, S, Vᴴ    = cache_USVᴴ
    dU, dS, dVᴴ = shadow_USVᴴ
    Aval        = isnothing(cache_A) ? A.val : cache_A
    if !isa(A, Const) && !isa(USVᴴ, Const)
        svd_trunc_pullback!(A.dval, Aval, (U, S, Vᴴ), shadow_USVᴴ, ind)
    end
    return (nothing, nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(eigh_trunc!)},
        ::Type{RT},
        A::Annotation,
        DV::Annotation{Tuple{TD, TV}},
        alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
    ) where {RT, TD, TV}
    # form cache if needed
    cache_A = copy(A.val)
    MatrixAlgebraKit.eigh_full!(A.val, DV.val, alg.val.alg)
    cache_DV = copy.(DV.val)
    DV′, ind = MatrixAlgebraKit.truncate(eigh_trunc!, DV.val, alg.val.trunc)
    ϵ.val = MatrixAlgebraKit.truncation_error!(diagview(DV.val[1]), ind)
    primal = EnzymeRules.needs_primal(config) ? (DV′..., ϵ.val) : nothing
    shadow_DV = if !isa(A, Const) && !isa(DV, Const)
        dD, dV = DV.dval
        dDtrunc = Diagonal(diagview(dD)[ind])
        dVtrunc = dV[:, ind]
        (dDtrunc, dVtrunc)
    else
        (nothing, nothing)
    end
    !isa(ϵ, Const) && make_zero(ϵ.dval)
    shadow_ϵ = !isa(ϵ, Const) ? ϵ.dval : zero(T)
    shadow = EnzymeRules.needs_shadow(config) ? (shadow_DV..., shadow_ϵ) : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_DV, shadow_DV, ind))
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(eigh_trunc!)},
        ::Type{RT},
        cache,
        A::Annotation,
        DV::Annotation{Tuple{TD, TV}},
        alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
    ) where {RT, TD, TV}
    cache_A, cache_DV, cache_dDVtrunc, ind = cache
    Aval = cache_A
    D, V = cache_DV
    dD, dV = cache_dDVtrunc
    if !isa(A, Const) && !isa(DV, Const)
        MatrixAlgebraKit.eigh_pullback!(A.dval, Aval, (D, V), (dD, dV), ind)
    end
    !isa(DV, Const) && make_zero!(DV.dval)
    !isa(ϵ, Const)  && make_zero!(ϵ.dval)
    return (nothing, nothing, nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(eig_trunc!)},
        ::Type{RT},
        A::Annotation,
        DV::Annotation{Tuple{TD, TV}},
        alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
    ) where {RT, TD, TV}
    # form cache if needed
    cache_A = copy(A.val)
    eig_full!(A.val, DV.val, alg.val.alg)
    cache_DV = copy.(DV.val)
    DV′, ind = MatrixAlgebraKit.truncate(eig_trunc!, DV.val, alg.val.trunc)
    ϵ.val = MatrixAlgebraKit.truncation_error!(diagview(DV.val[1]), ind)
    primal = EnzymeRules.needs_primal(config) ? (DV′..., ϵ.val) : nothing
    shadow_DV = if !isa(A, Const) && !isa(DV, Const)
        dD, dV = DV.dval
        dDtrunc = Diagonal(diagview(dD)[ind])
        dVtrunc = dV[:, ind]
        (dDtrunc, dVtrunc)
    else
        (nothing, nothing)
    end
    shadow = EnzymeRules.needs_shadow(config) ? (shadow_DV..., zero(T)) : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_DV, shadow_DV))
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(eig_trunc!)},
        ::Type{RT},
        cache,
        A::Annotation,
        DV::Annotation{Tuple{TD, TV}},
        alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm},
    ) where {RT, TD, TV}
    cache_A, cache_DV, cache_dDVtrunc = cache
    D, V = cache_DV
    Aval = cache_A
    dD, dV = cache_dDVtrunc
    if !isa(A, Const) && !isa(DV, Const)
        eig_trunc_pullback!(A.dval, Aval, (D, V), (dD, dV))
    end
    !isa(DV, Const) && make_zero!(DV.dval)
    !isa(ϵ, Const)  && make_zero!(ϵ.dval)
    return (nothing, nothing, nothing, nothing)
end

for (f!, f_full!, pb!) in (
        (eig_vals!, eig_full!, eig_pullback!),
        (eigh_vals!, eigh_full!, eigh_pullback!),
    )
    @eval begin
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f!)},
                ::Type{RT},
                A::Annotation,
                D::Annotation,
                alg::Annotation{<:MatrixAlgebraKit.AbstractAlgorithm},
            ) where {RT}
            cache_A = nothing
            cache_D = nothing
            nD, V = MatrixAlgebraKit.initialize_output($f_full!, A.val, alg.val)
            nD, V = $f_full!(A.val, (nD, V), alg.val)
            copy!(D.val, diagview(nD))
            primal = EnzymeRules.needs_primal(config) ? D.val : nothing
            shadow = EnzymeRules.needs_shadow(config) ? D.dval : nothing
            return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_D, V))
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

            cache_A, cache_D, V = cache
            Dval = !isnothing(cache_D) ? cache_D : D.val
            Aval = !isnothing(cache_A) ? cache_A : A.val
            ∂D = isa(D, Const) ? nothing : D.dval
            if !isa(A, Const) && !isa(D, Const)
                $pb!(A.dval, Aval, (Diagonal(Dval), V), (Diagonal(∂D), nothing))
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
        S::Annotation,
        alg::Annotation{<:MatrixAlgebraKit.AbstractAlgorithm},
    ) where {RT}
    cache_S = nothing
    cache_A = copy(A.val)
    U, nS, Vᴴ = svd_compact!(A.val, alg.val)
    copy!(S.val, diagview(nS))
    primal = EnzymeRules.needs_primal(config) ? S.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? S.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_S, U, Vᴴ))
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

    cache_A, cache_S, U, Vᴴ = cache
    Sval = !isnothing(cache_S) ? cache_S : S.val
    Aval = !isnothing(cache_A) ? cache_A : A.val
    ∂S = isa(S, Const) ? nothing : S.dval
    if !isa(A, Const) && !isa(S, Const)
        svd_pullback!(A.dval, Aval, (U, Diagonal(Sval), Vᴴ), (nothing, Diagonal(∂S), nothing))
    end
    !isa(S, Const) && make_zero!(S.dval)
    return (nothing, nothing, nothing)
end

end
