module MatrixAlgebraKitEnzymeExt

using MatrixAlgebraKit
using MatrixAlgebraKit: diagview, inv_safe, eig_trunc!, eigh_trunc!
using MatrixAlgebraKit: qr_pullback!, lq_pullback!
using MatrixAlgebraKit: qr_null_pullback!, lq_null_pullback!
using MatrixAlgebraKit: eig_pullback!, eigh_pullback!
using MatrixAlgebraKit: left_polar_pullback!, right_polar_pullback!
using Enzyme
using Enzyme.EnzymeCore
using Enzyme.EnzymeCore: EnzymeRules
using LinearAlgebra

@inline EnzymeRules.inactive_type(v::Type{<:MatrixAlgebraKit.AbstractAlgorithm}) = true


# two-argument factorizations like LQ, QR, EIG
for (f, pb) in (
        (qr_full!, qr_pullback!),
        (qr_compact!, qr_pullback!),
        (lq_full!, lq_pullback!),
        (lq_compact!, lq_pullback!),
        (eig_full!, eig_pullback!),
        (left_polar!, left_polar_pullback!),
        (right_polar!, right_polar_pullback!),
    )
    @eval begin
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f)},
                ::Type{RT},
                A::Annotation{<:AbstractMatrix},
                arg::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}},
                alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                kwargs...,
            ) where {RT}
            cache_arg = nothing
            # form cache if needed
            cache_A = (EnzymeRules.overwritten(config)[2] && !(typeof(arg) <: Const)) ? copy(A.val) : nothing
            func.val(A.val, arg.val, alg.val; kwargs...)
            primal = EnzymeRules.needs_primal(config) ? arg.val : nothing
            shadow = EnzymeRules.needs_shadow(config) ? arg.dval : nothing
            return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_arg))
        end
        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f)},
                dret::Type{RT},
                cache,
                A::Annotation{<:AbstractMatrix},
                arg::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}},
                alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                kwargs...
            ) where {RT}
            cache_A, cache_arg = cache
            argval = arg.val
            Aval = !isnothing(cache_A) ? cache_A : A.val
            ∂arg = isa(arg, Const) ? nothing : arg.dval
            if !isa(A, Const) && !isa(arg, Const)
                A.dval .= zero(eltype(Aval))
                $pb(A.dval, A.val, argval, ∂arg; kwargs...)
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
                A::Annotation{<:AbstractMatrix},
                arg::Annotation{<:AbstractMatrix},
                alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                kwargs...,
            ) where {RT}
            cache_arg = nothing
            # form cache if needed
            cache_A = nothing #copy(A.val)
            func.val(copy(A.val), arg.val, alg.val; kwargs...)
            primal = EnzymeRules.needs_primal(config) ? arg.val : nothing
            shadow = EnzymeRules.needs_shadow(config) ? arg.dval : nothing
            return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_arg))
        end

        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f)},
                dret::Type{RT},
                cache,
                A::Annotation{<:AbstractMatrix},
                arg::Annotation{<:AbstractMatrix},
                alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                tol::Real = MatrixAlgebraKit.default_pullback_gaugetol(arg.val),
                rank_atol::Real = tol,
                gauge_atol::Real = tol,
                kwargs...
            ) where {RT}
            cache_A, cache_arg = cache
            Aval = isnothing(cache_A) ? A.val : cache_A
            if !isa(A, Const) && !isa(arg, Const)
                A.dval .= zero(eltype(A.val))
                $pb(A.dval, A.val, arg.val, arg.dval; kwargs...)
            end
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
                A::Annotation{<:AbstractMatrix},
                USVᴴ::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix, <:AbstractMatrix}},
                alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                kwargs...,
            ) where {RT}
            # form cache if needed
            cache_USVᴴ = (EnzymeRules.overwritten(config)[3] && !(typeof(USVᴴ) <: Const)) ? copy(USVᴴ.val) : nothing
            cache_A = (EnzymeRules.overwritten(config)[2] && !(typeof(A) <: Const)) ? copy(A.val) : nothing
            func.val(A.val, USVᴴ.val, alg.val; kwargs...)
            primal = EnzymeRules.needs_primal(config) ? USVᴴ.val : nothing
            shadow = EnzymeRules.needs_shadow(config) ? USVᴴ.dval : nothing
            return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_USVᴴ))
        end
        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($f)},
                dret::Type{RT},
                cache,
                A::Annotation{<:AbstractMatrix},
                USVᴴ::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix, <:AbstractMatrix}},
                alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                kwargs...
            ) where {RT}
            cache_A, cache_USVᴴ = cache
            USVᴴval = !isnothing(cache_USVᴴ) ? cache_USVᴴ : USVᴴ.val
            U, S, Vᴴ = USVᴴval
            U, S, Vᴴ = USVᴴval
            ∂USVᴴ = isa(USVᴴ, Const) ? nothing : USVᴴ.dval
            if !isa(A, Const) && !isa(USVᴴ, Const)
                minmn = min(size(A.val)...)
                A.dval .= zero(eltype(A.dval))
                if size(U, 2) == size(Vᴴ, 1) == minmn # compact
                    MatrixAlgebraKit.svd_pullback!(A.dval, A.val, USVᴴval, ∂USVᴴ; kwargs...)
                else # full
                    vU = view(U, :, 1:minmn)
                    vS = Diagonal(diagview(S)[1:minmn])
                    vVᴴ = view(Vᴴ, 1:minmn, :)
                    vdU = view(∂USVᴴ[1], :, 1:minmn)
                    vdS = Diagonal(diagview(∂USVᴴ[2])[1:minmn])
                    vdVᴴ = view(∂USVᴴ[3], 1:minmn, :)
                    MatrixAlgebraKit.svd_pullback!(A.dval, A.val, (vU, vS, vVᴴ), (vdU, vdS, vdVᴴ); kwargs...)
                end
            end
            if !isa(USVᴴ, Const)
                make_zero!(USVᴴ.dval)
            end
            return (nothing, nothing, nothing)
        end
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(svd_trunc!)},
        ::Type{RT},
        A::Annotation{<:AbstractMatrix},
        USVᴴ::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix, <:AbstractMatrix}},
        ϵ::Annotation{Vector{T}},
        alg::Const{<:MatrixAlgebraKit.TruncatedAlgorithm};
        kwargs...,
    ) where {RT, T <: Real}
    # form cache if needed
    cache_A = copy(A.val)
    svd_compact!(A.val, USVᴴ.val, alg.val.alg)
    cache_USVᴴ = copy.(USVᴴ.val)
    USVᴴ′, ind = MatrixAlgebraKit.truncate(svd_trunc!, USVᴴ.val, alg.val.trunc)
    ϵ.val[1] = MatrixAlgebraKit.truncation_error!(diagview(USVᴴ.val[2]), ind)
    primal = EnzymeRules.needs_primal(config) ? (USVᴴ′..., ϵ.val) : nothing
    shadow_USVᴴ = if !isa(A, Const) && !isa(USVᴴ, Const)
        dU, dS, dVᴴ = USVᴴ.dval
        dStrunc = Diagonal(diagview(dS)[ind])
        dUtrunc = dU[:, ind]
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
        A::Annotation{<:AbstractMatrix},
        USVᴴ::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix, <:AbstractMatrix}},
        ϵ::Annotation{Vector{T}},
        alg::Const{<:MatrixAlgebraKit.TruncatedAlgorithm};
        kwargs...
    ) where {RT, T <: Real}
    cache_A, cache_USVᴴ, shadow_USVᴴ, ind = cache
    U, S, Vᴴ = cache_USVᴴ
    dU, dS, dVᴴ = shadow_USVᴴ
    if !isa(A, Const) && !isa(USVᴴ, Const)
        A.dval .= zero(eltype(A.val))
        A.dval .= MatrixAlgebraKit.svd_pullback!(A.dval, A.val, (U, S, Vᴴ), shadow_USVᴴ, ind; kwargs...)
    end
    if !isa(USVᴴ, Const)
        make_zero!(USVᴴ.dval)
    end
    if !isa(ϵ, Const)
        make_zero!(ϵ.dval)
    end
    return (nothing, nothing, nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(eigh_trunc!)},
        ::Type{RT},
        A::Annotation{<:AbstractMatrix},
        DV::Annotation{<:Tuple{<:Diagonal, <:AbstractMatrix}},
        ϵ::Annotation{Vector{T}},
        alg::Const{<:MatrixAlgebraKit.TruncatedAlgorithm};
        kwargs...,
    ) where {RT, T}
    # form cache if needed
    cache_A = copy(A.val)
    MatrixAlgebraKit.eigh_full!(A.val, DV.val, alg.val.alg)
    cache_DV = copy.(DV.val)
    DV′, ind = MatrixAlgebraKit.truncate(eigh_trunc!, DV.val, alg.val.trunc)
    ϵ.val[1] = MatrixAlgebraKit.truncation_error!(diagview(DV.val[1]), ind)
    primal = EnzymeRules.needs_primal(config) ? (DV′..., ϵ.val) : nothing
    shadow_DV = if !isa(A, Const) && !isa(DV, Const)
        dD, dV = DV.dval
        dDtrunc = Diagonal(diagview(dD)[ind])
        dVtrunc = dV[:, ind]
        (dDtrunc, dVtrunc)
    else
        (nothing, nothing)
    end
    shadow = EnzymeRules.needs_shadow(config) ? (shadow_DV..., [zero(T)]) : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_DV, shadow_DV, ind))
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(eigh_trunc!)},
        ::Type{RT},
        cache,
        A::Annotation{<:AbstractMatrix},
        DV::Annotation{<:Tuple{<:Diagonal, <:AbstractMatrix}},
        ϵ::Annotation{Vector{T}},
        alg::Const{<:MatrixAlgebraKit.TruncatedAlgorithm};
        kwargs...
    ) where {RT, T}
    cache_A, cache_DV, cache_dDVtrunc, ind = cache
    D, V = cache_DV
    dD, dV = cache_dDVtrunc
    if !isa(A, Const) && !isa(DV, Const)
        A.dval .= zero(eltype(A.val))
        A.dval .= MatrixAlgebraKit.eigh_pullback!(A.dval, A.val, (D, V), (dD, dV), ind; kwargs...)
    end
    if !isa(DV, Const)
        make_zero!(DV.dval)
    end
    if !isa(ϵ, Const)
        make_zero!(ϵ.dval)
    end
    return (nothing, nothing, nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(eig_trunc!)},
        ::Type{RT},
        A::Annotation{<:AbstractMatrix},
        DV::Annotation{<:Tuple{<:Diagonal, <:AbstractMatrix}},
        ϵ::Annotation{Vector{T}},
        alg::Const{<:MatrixAlgebraKit.TruncatedAlgorithm};
        kwargs...,
    ) where {RT, T}
    # form cache if needed
    cache_A = copy(A.val)
    eig_full!(A.val, DV.val, alg.val.alg)
    cache_DV = copy.(DV.val)
    DV′, ind = MatrixAlgebraKit.truncate(eig_trunc!, DV.val, alg.val.trunc)
    ϵ.val[1] = MatrixAlgebraKit.truncation_error!(diagview(DV.val[1]), ind)
    primal = EnzymeRules.needs_primal(config) ? (DV′..., ϵ.val) : nothing
    shadow_DV = if !isa(A, Const) && !isa(DV, Const)
        dD, dV = DV.dval
        dDtrunc = Diagonal(diagview(dD)[ind])
        dVtrunc = dV[:, ind]
        (dDtrunc, dVtrunc)
    else
        (nothing, nothing)
    end
    shadow = EnzymeRules.needs_shadow(config) ? (shadow_DV..., [zero(T)]) : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_DV, shadow_DV, ind))
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(eig_trunc!)},
        ::Type{RT},
        cache,
        A::Annotation{<:AbstractMatrix},
        DV::Annotation{<:Tuple{<:Diagonal, <:AbstractMatrix}},
        ϵ::Annotation{Vector{T}},
        alg::Const{<:MatrixAlgebraKit.TruncatedAlgorithm};
        kwargs...
    ) where {RT, T}
    cache_A, cache_DV, cache_dDVtrunc, ind = cache
    D, V = cache_DV
    dD, dV = cache_dDVtrunc
    if !isa(A, Const) && !isa(DV, Const)
        A.dval .= zero(eltype(A.val))
        A.dval .= MatrixAlgebraKit.eig_pullback!(A.dval, A.val, (D, V), (dD, dV), ind; kwargs...)
    end
    if !isa(DV, Const)
        make_zero!(DV.dval)
    end
    if !isa(ϵ, Const)
        make_zero!(ϵ.dval)
    end
    return (nothing, nothing, nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(eigh_full!)},
        ::Type{RT},
        A::Annotation{<:AbstractMatrix},
        DV::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}},
        alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
        kwargs...,
    ) where {RT}
    # form cache if needed
    cache_DV = nothing
    cache_A = EnzymeRules.overwritten(config)[2] ? copy(A.val) : nothing
    func.val(A.val, DV.val, alg.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? DV.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? DV.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_DV))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(eigh_full!)},
        ::Type{RT},
        cache,
        A::Annotation{<:AbstractMatrix},
        DV::Annotation{<:Tuple{<:Diagonal, <:AbstractMatrix}},
        alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
        kwargs...,
    ) where {RT}

    cache_A, cache_DV = cache
    DVval = !isnothing(cache_DV) ? cache_DV : DV.val
    Aval = !isnothing(cache_A) ? cache_A : A.val
    ∂DV = isa(DV, Const) ? nothing : DV.dval
    if !isa(A, Const) && !isa(DV, Const)
        Dmat, V = DVval
        ∂Dmat, ∂V = ∂DV
        A.dval .= zero(eltype(Aval))
        MatrixAlgebraKit.eigh_pullback!(A.dval, A.val, DVval, ∂DV; kwargs...)
        A.dval .*= 2
        diagview(A.dval) ./= 2
        for i in 1:size(A.dval, 1), j in 1:size(A.dval, 2)
            if i > j
                A.dval[i, j] = zero(eltype(A.dval))
            end
        end
    end
    if !isa(DV, Const)
        make_zero!(DV.dval)
    end
    return (nothing, nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(eig_vals!)},
        ::Type{RT},
        A::Annotation{<:AbstractMatrix},
        D::Annotation{<:AbstractVector},
        alg::Annotation{<:MatrixAlgebraKit.AbstractAlgorithm};
        kwargs...,
    ) where {RT}
    cache_D = nothing
    cache_A = EnzymeRules.overwritten(config)[2] ? copy(A.val) : nothing
    func.val(A.val, D.val, alg.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? D.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? D.dval : nothing
    # form cache if needed
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_D))
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(eig_vals!)},
        ::Type{RT},
        cache,
        A::Annotation{<:AbstractMatrix},
        D::Annotation{<:AbstractVector},
        alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
        kwargs...,
    ) where {RT}

    cache_A, cache_D = cache
    Dval = !isnothing(cache_D) ? cache_D : D.val
    Aval = !isnothing(cache_A) ? cache_A : A.val
    ∂D = isa(D, Const) ? nothing : D.dval
    if !isa(A, Const) && !isa(D, Const)
        _, V = eig_full(Aval, alg.val)
        A.dval .= zero(eltype(Aval))
        PΔV = V' \ Diagonal(D.dval)
        if eltype(A.dval) <: Real
            ΔAc = PΔV * V'
            A.dval .+= real.(ΔAc)
        else
            mul!(A.dval, PΔV, V', 1, 0)
        end
    end
    if !isa(D, Const)
        make_zero!(D.dval)
    end
    return (nothing, nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(eigh_vals!)},
        ::Type{RT},
        A::Annotation{<:AbstractMatrix},
        D::Annotation{<:AbstractVector},
        alg::Annotation{<:MatrixAlgebraKit.AbstractAlgorithm};
        kwargs...,
    ) where {RT}
    cache_D = nothing
    cache_A = EnzymeRules.overwritten(config)[2] ? copy(A.val) : nothing
    func.val(A.val, D.val, alg.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? D.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? D.dval : nothing
    # form cache if needed
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_D))
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(eigh_vals!)},
        ::Type{RT},
        cache,
        A::Annotation{<:AbstractMatrix},
        D::Annotation{<:AbstractVector},
        alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
        kwargs...,
    ) where {RT}

    cache_A, cache_D = cache
    Dval = !isnothing(cache_D) ? cache_D : D.val
    Aval = !isnothing(cache_A) ? cache_A : A.val
    ∂D = isa(D, Const) ? nothing : D.dval
    if !isa(A, Const) && !isa(D, Const)
        _, V = eigh_full(Aval, alg.val)
        A.dval .= zero(eltype(Aval))
        mul!(A.dval, V * Diagonal(real(∂D)), V', 1, 0)
        A.dval .*= 2
        diagview(A.dval) ./= 2
        for i in 1:size(A.dval, 1), j in 1:size(A.dval, 2)
            if i > j
                A.dval[i, j] = zero(eltype(A.dval))
            end
        end
    end
    if !isa(D, Const)
        make_zero!(D.dval)
    end
    return (nothing, nothing, nothing)
end

end
