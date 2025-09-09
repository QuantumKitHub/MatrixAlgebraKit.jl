module MatrixAlgebraKitEnzymeExt

using MatrixAlgebraKit
using MatrixAlgebraKit: diagview, inv_safe
using MatrixAlgebraKit: qr_pullback!, lq_pullback!, qr_pullfwd!, lq_pullfwd!
using MatrixAlgebraKit: qr_null_pullback!, lq_null_pullback!, qr_null_pullfwd!, lq_null_pullfwd!
using MatrixAlgebraKit: eig_pullback!, eigh_pullback!, eig_pullfwd!, eigh_pullfwd!
using MatrixAlgebraKit: left_polar_pullback!, right_polar_pullback!, left_polar_pullfwd!, right_polar_pullfwd!
using Enzyme
using Enzyme.EnzymeCore
using Enzyme.EnzymeCore: EnzymeRules
using LinearAlgebra

@inline EnzymeRules.inactive_type(v::Type{<:MatrixAlgebraKit.AbstractAlgorithm}) = true


# two-argument factorizations like LQ, QR, EIG
for (f, pb, pf) in ((qr_full!, qr_pullback!, qr_pullfwd!), 
                    (qr_compact!, qr_pullback!, qr_pullfwd!),
                    (lq_full!, lq_pullback!, lq_pullfwd!), 
                    (lq_compact!, lq_pullback!, lq_pullfwd!),
                    (eig_full!, eig_pullback!, eig_pullfwd!),
                    (left_polar!, left_polar_pullback!, left_polar_pullfwd!),
                    (right_polar!, right_polar_pullback!, right_polar_pullfwd!),
                   )
    @eval begin
        function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                              func::Const{typeof($f)},
                                              ::Type{RT},
                                              A::Annotation{<:AbstractMatrix},
                                              arg::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}},
                                              alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                                              kwargs...,
                                             ) where {RT}
            cache_arg = nothing
            # form cache if needed
            cache_A = (EnzymeRules.overwritten(config)[2] && !(typeof(arg) <: Const)) ? copy(A.val)  : nothing
            func.val(A.val, arg.val, alg.val; kwargs...)
            primal = EnzymeRules.needs_primal(config) ? arg.val  : nothing
            shadow = EnzymeRules.needs_shadow(config) ? arg.dval : nothing
            return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_arg))
        end
        function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
                                     func::Const{typeof($f)},
                                     dret::Type{RT},
                                     cache,
                                     A::Annotation{<:AbstractMatrix},
                                     arg::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}},
                                     alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                                     kwargs...) where {RT}
            cache_A, cache_arg = cache
            argval = arg.val
            Aval   = !isnothing(cache_A) ? cache_A  : A.val
            ∂arg   = isa(arg, Const)     ? nothing : arg.dval
            if !isa(A, Const) && !isa(arg, Const)
                A.dval .= zero(eltype(Aval))
                $pb(A.dval, A.val, argval, ∂arg; kwargs...)
            end
            !isa(arg, Const) && make_zero!(arg.dval)
            return (nothing, nothing, nothing)
        end
        function EnzymeRules.forward(config::EnzymeRules.FwdConfig,
                                     func::Const{typeof($f)},
                                     ::Type{RT},
                                     A::Annotation{<:AbstractMatrix},
                                     arg::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}},
                                     alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                                     kwargs...,
                                    ) where {RT}
            ret  = func.val(A.val, arg.val, alg.val; kwargs...)
            arg1, arg2 = ret
            m, n = size(A.val)

            if isa(arg, Union{Duplicated, DuplicatedNoNeed}) && !isa(A, Const)
                darg1, darg2 = arg.dval
                dA           = A.dval
                darg1, darg2 = $pf(dA, A.val, ret, arg.dval)
                dA          .= zero(eltype(A.val))
                shadow       = (darg1, darg2)
            elseif isa(A, Const) && !!isa(arg, Union{Duplicated, DuplicatedNoNeed})
                make_zero!(arg.dval)
                shadow = arg.dval
            end

            if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
                return Duplicated(ret, shadow)
            elseif EnzymeRules.needs_shadow(config)
                return shadow
            elseif EnzymeRules.needs_primal(config)
                return ret
            else
                return nothing
            end
        end
    end
end

for (f, pb, pf) in ((qr_null!, qr_null_pullback!, qr_null_pullfwd!), 
                    (lq_null!, lq_null_pullback!, lq_null_pullfwd!), 
                   )
    @eval begin
        function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
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
            primal = EnzymeRules.needs_primal(config) ? arg.val  : nothing
            shadow = EnzymeRules.needs_shadow(config) ? arg.dval : nothing
            return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_arg))
        end
                
        function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
                                     func::Const{typeof($f)},
                                     dret::Type{RT},
                                     cache,
                                     A::Annotation{<:AbstractMatrix},
                                     arg::Annotation{<:AbstractMatrix},
                                     alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                                     tol::Real=MatrixAlgebraKit.default_pullback_gaugetol(arg.val),
                                     rank_atol::Real=tol,
                                     gauge_atol::Real=tol,
                                     kwargs...) where {RT}
            cache_A, cache_arg = cache
            Aval   = isnothing(cache_A) ? A.val  : cache_A
            if !isa(A, Const) && !isa(arg, Const)
                A.dval .= zero(eltype(A.val))
                $pb(A.dval, A.val, arg.val, arg.dval; kwargs...)
            end
            return (nothing, nothing, nothing)
        end
        function EnzymeRules.forward(config::EnzymeRules.FwdConfig,
                                     func::Const{typeof($f)},
                                     ::Type{RT},
                                     A::Annotation{<:AbstractMatrix},
                                     arg::Annotation{<:AbstractMatrix},
                                     alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                                     kwargs...,
                                    ) where {RT}
            ret = func.val(A.val, arg.val, alg.val; kwargs...)

            if isa(arg, Union{Duplicated, DuplicatedNoNeed}) && !isa(A, Const)
                darg = arg.dval
                dA = A.dval
                $pf(dA, A.val, arg.val, darg)
                shadow = darg
            elseif isa(A, Const) && !!isa(arg, Union{Duplicated, DuplicatedNoNeed})
                make_zero!(arg.dval)
                shadow = arg.dval
            end

            if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
                return Duplicated(ret, shadow)
            elseif EnzymeRules.needs_shadow(config)
                return shadow
            elseif EnzymeRules.needs_primal(config)
                return ret
            else
                return nothing
            end
        end
    end
end


function EnzymeRules.forward(config::EnzymeRules.FwdConfig,
                             func::Const{typeof(svd_compact!)},
                             ::Type{RT},
                             A::Annotation{<:AbstractMatrix},
                             USVᴴ::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix, <:AbstractMatrix}},
                             alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                             kwargs...,
                            ) where {RT}
    ret    = EnzymeRules.needs_primal(config) || EnzymeRules.needs_shadow(config) ? func.val(A.val, USVᴴ.val; kwargs...) : nothing
    shadow = if EnzymeRules.needs_shadow(config)
        U, S, Vᴴ = ret
        V        = adjoint(Vᴴ)
        ∂S       = Diagonal(diag(real.(U' * A.dval * V)))
        m, n     = size(A.val)
        F        = one(eltype(S)) ./ ((diagview(S).^2)'  .- (diagview(S) .^ 2))
        diagview(F) .= zero(eltype(F))
        invSdiag = zeros(eltype(S), length(S.diag))
        for i in 1:length(S.diag)
            @inbounds invSdiag[i] = inv(diagview(S)[i])
        end
        invS = Diagonal(invSdiag)
        ∂U = U * (F .* (U' * A.dval * V * S + S * Vᴴ * A.dval' * U)) + (diagm(ones(eltype(U), m)) - U*U') * A.dval * V * invS
        #∂Vᴴ  = (FSdS' * Vᴴ) + (invS * U' * A.dval * (diagm(ones(eltype(U), size(V, 2))) - Vᴴ*V))
        ∂V = V * (F .* (S * U' * A.dval * V + Vᴴ * A.dval' * U * S)) + (diagm(ones(eltype(V), n)) - V*Vᴴ) * A.dval' * U * invS
        ∂Vᴴ = similar(Vᴴ)
        adjoint!(∂Vᴴ, ∂V)
        (∂U, ∂S, ∂Vᴴ)
    else
        nothing
    end
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated(ret, shadow)
    elseif EnzymeRules.needs_shadow(config)
        return shadow
    elseif EnzymeRules.needs_primal(config)
        return ret
    else
        return nothing
    end
end

# TODO
function EnzymeRules.forward(config::EnzymeRules.FwdConfig,
                             func::Const{typeof(svd_full!)},
                             ::Type{RT},
                             A::Annotation{<:AbstractMatrix},
                             USVᴴ::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix, <:AbstractMatrix}},
                             alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                             kwargs...,
                            ) where {RT}
    ret        = EnzymeRules.needs_primal(config) || EnzymeRules.needs_shadow(config) ? func.val(A.val, USVᴴ.val; kwargs...) : nothing
    shadow = if EnzymeRules.needs_shadow(config)
            fatU, fatS, fatVᴴ = ret
            ∂Ufat  = zeros(eltype(fatU), size(fatU))
            ∂Sfat  = zeros(eltype(fatS), size(fatS))
            ∂Vᴴfat = zeros(eltype(fatVᴴ), size(fatVᴴ))
            m, n       = size(A.val)
            minmn      = min(m, n)
            #U = view(fatU, :, 1:minmn)
            #S = Diagonal(diagview(fatS))
            #Vᴴ = view(fatVᴴ, 1:minmn, :)
            U = fatU 
            S = fatS 
            Vᴴ = fatVᴴ
            V        = adjoint(Vᴴ)
            ∂S       = Diagonal(diag(real.(U' * A.dval * V)))
            diagview(∂Sfat) .= diagview(∂S)
            m, n     = size(A.val)
            F        = one(eltype(S)) ./ ((diagview(S).^2)'  .- (diagview(S) .^ 2))
            diagview(F) .= zero(eltype(F))
            invSdiag = zeros(eltype(S), size(S))
            for ix in diagind(S)
                @inbounds invSdiag[ix] = inv(S[ix])
            end
            invS = invSdiag
            #FSdS = F .* (∂S * S .+ S * ∂S)
            ∂U = U * (F .* (U' * A.dval * V * S + S * Vᴴ * A.dval' * U)) + (diagm(ones(eltype(U), m)) - U*U') * A.dval * V * invS
            #view(∂Ufat, :, 1:minmn) .= view(∂U, :, :)
            ∂Ufat .= ∂U
            

            #∂Vᴴ  = (FSdS' * Vᴴ) + (invS * U' * A.dval * (diagm(ones(eltype(U), size(V, 2))) - Vᴴ*V))
            ∂V = V * (F .* (S * U' * A.dval * V + Vᴴ * A.dval' * U * S)) + (diagm(ones(eltype(V), n)) - V*Vᴴ) * A.dval' * U * invS
            ∂Vᴴ = similar(Vᴴ)
            adjoint!(∂Vᴴ, ∂V)
            #view(∂Vᴴfat, 1:minmn, :)   .= view(∂Vᴴ, :, :)
            ∂Vᴴfat .= ∂Vᴴ
            #=view(∂Ufat, :, minmn+1:m)  .= zero(eltype(fatU))
            view(∂Vᴴfat, minmn+1:n, :) .= zero(eltype(fatVᴴ))
            view(∂Sfat, minmn+1:m, :)  .= zero(eltype(fatVᴴ))
            view(∂Sfat, :, minmn+1:n)  .= zero(eltype(fatVᴴ))=#
            (∂Ufat, ∂Sfat, ∂Vᴴfat)
        else
            nothing
        end
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated(ret, shadow)
    elseif EnzymeRules.needs_shadow(config)
        return shadow
    elseif EnzymeRules.needs_primal(config)
        return ret
    else
        return nothing
    end
end
for f in (:svd_compact!, :svd_full!)
    @eval begin
        function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                              func::Const{typeof($f)},
                                              ::Type{RT},
                                              A::Annotation{<:AbstractMatrix},
                                              USVᴴ::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix, <:AbstractMatrix}},
                                              alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                                              kwargs...,
                                             ) where {RT}
            # form cache if needed
            cache_USVᴴ = (EnzymeRules.overwritten(config)[3] && !(typeof(USVᴴ) <: Const)) ? copy(USVᴴ.val)  : nothing
            cache_A    = (EnzymeRules.overwritten(config)[2] && !(typeof(A) <: Const)) ? copy(A.val)  : nothing
            func.val(A.val, USVᴴ.val, alg.val; kwargs...)
            primal = EnzymeRules.needs_primal(config) ? USVᴴ.val  : nothing
            shadow = EnzymeRules.needs_shadow(config) ? USVᴴ.dval : nothing
            return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_USVᴴ))
        end
        function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
                                     func::Const{typeof($f)},
                                     dret::Type{RT},
                                     cache,
                                     A::Annotation{<:AbstractMatrix},
                                     USVᴴ::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix, <:AbstractMatrix}},
                                     alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                                     kwargs...) where {RT}
            cache_A, cache_USVᴴ = cache
            USVᴴval = !isnothing(cache_USVᴴ) ? cache_USVᴴ : USVᴴ.val
            ∂USVᴴ   = isa(USVᴴ, Const) ? nothing : USVᴴ.dval
            if !isa(A, Const) && !isa(USVᴴ, Const)
                A.dval .= zero(eltype(A.dval))
                MatrixAlgebraKit.svd_pullback!(A.dval, A.val, USVᴴval, ∂USVᴴ; kwargs...)
            end
            if !isa(USVᴴ, Const)
                make_zero!(USVᴴ.dval)
            end
            return (nothing, nothing, nothing)
        end
    end
end

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                      func::Const{typeof(svd_trunc!)},
                                      ::Type{RT},
                                      A::Annotation{<:AbstractMatrix},
                                      USVᴴ::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix, <:AbstractMatrix}},
                                      ϵ::Annotation{Vector{T}},
                                      alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                                      kwargs...,
                                     ) where {RT, T<:Real}
    # form cache if needed
    cache_A     = copy(A.val)
    svd_compact!(A.val, USVᴴ.val, alg.val.alg)
    cache_USVᴴ  = copy.(USVᴴ.val)
    USVᴴ′, ind  = MatrixAlgebraKit.truncate(svd_trunc!, USVᴴ.val, alg.val.trunc)
    ϵ.val[1]    = MatrixAlgebraKit.truncation_error!(diagview(USVᴴ.val[2]), ind)
    primal      = EnzymeRules.needs_primal(config) ? (USVᴴ′..., ϵ.val)  : nothing
    shadow_USVᴴ = if !isa(A, Const) && !isa(USVᴴ, Const)
        dU, dS, dVᴴ = USVᴴ.dval
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
function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
                             func::Const{typeof(svd_trunc!)},
                             dret::Type{RT},
                             cache,
                             A::Annotation{<:AbstractMatrix},
                             USVᴴ::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix, <:AbstractMatrix}},
                             ϵ::Annotation{Vector{T}},
                             alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                             kwargs...) where {RT, T<:Real}
    cache_A, cache_USVᴴ, shadow_USVᴴ, ind = cache
    U, S, Vᴴ    = cache_USVᴴ
    dU, dS, dVᴴ = shadow_USVᴴ
    if !isa(A, Const) && !isa(USVᴴ, Const)
        A.dval .= zero(eltype(A.val))
        A.dval .= MatrixAlgebraKit.svd_pullback!(A.dval, A.val, (U, S, Vᴴ), shadow_USVᴴ, ind; kwargs...)
    end
    if !isa(USVᴴ, Const)
        make_zero!(USVᴴ.dval)
    end
    if !isa(ϵ, Const)
        ϵ.dval .= zero(T)
    end
    return (nothing, nothing, nothing, nothing)
end

function EnzymeRules.forward(config::EnzymeRules.FwdConfig,
                             func::Const{typeof(eigh_vals!)},
                             ::Type{RT},
                             A::Annotation{<:AbstractMatrix},
                             D::Annotation{<:AbstractVector},
                             alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                             kwargs...,
                            ) where {RT}
    Dmat, V = eigh_full(A.val; kwargs...)
    if isa(D, Union{Duplicated, DuplicatedNoNeed}) && !isa(A, Const)
        ∂K      = inv(V) * A.dval * V
        ∂Kdiag  = diag(∂K)
        D.dval .= real.(copy(∂Kdiag))
        A.dval .= zero(eltype(A.val))
        shadow  = D.dval 
    elseif isa(A, Const) && !!isa(D, Union{Duplicated, DuplicatedNoNeed})
        make_zero!(D.dval)
        shadow = D.dval
    end
    eigh_vals!(A.val, zeros(real(eltype(A.val)), size(A.val, 1)))
    D.val .= diagview(Dmat)
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated(Dmat.diag, shadow)
    elseif EnzymeRules.needs_shadow(config)
        return shadow
    elseif EnzymeRules.needs_primal(config)
        return Dmat.diag 
    else
        return nothing
    end
end

function EnzymeRules.forward(config::EnzymeRules.FwdConfig,
                             func::Const{typeof(eigh_full!)},
                             ::Type{RT},
                             A::Annotation{<:AbstractMatrix},
                             DV::Annotation{<:Tuple{<:Diagonal, <:AbstractMatrix}},
                             alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                             kwargs...,
                            ) where {RT}
    Dmat, V = func.val(A.val, DV.val; kwargs...)
    if isa(A, Const) || all(iszero, A.dval)
        make_zero!(DV.dval[1])
        make_zero!(DV.dval[2])
        make_zero!(A.dval)
        shadow = (DV.dval[1], DV.dval[2])
    else
        ∂K      = inv(V) * A.dval * V
        ∂Kdiag  = diagview(∂K)
        ∂Ddiag  = diagview(DV.dval[1])
        ∂Ddiag .= real.(∂Kdiag)
        D       = diagview(Dmat)
        dDD     = transpose(D) .- D
        ∂K    ./= dDD 
        ∂Kdiag .= zero(eltype(V))
        mul!(DV.dval[2], V, ∂K, 1, 0)
        shadow  = DV.dval[2]
        A.dval .= zero(eltype(A.val))
        shadow  = (Diagonal(∂Ddiag), DV.dval[2])
    end
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated((Dmat, V), shadow)
    elseif EnzymeRules.needs_shadow(config)
        return shadow
    elseif EnzymeRules.needs_primal(config)
        return (Dmat, V)
    else
        return nothing
    end
end

function EnzymeRules.forward(config::EnzymeRules.FwdConfig,
                             func::Const{typeof(eig_vals!)},
                             ::Type{RT},
                             A::Annotation{<:AbstractMatrix},
                             D::Annotation{<:AbstractVector},
                             alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                             kwargs...,
                            ) where {RT}
    Dval, V = eig_full(A.val, alg.val; kwargs...)
    if isa(D, Union{Duplicated, DuplicatedNoNeed}) && !isa(A, Const)
        ∂K      = inv(V) * A.dval * V
        ∂Kdiag  = diag(∂K)
        D.dval .= copy(∂Kdiag)
        A.dval .= zero(eltype(A.val))
        shadow  = D.dval
    elseif isa(A, Const) && !!isa(D, Union{Duplicated, DuplicatedNoNeed})
        make_zero!(D.dval)
        shadow  = D.dval
    end
    eig_vals!(A.val, zeros(complex(eltype(A.val)), size(A.val, 1)))
    D.val .= diagview(Dval)
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated(Dmat.diag, shadow)
    elseif EnzymeRules.needs_shadow(config)
        return shadow
    elseif EnzymeRules.needs_primal(config)
        return Dmat.diag 
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                      func::Const{typeof(eigh_trunc!)},
                                      ::Type{RT},
                                      A::Annotation{<:AbstractMatrix},
                                      DV::Annotation{<:Tuple{<:Diagonal, <:AbstractMatrix}},
                                      alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                                      kwargs...,
                                     ) where {RT}
    # form cache if needed
    cache_A   = copy(A.val)
    eigh_full!(A.val, DV.val, alg.val.alg)
    cache_DV  = copy.(DV.val)
    DV′, ind  = MatrixAlgebraKit.truncate(eigh_trunc!, DV.val, alg.val.trunc)
    ϵ         = MatrixAlgebraKit.truncation_error!(diagview(DV.val[1]), ind)
    primal    = EnzymeRules.needs_primal(config) ? (DV′..., ϵ)  : nothing
    shadow_DV = if !isa(A, Const) && !isa(DV, Const)
        dD, dV  = DV.dval
        dDtrunc = Diagonal(diagview(dD)[ind])
        dVtrunc = dV[:, ind]
        (dDtrunc, dVtrunc)
    else
        (nothing, nothing)
    end
    shadow = EnzymeRules.needs_shadow(config) ? (shadow_DV..., zero(ϵ)) : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_DV, shadow_DV, ind))
end
function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
                             func::Const{typeof(eigh_trunc!)},
                             dret,
                             cache,
                             A::Annotation{<:AbstractMatrix},
                             DV::Annotation{<:Tuple{<:Diagonal, <:AbstractMatrix}},
                             alg::Const{<:MatrixAlgebraKit.TruncatedAlgorithm};
                             kwargs...)
    cache_A, cache_DV, cache_dDVtrunc, ind = cache
    D, V   = cache_DV
    dD, dV = cache_dDVtrunc
    if !isa(A, Const) && !isa(DV, Const)
        A.dval .= zero(eltype(A.val))
        A.dval .= MatrixAlgebraKit.eigh_pullback!(A.dval, A.val, (D, V), (dD, dV), ind; kwargs...)
    end
    if !isa(DV, Const)
        make_zero!(DV.dval)
    end
    return (nothing, nothing, nothing, nothing)
end
#=
function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                      func::Const{typeof(eig_trunc!)},
                                      ::Type{RT},
                                      A::Annotation{<:AbstractMatrix},
                                      DV::Annotation{<:Tuple{<:Diagonal, <:AbstractMatrix}},
                                      alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                                      kwargs...,
                                     ) where {RT}
    # form cache if needed
    cache_A   = copy(A.val)
    eig_full!(A.val, DV.val, alg.val.alg)
    cache_DV  = copy.(DV.val)
    DV′, ind  = MatrixAlgebraKit.truncate(eig_trunc!, DV.val, alg.val.trunc)
    ϵ         = MatrixAlgebraKit.truncation_error!(diagview(DV.val[1]), ind)
    primal    = EnzymeRules.needs_primal(config) ? (DV′..., ϵ)  : nothing
    shadow_DV = if !isa(A, Const) && !isa(DV, Const)
        dD, dV  = DV.dval
        dDtrunc = Diagonal(diagview(dD)[ind])
        dVtrunc = dV[:, ind]
        (dDtrunc, dVtrunc)
    else
        (nothing, nothing)
    end
    shadow = EnzymeRules.needs_shadow(config) ? (shadow_DV..., zero(ϵ)) : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_DV, shadow_DV, ind))
end
function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
                             func::Const{typeof(eig_trunc!)},
                             dret,
                             cache,
                             A::Annotation{<:AbstractMatrix},
                             DV::Annotation{<:Tuple{<:Diagonal, <:AbstractMatrix}},
                             alg::Const{<:MatrixAlgebraKit.TruncatedAlgorithm};
                             kwargs...)
    cache_A, cache_DV, cache_dDVtrunc, ind = cache
    D, V   = cache_DV
    dD, dV = cache_dDVtrunc
    if !isa(A, Const) && !isa(DV, Const)
        A.dval .= zero(eltype(A.val))
        A.dval .= MatrixAlgebraKit.eigh_pullback!(A.dval, A.val, (D, V), (dD, dV), ind; kwargs...)
    end
    if !isa(DV, Const)
        make_zero!(DV.dval)
    end
    return (nothing, nothing, nothing, nothing)
end
=#
function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                      func::Const{typeof(eigh_full!)},
                                      ::Type{RT},
                                      A::Annotation{<:AbstractMatrix},
                                      DV::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}},
                                      alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                                      kwargs...,
                                     ) where {RT}
    # form cache if needed
    cache_DV = nothing
    cache_A  = EnzymeRules.overwritten(config)[2] ? copy(A.val)  : nothing
    func.val(A.val, DV.val, alg.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? DV.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? DV.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_DV))
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
                             func::Const{typeof(eigh_full!)},
                             ::Type{RT},
                             cache,
                             A::Annotation{<:AbstractMatrix},
                             DV::Annotation{<:Tuple{<:Diagonal, <:AbstractMatrix}},
                             alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                             kwargs...,
                            ) where {RT}

    cache_A, cache_DV = cache
    DVval   = !isnothing(cache_DV) ? cache_DV : DV.val
    Aval    = !isnothing(cache_A)  ? cache_A  : A.val
    ∂DV     = isa(DV, Const) ? nothing : DV.dval
    if !isa(A, Const) && !isa(DV, Const)
        Dmat, V   = DVval
        ∂Dmat, ∂V = ∂DV
        A.dval   .= zero(eltype(Aval))
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

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                      func::Const{typeof(eig_vals!)},
                                      ::Type{RT},
                                      A::Annotation{<:AbstractMatrix},
                                      D::Annotation{<:AbstractVector},
                                      alg::Annotation{<:MatrixAlgebraKit.AbstractAlgorithm};
                                      kwargs...,
                                     ) where {RT}
    cache_D = nothing
    cache_A = EnzymeRules.overwritten(config)[2] ? copy(A.val)  : nothing
    func.val(A.val, D.val, alg.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? D.val  : nothing
    shadow = EnzymeRules.needs_shadow(config) ? D.dval : nothing
    # form cache if needed
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_D))
end
function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
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
    ∂D   = isa(D, Const) ? nothing : D.dval
    if !isa(A, Const) && !isa(D, Const)
        _, V    = eig_full(Aval, alg.val)
        A.dval .= zero(eltype(Aval))
        PΔV     = V' \ Diagonal(D.dval)
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

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                      func::Const{typeof(eigh_vals!)},
                                      ::Type{RT},
                                      A::Annotation{<:AbstractMatrix},
                                      D::Annotation{<:AbstractVector},
                                      alg::Annotation{<:MatrixAlgebraKit.AbstractAlgorithm};
                                      kwargs...,
                                     ) where {RT}
    cache_D = nothing
    cache_A = EnzymeRules.overwritten(config)[2] ? copy(A.val)  : nothing
    func.val(A.val, D.val, alg.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? D.val  : nothing
    shadow = EnzymeRules.needs_shadow(config) ? D.dval : nothing
    # form cache if needed
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_D))
end
function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
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
    ∂D   = isa(D, Const) ? nothing : D.dval
    if !isa(A, Const) && !isa(D, Const)
        _, V = eigh_full(Aval, alg.val)
        A.dval   .= zero(eltype(Aval))
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
