module MatrixAlgebraKitEnzymeExt

using MatrixAlgebraKit
using MatrixAlgebraKit: diagview, inv_safe
using ChainRulesCore
using Enzyme
using Enzyme.EnzymeCore
using Enzyme.EnzymeCore: EnzymeRules
using LinearAlgebra

@inline EnzymeRules.inactive_type(v::Type{<:MatrixAlgebraKit.AbstractAlgorithm}) = true

#Enzyme.@import_rrule(typeof(MatrixAlgebraKit.svd_trunc!), AbstractMatrix, Any, MatrixAlgebraKit.TruncatedAlgorithm)

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                      func::Const{typeof(lq_null!)},
                                      ::Type{RT},
                                      A::Annotation{<:AbstractMatrix},
                                      Nᴴ::Annotation{<:AbstractMatrix},
                                      alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                                      kwargs...,
                                     ) where {RT}
    cache_Nᴴ = nothing
    # form cache if needed
    cache_A = (EnzymeRules.overwritten(config)[2] && !(typeof(Nᴴ) <: Const)) ? copy(A.val)  : nothing
    func.val(A.val, Nᴴ.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? Nᴴ.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? Nᴴ.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_Nᴴ))
end
        
function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
                             func::Const{typeof(lq_null!)},
                             dret::Type{RT},
                             cache,
                             A::Annotation{<:AbstractMatrix},
                             Nᴴ::Annotation{<:AbstractMatrix},
                             alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                             tol::Real=MatrixAlgebraKit.default_pullback_gaugetol(Nᴴ.val),
                             rank_atol::Real=tol,
                             gauge_atol::Real=tol,
                             kwargs...) where {RT}
    cache_A, cache_Nᴴ = cache
    Aval    = isnothing(cache_A) ? A.val : cache_A
    Nᴴval   = isnothing(cache_Nᴴ) ? Nᴴ.val : cache_Nᴴ
    ∂Nᴴ     = isa(Nᴴ, Const) ? nothing : Nᴴ.dval
    A.dval .= zero(eltype(A.val))
    if !isa(A, Const) && !isa(Nᴴ, Const)
        Ac = MatrixAlgebraKit.copy_input(lq_full, Aval)
        LQ = MatrixAlgebraKit.initialize_output(lq_full!, Aval, alg.val)
        L, Q = lq_full!(Ac, LQ, alg.val)
        copy!(Nᴴval, view(Q, (size(Aval, 1) + 1):size(Aval, 2), 1:size(Aval, 2)))
        (m, n) = size(Aval)
        minmn  = min(m, n)
        ∂Q     = zeros(eltype(Aval), (n, n))
        view(∂Q, (minmn + 1):n, 1:n) .= ∂Nᴴ
        EnzymeRules.reverse(config, Const(lq_compact!), RT, (nothing, nothing), A, Duplicated((L, Q), (zeros(eltype(L), size(L)), ∂Q)), alg)
    end
    if !isa(Nᴴ, Const)
        make_zero!(Nᴴ.dval)
    end
    return (nothing, nothing, nothing)
end

for f in (:lq_compact!, :lq_full!)
    @eval begin
        function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                              func::Const{typeof($f)},
                                              ::Type{RT},
                                              A::Annotation{<:AbstractMatrix},
                                              LQ::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}},
                                              alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                                              kwargs...,
                                             ) where {RT}
            cache_LQ = nothing
            # form cache if needed
            cache_A = (EnzymeRules.overwritten(config)[2] && !(typeof(LQ) <: Const)) ? copy(A.val)  : nothing
            func.val(A.val, LQ.val, alg.val; kwargs...)
            primal = EnzymeRules.needs_primal(config) ? LQ.val : nothing
            shadow = EnzymeRules.needs_shadow(config) ? LQ.dval : nothing
            return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_LQ))
        end
        function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
                                     func::Const{typeof($f)},
                                     dret::Type{RT},
                                     cache,
                                     A::Annotation{<:AbstractMatrix},
                                     LQ::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}},
                                     alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                                     tol::Real=MatrixAlgebraKit.default_pullback_gaugetol(LQ.val[1]),
                                     rank_atol::Real=tol,
                                     gauge_atol::Real=tol,
                                     kwargs...) where {RT}
            cache_A, cache_LQ = cache
            LQval = !isnothing(cache_LQ) ? cache_LQ : LQ.val
            Aval  = !isnothing(cache_A) ? cache_A : A.val
            ∂LQ   = isa(LQ, Const) ? nothing : LQ.dval
            if !isa(A, Const) && !isa(LQ, Const)
                A.dval .= zero(eltype(Aval))
                MatrixAlgebraKit.lq_compact_pullback!(A.dval, LQval, ∂LQ; tol, rank_atol, gauge_atol)
            end
            if !isa(LQ, Const)
                make_zero!(LQ.dval)
            end
            return (nothing, nothing, nothing)
        end
        function EnzymeRules.forward(config::EnzymeRules.FwdConfig,
                                     func::Const{typeof($f)},
                                     ::Type{RT},
                                     A::Annotation{<:AbstractMatrix},
                                     LQ::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}},
                                     alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                                     kwargs...,
                                    ) where {RT}
            ret  = func.val(A.val, LQ.val, alg.val; kwargs...)
            L, Q = ret
            m, n = size(A.val)

            if isa(LQ, Union{Duplicated, DuplicatedNoNeed}) && !isa(A, Const)
                invL = inv(L)
                ∂K   = invL * A.dval * Q'
                ∂K   = ∂K + ∂K'
                diagview(∂K) ./= 2
                ∂K[MatrixAlgebraKit.uppertriangularind(∂K)] .= zero(eltype(∂K))
                ∂L   = L * ∂K
                ∂Q   = invL * A.dval - invL * ∂L * Q 
                shadow = (∂L, ∂Q)
            elseif isa(A, Const) && !!isa(LQ, Union{Duplicated, DuplicatedNoNeed})
                make_zero!(LQ.dval)
                shadow = LQ.dval
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

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                      func::Const{typeof(qr_null!)},
                                      ::Type{RT},
                                      A::Annotation{<:AbstractMatrix},
                                      N::Annotation{<:AbstractMatrix},
                                      alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                                      kwargs...,
                                     ) where {RT}
    # form cache if needed
    cache_N = EnzymeRules.overwritten(config)[3] ? copy(N.val) : nothing
    cache_A = EnzymeRules.overwritten(config)[2] ? copy(A.val) : nothing
    func.val(A.val, N.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? N.val  : nothing
    shadow = EnzymeRules.needs_shadow(config) ? N.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_N))
end
        
function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
                             func::Const{typeof(qr_null!)},
                             dret::Type{RT},
                             cache,
                             A::Annotation{<:AbstractMatrix},
                             N::Annotation{<:AbstractMatrix},
                             alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                             tol::Real=MatrixAlgebraKit.default_pullback_gaugetol(N.val),
                             rank_atol::Real=tol,
                             gauge_atol::Real=tol,
                             kwargs...) where {RT}
    cache_A, cache_N = cache
    Aval = isnothing(cache_A) ? A.val : cache_A
    Nval = isnothing(cache_N) ? N.val : cache_N
    ∂N   = isa(N, Const) ? nothing : N.dval
    A.dval .= zero(eltype(A.val))
    if !isa(A, Const) && !isa(N, Const)
        Ac = MatrixAlgebraKit.copy_input(qr_full, Aval)
        QR = MatrixAlgebraKit.initialize_output(qr_full!, Aval, alg.val)
        Q, R = qr_full!(Ac, QR, alg.val)
        copy!(Nval, view(Q, 1:size(Aval, 1), (size(Aval, 2) + 1):size(Aval, 1)))
        (m, n) = size(Aval)
        minmn  = min(m, n)
        ∂Q     = zeros(eltype(Aval), (m, m))
        view(∂Q, 1:m, (minmn + 1):m) .= ∂N
        EnzymeRules.reverse(config, Const(qr_compact!), RT, (nothing, nothing), A, Duplicated((Q, R), (∂Q, zeros(eltype(R), size(R)))), alg)
    end
    if !isa(N, Const)
        make_zero!(N.dval)
    end
    return (nothing, nothing, nothing)
end

for f in (:qr_compact!, :qr_full!)
    @eval begin
        function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                              func::Const{typeof($f)},
                                              ::Type{RT},
                                              A::Annotation{<:AbstractMatrix},
                                              QR::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}},
                                              alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                                              kwargs...,
                                             ) where {RT}
            cache_QR = nothing
            # form cache if needed
            cache_A = (EnzymeRules.overwritten(config)[2] && !(typeof(QR) <: Const)) ? copy(A.val)  : nothing
            func.val(A.val, QR.val, alg.val; kwargs...)
            primal = EnzymeRules.needs_primal(config) ? QR.val : nothing
            shadow = EnzymeRules.needs_shadow(config) ? QR.dval : nothing
            return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_QR))
        end
        function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
                                     func::Const{typeof($f)},
                                     dret::Type{RT},
                                     cache,
                                     A::Annotation{<:AbstractMatrix},
                                     QR::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}},
                                     alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                                     tol::Real=MatrixAlgebraKit.default_pullback_gaugetol(QR.val[2]),
                                     rank_atol::Real=tol,
                                     gauge_atol::Real=tol,
                                     kwargs...) where {RT}
            cache_A, cache_QR = cache
            QRval = !isnothing(cache_QR) ? cache_QR : QR.val
            Aval  = !isnothing(cache_A) ? cache_A : A.val
            ∂QR   = isa(QR, Const) ? nothing : QR.dval
            if !isa(A, Const) && !isa(QR, Const)
                A.dval .= zero(eltype(Aval))
                MatrixAlgebraKit.qr_compact_pullback!(A.dval, QRval, ∂QR; tol, rank_atol, gauge_atol)
            end
            if !isa(QR, Const)
                make_zero!(QR.dval)
            end
            return (nothing, nothing, nothing)
        end
        function EnzymeRules.forward(config::EnzymeRules.FwdConfig,
                                     func::Const{typeof($f)},
                                     ::Type{RT},
                                     A::Annotation{<:AbstractMatrix},
                                     QR::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}},
                                     alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                                     kwargs...,
                                    ) where {RT}
            ret  = func.val(A.val, QR.val, alg.val; kwargs...)
            Q, R = ret
            m, n = size(A.val)

            if isa(QR, Union{Duplicated, DuplicatedNoNeed}) && !isa(A, Const)
                invR = inv(R)
                tmp  = Q' * A.dval * invR
                Rtmp = tmp + tmp'
                diagview(Rtmp) ./= 2
                Rtmp[MatrixAlgebraKit.lowertriangularind(Rtmp)] .= zero(eltype(Rtmp))
                ∂R   = Rtmp * R
                ∂Q   = A.dval * invR - Q * ∂R * invR
                shadow = (∂Q, ∂R)
            elseif isa(A, Const) && !!isa(QR, Union{Duplicated, DuplicatedNoNeed})
                make_zero!(QR.dval)
                shadow = QR.dval
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

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                      func::Const{typeof(left_polar!)},
                                      ::Type{RT},
                                      A::Annotation{<:AbstractMatrix},
                                      WP::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}},
                                      alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                                      kwargs...,
                                     ) where {RT}
    # form cache if needed
    cache_WP = (EnzymeRules.overwritten(config)[3] && !(typeof(WP) <: Const)) ? copy(WP.val) : nothing
    cache_A  = (EnzymeRules.overwritten(config)[2] && !(typeof(A) <: Const))  ? copy(A.val)  : nothing
    func.val(A.val, WP.val, alg.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? WP.val  : nothing
    shadow = EnzymeRules.needs_shadow(config) ? WP.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_WP))
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
                             func::Const{typeof(left_polar!)},
                             dret::Type{RT},
                             cache,
                             A::Annotation{<:AbstractMatrix},
                             WP::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}},
                             alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                             kwargs...) where {RT}
    cache_A, cache_WP = cache
    WPval = !isnothing(cache_WP) ? cache_WP : WP.val
    Aval  = !isnothing(cache_A)  ? cache_A  : A.val
    ∂WP   = isa(WP, Const)       ? nothing  : WP.dval
    if !isa(A, Const) && !isa(WP, Const)
        A.dval .= zero(eltype(Aval))
        MatrixAlgebraKit.left_polar_pullback!(A.dval, WPval, ∂WP)
    end
    if !isa(WP, Const)
        make_zero!(WP.dval)
    end
    return (nothing, nothing, nothing)
end

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                      func::Const{typeof(right_polar!)},
                                      ::Type{RT},
                                      A::Annotation{<:AbstractMatrix},
                                      PWᴴ::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}},
                                      alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                                      kwargs...,
                                     ) where {RT}
    # form cache if needed
    cache_PWᴴ = (EnzymeRules.overwritten(config)[3] && !(typeof(PWᴴ) <: Const)) ? copy(PWᴴ.val) : nothing
    cache_A   = (EnzymeRules.overwritten(config)[2] && !(typeof(A) <: Const))  ? copy(A.val)  : nothing
    func.val(A.val, PWᴴ.val, alg.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? PWᴴ.val  : nothing
    shadow = EnzymeRules.needs_shadow(config) ? PWᴴ.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_PWᴴ))
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
                             func::Const{typeof(right_polar!)},
                             dret::Type{RT},
                             cache,
                             A::Annotation{<:AbstractMatrix},
                             PWᴴ::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}},
                             alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                             kwargs...) where {RT}
    cache_A, cache_PWᴴ = cache
    PWᴴval = !isnothing(cache_PWᴴ) ? cache_PWᴴ : PWᴴ.val
    Aval   = !isnothing(cache_A)   ? cache_A   : A.val
    ∂PWᴴ   = isa(PWᴴ, Const)       ? nothing   : PWᴴ.dval
    if !isa(A, Const) && !isa(PWᴴ, Const)
        A.dval .= zero(eltype(Aval))
        MatrixAlgebraKit.right_polar_pullback!(A.dval, PWᴴval, ∂PWᴴ)
    end
    if !isa(PWᴴ, Const)
        make_zero!(PWᴴ.dval)
    end
    return (nothing, nothing, nothing)
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
                MatrixAlgebraKit.svd_compact_pullback!(A.dval, USVᴴval, ∂USVᴴ; kwargs...)
            end
            if !isa(USVᴴ, Const)
                make_zero!(USVᴴ.dval)
            end
            return (nothing, nothing, nothing)
        end
    end
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
        # TODO provide this as the input
        A.dval .*= 2
        diagview(A.dval) ./= 2
        for i in 1:size(A.dval, 1), j in 1:size(A.dval, 2)
            if i > j
                A.dval[i, j] = zero(eltype(A.dval)) 
            end
        end
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
        #=A.dval .*= 2
        diagview(A.dval) ./= 2
        for i in 1:size(A.dval, 1), j in 1:size(A.dval, 2)
            if i > j
                A.dval[i, j] = zero(eltype(A.dval)) 
            end
        end=#
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
        println("∂K")
        display(∂K)
        println()
        println("∂V after")
        display(DV.dval[2])
        println()
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

function EnzymeRules.forward(config::EnzymeRules.FwdConfig,
                             func::Const{typeof(eig_full!)},
                             ::Type{RT},
                             A::Annotation{<:AbstractMatrix},
                             DV::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}},
                             alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                             kwargs...,
                            ) where {RT}
    tol             = MatrixAlgebraKit.default_pullback_gaugetol(DV.val[1])
    degeneracy_atol = tol
    gauge_atol      = tol
    Dmat, V = func.val(A.val, DV.val, alg.val; kwargs...)
    if isa(A, Const) || all(iszero, A.dval)
        if !isa(DV, Const)
            make_zero!(DV.dval[1])
            make_zero!(DV.dval[2])
        end
        if !isa(A, Const)
            make_zero!(A.dval)
        end
        shadow  = !isa(DV, Const) ? DV.dval : nothing
    else
        D       = diagview(Dmat) 
        ∂K      = inv(V) * A.dval * V
        ∂Kdiag  = diagview(∂K)
        DV.dval[1].diag .= ∂Kdiag
        ∂K    ./= transpose(D) .- D
        fill!(∂Kdiag, zero(eltype(D)))
        DV.dval[2] .= (V * ∂K)
        shadow  = DV.dval
        A.dval .= zero(eltype(A.val))
    end
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated(DV.val, shadow)
    elseif EnzymeRules.needs_shadow(config)
        return shadow
    elseif EnzymeRules.needs_primal(config)
        return DV.val
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                      func::Const{typeof(eig_full!)},
                                      ::Type{RT},
                                      A::Annotation{<:AbstractMatrix},
                                      DV::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}},
                                      alg::Annotation{<:MatrixAlgebraKit.AbstractAlgorithm};
                                      kwargs...,
                                     ) where {RT}
    cache_DV        = nothing
    cache_A = EnzymeRules.overwritten(config)[2] ? copy(A.val)  : nothing
    func.val(A.val, DV.val, alg.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? DV.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? DV.dval : nothing
    # form cache if needed
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_DV))
end

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
                             func::Const{typeof(eig_full!)},
                             dret::Type{RT},
                             cache,
                             A::Annotation{<:AbstractMatrix},
                             DV::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}},
                             alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                             kwargs...,
                            ) where {RT}
    tol             = MatrixAlgebraKit.default_pullback_gaugetol(DV.val[1])
    degeneracy_atol = tol
    gauge_atol      = tol

    cache_A, cache_DV = cache
    DVval   = !isnothing(cache_DV) ? cache_DV : DV.val

    A.dval .= zero(eltype(A.dval))
    ∂DV     = DV.dval
    
    if !isa(A, Const) && !isa(DV, Const) 
        Dmat, V   = DVval
        D         = diagview(Dmat)
        ∂Dmat, ∂V = ∂DV
        Vd∂V      = V' * ∂V

        dDD       = transpose(D) .- D
        mask      = abs.(dDD) .< degeneracy_atol

        ∂gauge    = norm(view(Vd∂V, mask), Inf)
        ∂gauge    < gauge_atol || @warn "`eig` cotangents sensitive to gauge choice: (|Δgauge| = $∂gauge)"
        F         = one(eltype(dDD)) ./ dDD
        diagview(F) .= zero(eltype(F))
        conj!(F)
        Vd∂V    .*= F
        diagview(Vd∂V) .+= diagview(∂Dmat)

        P∂V = V' \ Vd∂V
        if eltype(A.dval) <: Real
            ∂Ac = mul!(Vd∂V, P∂V, V') # recycle VdΔV memory
            A.dval .+= real.(∂Ac)
        else
            mul!(A.dval, P∂V, V', 1, 1)
        end
        make_zero!(DV.dval)
    end
    return (nothing, nothing, nothing)
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
        MatrixAlgebraKit.eigh_full_pullback!(A.dval, DVval, ∂DV; kwargs...)
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
