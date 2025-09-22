module MatrixAlgebraKitEnzymeExt

using MatrixAlgebraKit
using MatrixAlgebraKit: diagview, inv_safe
using ChainRulesCore
using Enzyme
using Enzyme.EnzymeCore
using Enzyme.EnzymeCore: EnzymeRules
using LinearAlgebra

@inline EnzymeRules.inactive_type(v::Type{<:MatrixAlgebraKit.AbstractAlgorithm}) = true

Enzyme.@import_rrule(typeof(MatrixAlgebraKit.copy_input), Any, AbstractMatrix)

#Enzyme.@import_rrule(typeof(MatrixAlgebraKit.eigh_vals!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm)
#Enzyme.@import_rrule(typeof(MatrixAlgebraKit.eigh_full!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm)

#Enzyme.@import_rrule(typeof(MatrixAlgebraKit.eig_vals!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm)

#Enzyme.@import_rrule(typeof(MatrixAlgebraKit.svd_trunc!), AbstractMatrix, Any, MatrixAlgebraKit.TruncatedAlgorithm)

#Enzyme.@import_rrule(typeof(MatrixAlgebraKit.left_polar!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm)
#Enzyme.@import_rrule(typeof(MatrixAlgebraKit.right_polar!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm)

#=function EnzymeRules.forward(config::EnzymeRules.FwdConfig,
                             func::Const{typeof(qr_full!)},
                             ::Type{RT},
                             A::Annotation{<:AbstractMatrix},
                             QR::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}},
                             alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                             kwargs...,
                            ) where {RT}
    # adapted from ChainRules
    ret  = func.val(A.val, USVᴴ.val, alg.val; kwargs...)
    Q, R = ret
    m, n = size(A)

    F = inv_safe.(((S.diag') .^2)  .- (S.diag .^ 2), tol)
    invS = inv(S)
    ∂U = U * ( F .* (∂S * S .+ S * ∂S)) .+ (diagm(ones(eltype(U), m)) - U*U') * A.dval * V * invS
    ∂V = V * ( F .* (S * ∂S .+ ∂S * S)) .+ (diagm(ones(eltype(U), n)) - V*Vᴴ) * A.dval' * U * invS

    shadow = (∂U, ∂S, ∂Vᴴ)
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated(ret, shadow)
    elseif EnzymeRules.needs_shadow(config)
        return shadow
    elseif EnzymeRules.needs_primal(config)
        return ret
    else
        return nothing
    end
end=#

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
    cache_N = nothing
    # form cache if needed
    cache_A = (EnzymeRules.overwritten(config)[2] && !(typeof(N) <: Const)) ? copy(A.val)  : nothing
    func.val(A.val, N.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? N.val : nothing
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
                Q, R = QRval
                ∂Q, ∂R = ∂QR
                ∂A = A.dval
                ∂A .= zero(eltype(A.dval))
                m = size(Q, 1)
                n = size(R, 2)
                minmn = min(m, n)
                Rd = diagview(R)
                p = findlast(>=(rank_atol) ∘ abs, Rd)

                Q1 = view(Q, :, 1:p)
                Q2 = view(Q, :, (p + 1):size(Q, 2))
                R11 = view(R, 1:p, 1:p)
                ∂A1 = view(∂A, :, 1:p)
                ∂A2 = view(∂A, :, (p + 1):n)

                if minmn > p # case where A is rank-deficient
                    ∂gauge = abs(zero(eltype(Q)))
                    #if !iszerotangent(∂Q)
                        # in this case the number Householder reflections will
                        # change upon small variations, and all of the remaining
                        # columns of ∂Q should be zero for a gauge-invariant
                        # cost function
                    ∂Q2 = view(∂Q, :, (p + 1):size(Q, 2))
                    ∂gauge = max(∂gauge, norm(∂Q2, Inf))
                    #end
                    #if !iszerotangent(∂R)
                    ∂R22 = view(∂R, (p + 1):minmn, (p + 1):n)
                    ∂gauge = max(∂gauge, norm(∂R22, Inf))
                    #end
                    ∂gauge < gauge_atol ||
                        @warn "`$(string($f))` cotangents sensitive to gauge choice: (|∂gauge| = $∂gauge)"
                end

                ∂Q̃ = fill!(similar(Q, (m, p)), zero(eltype(Q)))
                copy!(∂Q̃, view(∂Q, :, 1:p))
                if p < size(Q, 2)
                    Q2 = view(Q, :, (p + 1):size(Q, 2))
                    ∂Q2 = view(∂Q, :, (p + 1):size(Q, 2))
                    # in the case where A is full rank, but there are more columns in Q than in A
                    # (the case of `qr_full`), there is gauge-invariant information in the
                    # projection of ΔQ2 onto the column space of Q1, by virtue of Q being a unitary
                    # matrix. As the number of Householder reflections is in fixed in the full rank
                    # case, Q is expected to rotate smoothly (we might even be able to predict) also
                    # how the full Q2 will change, but this we omit for now, and we consider
                    # Q2' * ΔQ2 as a gauge dependent quantity.
                    Q1d∂Q2 = Q1' * ∂Q2
                    ∂gauge = norm(mul!(copy(∂Q2), Q1, Q1d∂Q2, -1, 1), Inf)
                    ∂gauge < tol ||
                        @warn "`$(string($f))` cotangents sensitive to gauge choice: (|∂gauge| = $∂gauge)"
                    ∂Q̃ = mul!(∂Q̃, Q2, Q1d∂Q2', -1, 1)
                end
                if n > p
                    R12  = view(R, 1:p, (p + 1):n)
                    ∂R12 = view(∂R, 1:p, (p + 1):n)
                    ∂Q̃   = mul!(∂Q̃, Q1, ∂R12 * R12', -1, 1)
                    # Adding ∂A2 contribution
                    ∂A2  = mul!(∂A2, Q1, ∂R12, 1, 1)
                end

                # construct M
                M = fill!(similar(R, (p, p)), zero(eltype(R)))
                ∂R11 = view(∂R, 1:p, 1:p)
                M = mul!(M, ∂R11, R11', 1, 1)
                M = mul!(M, Q1', ∂Q̃, -1, 1)
                view(M, MatrixAlgebraKit.lowertriangularind(M)) .= conj.(view(M, MatrixAlgebraKit.uppertriangularind(M)))
                if eltype(M) <: Complex
                    Md = diagview(M)
                    Md .= real.(Md)
                end
                rdiv!(M, UpperTriangular(R11)')
                rdiv!(∂Q̃, UpperTriangular(R11)')
                ∂A1 = mul!(∂A1, Q1, M, +1, 1)
                ∂A1 .+= ∂Q̃
            end
            if !isa(QR, Const)
                make_zero!(QR.dval)
            end
            return (nothing, nothing, nothing)
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
        #FSdS = F .* (∂S * S .+ S * ∂S)
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

#=
function EnzymeRules.forward(config::EnzymeRules.FwdConfig,
                             func::Const{typeof(eigh_full!)},
                             ::Type{RT},
                             A::Annotation{<:AbstractMatrix},
                             DV::Annotation{<:Tuple{<:Diagonal, <:AbstractMatrix}},
                             alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                             kwargs...,
                            ) where {RT}
    # adapted from ChainRules
    ret     = func.val(A.val, DV.val; kwargs...)
    Dmat, V = ret
    if isa(A, Const) || all(iszero, A.dval)
        make_zero!(DV.dval[1])
        make_zero!(DV.dval[2])
        make_zero!(A.dval)
        shadow = (DV.dval[1], DV.dval[2])
    else
        tmpV    = V \ A.dval
        ∂K      = tmpV * V
        ∂Kdiag  = diag(∂K)
        ∂Ddiag  = zeros(eltype(Dmat), size(Dmat, 1))
        ∂Ddiag .= eltype(Dmat) <: Real ? real.(∂Kdiag) : ∂Kdiag
        D       = diagview(Dmat)
        dDD          = transpose(D) .- D
        F            = one(eltype(dDD)) ./ dDD
        diagview(F) .= zero(eltype(F))
        ∂K         .*= conj.(F)
        ∂V           = mul!(tmpV, V, ∂K) 
        A.dval      .= zero(eltype(A.val))
        shadow = (Diagonal(∂Ddiag), ∂V)
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
=#
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
    ret = func.val(A.val, DV.val, alg.val; kwargs...)

    if isa(A, Const) || all(iszero, A.dval)
        make_zero!(DV.dval[1])
        make_zero!(DV.dval[2])
        make_zero!(A.dval)
        shadow = (DV.dval[1], DV.dval[2])
    else
        Dmat, V = ret
        D            = diagview(Dmat)
        ∂K           = inv(V) * A.dval * V
        ∂Kdiag       = diag(∂K)
        ∂Ddiag       = zeros(eltype(D), size(D, 1))
        ∂Ddiag      .= eltype(D) <: Real ? real.(∂Kdiag) : ∂Kdiag
        ∂D           = Diagonal(∂Ddiag) 

        dDD          = transpose(D) .- D
        F            = one(eltype(dDD)) ./ dDD
        diagview(F) .= zero(eltype(F))
        ∂K         .*= conj.(F)
        ∂V           = V * ∂K
        A.dval      .= zero(eltype(A.val))
        DV.dval[2]  .= ∂V
        shadow       = (∂D, ∂V)
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

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                      func::Const{typeof(eig_full!)},
                                      ::Type{RT},
                                      A::Annotation{<:AbstractMatrix},
                                      DV::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}},
                                      alg::Annotation{<:MatrixAlgebraKit.AbstractAlgorithm};
                                      kwargs...,
                                     ) where {RT}
    cache_DV        = nothing
    func.val(A.val, DV.val, alg.val; kwargs...)
    primal = if EnzymeRules.needs_primal(config)
        DV.val
    else
        nothing
    end
    shadow = if EnzymeRules.needs_shadow(config)
        DV.dval
    else
        nothing
    end
    # form cache if needed
    cache_A = (EnzymeRules.overwritten(config)[2] && !(typeof(DV) <: Const)) ? copy(A.val)  : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_DV))
end
#=
function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                      func::Const{typeof(eigh_full!)},
                                      ::Type{RT},
                                      A::Annotation{<:AbstractMatrix},
                                      DV::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}},
                                      alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                                      kwargs...,
                                     ) where {RT}
    cache_DV        = nothing
    # form cache if needed
    cache_A = copy(A.val)
    func.val(A.val, DV.val, alg.val; kwargs...)
    primal = if EnzymeRules.needs_primal(config)
        DV.val
    else
        nothing
    end
    shadow = if EnzymeRules.needs_shadow(config)
        DV.dval
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_DV))
end
=#

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

#=
function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
                             func::Const{typeof(eigh_full!)},
                             ::Type{RT},
                             cache,
                             A::Annotation{<:AbstractMatrix},
                             DV::Annotation{<:Tuple{<:Diagonal, <:AbstractMatrix}},
                             alg::Const{<:MatrixAlgebraKit.AbstractAlgorithm};
                             kwargs...,
                            ) where {RT}
    tol             = MatrixAlgebraKit.default_pullback_gaugetol(DV.val[1])
    degeneracy_atol = tol
    gauge_atol      = tol

    cache_A, cache_DV = cache
    DVval   = !isnothing(cache_DV) ? cache_DV : DV.val
    Aval    = !isnothing(cache_A)  ? cache_A  : A.val
    ∂DV     = isa(DV, Const) ? nothing : DV.dval
    A.dval .= zero(eltype(A.dval))
    if !isa(A, Const) && !isa(DV, Const)
        Dmat, V   = DVval
        ∂Dmat, ∂V = ∂DV
        D      = diagview(Dmat)
        ∂D     = diagview(∂Dmat)
        VdΔV   = V' * ∂V
        aVdΔV  = rmul!(VdΔV - VdΔV', 1 / 2)
        mask   = abs.(D' .- D) .< degeneracy_atol
        Δgauge = norm(view(aVdΔV, mask))
        Δgauge < gauge_atol || @warn "`eigh` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"

        aVdΔV .*= MatrixAlgebraKit.inv_safe.(D' .- D, tol)
        diagview(aVdΔV) .+= real.(∂D)
        # recylce VdΔV space
        mul!(VdΔV, V, aVdΔV)
        A.dval .+= VdΔV * V'
        A.val .= cache_A
    end
    if !isa(DV, Const)
        make_zero!(DV.dval)
    end
    return (nothing, nothing, nothing)
end
=#
end
