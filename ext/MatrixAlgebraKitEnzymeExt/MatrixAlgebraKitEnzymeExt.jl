module MatrixAlgebraKitEnzymeExt

using MatrixAlgebraKit
using MatrixAlgebraKit: diagview
using ChainRulesCore
using Enzyme
using Enzyme.EnzymeCore
using Enzyme.EnzymeCore: EnzymeRules
using LinearAlgebra

@inline EnzymeRules.inactive_type(v::Type{<:MatrixAlgebraKit.AbstractAlgorithm}) = true

Enzyme.@import_rrule(typeof(MatrixAlgebraKit.copy_input), Any, AbstractMatrix)

#=Enzyme.@import_rrule(typeof(MatrixAlgebraKit.qr_null!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm)
Enzyme.@import_rrule(typeof(MatrixAlgebraKit.qr_full!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm)
Enzyme.@import_rrule(typeof(MatrixAlgebraKit.qr_compact!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm)
=#
Enzyme.@import_rrule(typeof(MatrixAlgebraKit.lq_null!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm)
Enzyme.@import_rrule(typeof(MatrixAlgebraKit.lq_full!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm)
Enzyme.@import_rrule(typeof(MatrixAlgebraKit.lq_compact!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm)

#Enzyme.@import_rrule(typeof(MatrixAlgebraKit.eigh_vals!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm)

#Enzyme.@import_rrule(typeof(MatrixAlgebraKit.eig_full!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm)
#Enzyme.@import_rrule(typeof(MatrixAlgebraKit.eig_vals!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm)

#Enzyme.@import_rrule(typeof(MatrixAlgebraKit.svd_full!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm)
#Enzyme.@import_rrule(typeof(MatrixAlgebraKit.svd_compact!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm)
#Enzyme.@import_rrule(typeof(MatrixAlgebraKit.svd_vals!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm)
#Enzyme.@import_rrule(typeof(MatrixAlgebraKit.svd_trunc!), AbstractMatrix, Any, MatrixAlgebraKit.TruncatedAlgorithm)

Enzyme.@import_rrule(typeof(MatrixAlgebraKit.left_polar!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm)
Enzyme.@import_rrule(typeof(MatrixAlgebraKit.right_polar!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm)

function EnzymeRules.forward(config::EnzymeRules.FwdConfig,
                             func::Const{typeof(qr_full!)},
                             ::Type{RT},
                             A::Annotation{<:AbstractMatrix},
                             QR::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}};
                             kwargs...,
                            ) where {RT}
    # adapted from ChainRules
    ret  = func.val(A.val, USVᴴ.val; kwargs...)
    Q, R = ret
    m, n     = size(A)

    F = MatrixAlgebraKit.inv_safe.(((S.diag') .^2)  .- (S.diag .^ 2), tol)
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
end

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                      func::Const{typeof(qr_full!)},
                                      ::Type{RT},
                                      A::Annotation{<:AbstractMatrix},
                                      QR::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}};
                                      kwargs...,
                                     ) where {RT}
    cache_QR = nothing
    func.val(A.val, QR.val; kwargs...)
    primal = if EnzymeRules.needs_primal(config)
        QR.val
    else
        nothing
    end
    shadow = if EnzymeRules.needs_shadow(config)
        QR.dval
    else
        nothing
    end
    # form cache if needed
    cache_A = (EnzymeRules.overwritten(config)[2] && !(typeof(QR) <: Const)) ? copy(A.val)  : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_QR))
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
                             func::Const{typeof(qr_full!)},
                             dret::Type{RT},
                             cache,
                             A::Annotation{<:AbstractMatrix},
                             QR::Annotation{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}};
                             tol::Real=MatrixAlgebraKit.default_pullback_gaugetol(QR.val[2]),
                             rank_atol::Real=tol,
                             gauge_atol::Real=tol,
                             kwargs...) where {RT}
    cache_A, cache_QR = cache
    QRval = !isnothing(cache_QR) ? cache_QR : QR.val
    ∂QR   = isa(QR, Const) ? nothing : QR.dval
    if !isa(A, Const) && !isa(QR, Const)
        Q, R = QRval
        ∂Q, ∂R = ∂QR
        ∂A = A.dval
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
                @warn "`qr` cotangents sensitive to gauge choice: (|∂gauge| = $∂gauge)"
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
                @warn "`qr` cotangents sensitive to gauge choice: (|∂gauge| = $∂gauge)"
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
        #return ∂A
    end
    if !isa(QR, Const)
        make_zero!(QR.dval)
    end
    return (nothing, nothing)
end
function EnzymeRules.forward(config::EnzymeRules.FwdConfig,
                             func::Const{typeof(svd_full!)},
                             ::Type{RT},
                             A::Annotation{<:AbstractMatrix},
                             USVᴴ::Annotation{<:Tuple{<:AbstractMatrix, <:Diagonal, <:AbstractMatrix}};
                             kwargs...,
                            ) where {RT}
    # adapted from ChainRules
    ret      = func.val(A.val, USVᴴ.val; kwargs...)
    U, S, Vᴴ = ret
    V        = adjoint(Vᴴ)
    ∂S       = Diagonal(U' * A.dval * V)

    m, n     = size(A)

    F = MatrixAlgebraKit.inv_safe.(((S.diag') .^2)  .- (S.diag .^ 2), tol)
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
end

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                      func::Const{typeof(svd_full!)},
                                      ::Type{RT},
                                      A::Annotation{<:AbstractMatrix},
                                      USVᴴ::Annotation{<:Tuple{<:AbstractMatrix, <:Diagonal, <:AbstractMatrix}};
                                      kwargs...,
                                     ) where {RT}
    tol             = 1e-8 #default_pullback_gaugetol(DV.val[1]),
    degeneracy_atol = tol
    gauge_atol      = tol
    cache_DV        = nothing
    func.val(A.val, USVᴴ.val; kwargs...)
    primal = if EnzymeRules.needs_primal(config)
        USVᴴ.val
    else
        nothing
    end
    shadow = if EnzymeRules.needs_shadow(config)
        USVᴴ.dval
    else
        nothing
    end
    # form cache if needed
    cache_A  = (EnzymeRules.overwritten(config)[2] && !(typeof(USVᴴ) <: Const)) ? copy(A.val)  : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_USVᴴ))
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
                             func::Const{typeof(svd_full!)},
                             dret::Type{RT},
                             cache,
                             A::Annotation{<:AbstractMatrix},
                             USVᴴ::Annotation{<:Tuple{<:AbstractMatrix, <:Diagonal, <:AbstractMatrix}};
                             kwargs...,
                            ) where {RT}
    tol             = 1e-8 #default_pullback_gaugetol(DV.val[1]),
    degeneracy_atol = tol
    gauge_atol      = tol

    cache_A, cache_USVᴴ = cache
    USVᴴval = !isnothing(cache_USVᴴ) ? cache_USVᴴ : USVᴴ.val
    ∂USVᴴ   = isa(USVᴴ, Const) ? nothing : USVᴴ.dval
    if !isa(A, Const) && !isa(USVᴴ, Const)
        U, S, Vᴴ = USVᴴval
        ∂U, ∂S, ∂Vᴴ = ∂USVᴴ
        A.dval .+= U * ∂S * Vᴴ
    end
    if !isa(USVᴴ, Const)
        make_zero!(USVᴴ.dval)
    end
    return (nothing, nothing)
end

function EnzymeRules.forward(config::EnzymeRules.FwdConfig,
                             func::Const{typeof(eigh_full!)},
                             ::Type{RT},
                             A::Annotation{<:AbstractMatrix},
                             DV::Annotation{<:Tuple{<:Diagonal, <:AbstractMatrix}};
                             kwargs...,
                            ) where {RT}
    # adapted from ChainRules
    ret    = func.val(A.val, DV.val; kwargs...)
    D, V   = ret
    tmpV   = V \ A.dval
    ∂K     = tmpV * V
    ∂Kdiag = diagview(∂K)
    ∂D     = eltype(D) <: Real ? Diagonal(real.(∂Kdiag)) : Diagonal(∂Kdiag)

    ∂K    ./= transpose(D) .- D
    ∂Kdiag .= zero(eltype(D))
    
    ∂V = mul!(tmpV, V, ∂K) 

    shadow = (∂D, ∂V)
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

function EnzymeRules.forward(config::EnzymeRules.FwdConfig,
                             func::Const{typeof(eig_full!)},
                             ::Type{RT},
                             A::Annotation{<:AbstractMatrix},
                             DV::Annotation{<:Tuple{<:Diagonal, <:AbstractMatrix}};
                             kwargs...,
                            ) where {RT}
    # adapted from ChainRules
    ret    = func.val(A.val, DV.val; kwargs...)
    D, V   = ret
    tmpV   = V \ A.dval
    ∂K     = tmpV * V
    ∂Kdiag = diagview(∂K)
    ∂D     = eltype(D) <: Real ? Diagonal(real.(∂Kdiag)) : Diagonal(∂Kdiag)

    ∂K    ./= transpose(D) .- D
    ∂Kdiag .= zero(eltype(D))
    
    ∂V = mul!(tmpV, V, ∂K) 

    shadow = (∂D, ∂V)
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
                                      DV::Annotation{<:Tuple{<:Diagonal, <:AbstractMatrix}};
                                      kwargs...,
                                     ) where {RT}
    tol             = 1e-8 #default_pullback_gaugetol(DV.val[1]),
    degeneracy_atol = tol
    gauge_atol      = tol
    cache_DV        = nothing
    func.val(A.val, DV.val; kwargs...)
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
    cache_A  = (EnzymeRules.overwritten(config)[2] && !(typeof(DV) <: Const)) ? copy(A.val)  : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_DV))
end


function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                      func::Const{typeof(eigh_full!)},
                                      ::Type{RT},
                                      A::Annotation{<:AbstractMatrix},
                                      DV::Annotation{<:Tuple{<:Diagonal, <:AbstractMatrix}};
                                      kwargs...,
                                     ) where {RT}
    tol             = 1e-8 #default_pullback_gaugetol(DV.val[1]),
    degeneracy_atol = tol
    gauge_atol      = tol
    cache_DV        = DV.val 
    func.val(A.val, DV.val; kwargs...)
    primal = if EnzymeRules.needs_primal(config)
        DV.val
    else
        nothing
    end
    shadow = if EnzymeRules.needs_shadow(config)
        if isa(A, Const) && isa(DV, Const)
            nothing
        elseif isa(A, Const) && !isa(DV, Const)
            make_zero!(DV.dval)
        elseif isa(DV, Const)
            make_zero!(DV.val)
        else
            DV.dval
        end
    else
        nothing
    end
    isa(A, Const) && make_zero!(DV.dval)
    # form cache if needed
    cache_A  = EnzymeRules.overwritten(config)[2] ? copy(A.val)  : nothing
    cache_DV = !isa(DV, Const) && EnzymeRules.overwritten(config)[3] ? copy(DV.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_DV))
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
                             func::Const{typeof(eig_full!)},
                             dret::Type{RT},
                             cache,
                             A::Annotation{<:AbstractMatrix},
                             DV::Annotation{<:Tuple{<:Diagonal, <:AbstractMatrix}};
                             kwargs...,
                            ) where {RT}
    tol             = 1e-8 #default_pullback_gaugetol(DV.val[1]),
    degeneracy_atol = tol
    gauge_atol      = tol

    cache_A, cache_DV = cache
    DVval = !isnothing(cache_DV) ? cache_DV : DV.val

    ∂A    = A.dval .+ dret.val 
    ∂DV   = DV.dval
    ∂A = if !isa(A, Const) && !isa(DV, Const) 
        D, V   = DVval
        ∂D, ∂V = ∂DV
        Vd∂V   = adjoint(V) * ∂V

        dDD    = transpose(D) .- D
        mask   = abs.(dDD) .< degeneracy_atol

        ∂gauge = norm(view(Vd∂V, mask), Inf)
        ∂gauge < gauge_atol || @warn "`eig` cotangents sensitive to gauge choice: (|Δgauge| = $∂gauge)"

        Vd∂V .*= conj.(MatrixAlgebraKit.inv_safe.(dDD, degeneracy_atol))
        diagview(Vd∂V) .+= diagview(∂D)
        
        P∂V = adjoint(V) \ Vd∂V
        if eltype(∂A) <: Real
            ∂Ac = mul!(Vd∂V, P∂V, adjoint(V)) # recycle VdΔV memory
            ∂A .+= real.(∂Ac)
        else
            ∂A = mul!(∂A, P∂V, adjoint(V), 1, 1)
        end
        A.dval .= ∂A
        make_zero!(DV.dval)
    else
        nothing
    end
    return (nothing, nothing)
end

# TODO support batch reverse
function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
                             func::Const{typeof(eigh_full!)},
                             ::Type{RT},
                             cache,
                             A::Annotation{<:AbstractMatrix},
                             DV::Annotation{<:Tuple{<:Diagonal, <:AbstractMatrix}};
                             kwargs...,
                            ) where {RT}
    tol             = 1e-8 #default_pullback_gaugetol(DV.val[1]),
    degeneracy_atol = tol
    gauge_atol      = tol

    cache_A, cache_DV = cache
    DVval = !isnothing(cache_DV) ? cache_DV : DV.val
    ∂DV   = isa(DV, Const) ? nothing : DV.dval
    if !isa(A, Const) && !isa(DV, Const)
        D, V   = DVval
        ∂D, ∂V = ∂DV
        VdΔV   = V' * ∂V 
        aVdΔV  = rmul!(VdΔV - VdΔV', 1 / 2)
        mask   = abs.(D.diag' .- D.diag) .< degeneracy_atol
        Δgauge = norm(view(aVdΔV, mask))
        Δgauge < gauge_atol || @warn "`eigh` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"

        aVdΔV .*= MatrixAlgebraKit.inv_safe.(D.diag' .- D.diag, tol)
        diagview(aVdΔV) .+= real.(diagview(∂D))
        # recylce VdΔV space
        mul!(A.dval, mul!(VdΔV, V, aVdΔV), V', 1, 1)
    end
    if !isa(DV, Const)
        make_zero!(DV.dval)
    end
    return (nothing, nothing)
end

end
        #=if !all(iszero.(∂V))
            Vd∂V   = V' * ∂V
            aVd∂V  = rmul!(Vd∂V - Vd∂V', 1 / 2)

            mask   = abs.(D.diag' .- D.diag) .< degeneracy_atol
            ∂gauge = norm(view(aVd∂V, mask))
            ∂gauge < gauge_atol ||
                @warn "`eigh` cotangents sensitive to gauge choice: (|∂gauge| = $∂gauge)"

            aVd∂V .*= MatrixAlgebraKit.inv_safe.(D.diag' .- D.diag, tol)

            if !all(iszero.(∂D))
                diagview(aVd∂V) .+= real.(diagview(∂D))
            end
            mul!(A.dval, mul!(Vd∂V, V, aVd∂V), V', 1, 1)
        elseif !all(iszero.(∂D))
            mul!(A.dval, V * Diagonal(real(diagview(∂D))), V', 1, 1)
        end=#
        
        #=Atmp = similar(A.val, eltype(V))
        tmp = ∂V
        ∂K = mul!(Atmp, V', ∂V) ∂K .*= MatrixAlgebraKit.inv_safe.(D.diag' .- D.diag, tol) ∂K[diagind(∂K)] .+= real.(∂D.diag) mul!(tmp, ∂K, V') mul!(A.dval, V, tmp, 1, 1)=#
