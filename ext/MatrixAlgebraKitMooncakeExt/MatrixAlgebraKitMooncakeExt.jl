module MatrixAlgebraKitMooncakeExt

using Mooncake
using Mooncake: @from_chainrules, DefaultCtx, CoDual, Dual, NoRData, rrule!!, frule!!, arrayify, @is_primitive
using MatrixAlgebraKit
using MatrixAlgebraKit: inv_safe, diagview
using MatrixAlgebraKit.YALAPACK
using ChainRulesCore
using LinearAlgebra
using LinearAlgebra: BlasFloat, BlasComplex, diagind

@from_chainrules DefaultCtx Tuple{typeof(MatrixAlgebraKit.qr_null!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm}
@from_chainrules DefaultCtx Tuple{typeof(MatrixAlgebraKit.qr_full!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm}
@from_chainrules DefaultCtx Tuple{typeof(MatrixAlgebraKit.qr_compact!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm}

@from_chainrules DefaultCtx Tuple{typeof(MatrixAlgebraKit.lq_null!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm}
@from_chainrules DefaultCtx Tuple{typeof(MatrixAlgebraKit.lq_full!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm}
@from_chainrules DefaultCtx Tuple{typeof(MatrixAlgebraKit.lq_compact!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm}

@from_chainrules DefaultCtx Tuple{typeof(MatrixAlgebraKit.eigh_full!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm}
@from_chainrules DefaultCtx Tuple{typeof(MatrixAlgebraKit.eigh_vals!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm}

#@from_chainrules DefaultCtx Tuple{typeof(MatrixAlgebraKit.eig_full!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm}
@from_chainrules DefaultCtx Tuple{typeof(MatrixAlgebraKit.eig_vals!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm}

Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(MatrixAlgebraKit.copy_input), Any, AbstractMatrix}

# TODO THIS IS BAD!!!!
function MatrixAlgebraKit.diagview(dx::Tangent)
    if isa(dx, ChainRulesCore.Tangent)
        if isa(dx.diag, Vector{<:Real})
            return dx.diag
        elseif isa(dx.diag, Vector{Tangent{<:Any, Vector{@NamedTuple{re::Float64, im::Float64}}}})
            return [complex(dxd.re, dxd.im) for dxd in dx.diag]
        end
        return dx.diag
    else
        hasfield(Mooncake._fields(dx), :diag) && return Mooncake._fields(dx).diag
    end
    throw(ErrorException(""))
end

#Base.one(::Type{Tangent{Any, @NamedTuple{re::ComplexF64, im::ComplexF64}}}) = one(ComplexF64)


# redo all of this because of no `one` method for Tangents... hmmmm
@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(MatrixAlgebraKit.eig_full!), AbstractMatrix, Tuple{<:Diagonal, <:AbstractMatrix}, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(::CoDual{typeof(MatrixAlgebraKit.eig_full!)}, A_dA::CoDual, DV_dDV::CoDual, alg_dalg::CoDual;
                 tol::Real=MatrixAlgebraKit.default_pullback_gaugetol(Mooncake.primal(DV_dDV)[1]),
                 degeneracy_atol::Real=tol,
                 gauge_atol::Real=tol,
                 kwargs...)
    A, dA   = arrayify(A_dA)
    ∂A      = zero(A)
    DV      = Mooncake.primal(DV_dDV)
    dDV     = Mooncake.tangent(DV_dDV)
    D, V    = DV
    D, dD   = arrayify(D, dDV[1])
    V, dV   = arrayify(V, dDV[2])
    function deig_adjoint(::Mooncake.NoRData)
        if !isa(dV, Mooncake.NoTangent)
            VdΔV   = V' * dV
            mask   = abs.(transpose(D.diag) .- D.diag) .< degeneracy_atol
            Δgauge = norm(view(VdΔV, mask), Inf)
            Δgauge < gauge_atol ||
                @warn "`eig` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
            VdΔV .*= conj.(inv_safe.(transpose(D.diag) .- D.diag, degeneracy_atol))

            if !isa(dD, Mooncake.NoTangent)
                VdΔV[diagind(VdΔV)] .+= diagview(dD)
            end
            PΔV = V' \ VdΔV
            if eltype(∂A) <: Real
                ∂Ac = mul!(VdΔV, PΔV, V') # recycle VdΔV memory
                ∂A .+= real.(∂Ac)
            else
                ∂A = mul!(∂A, PΔV, V', 1, 1)
            end
        elseif !isa(dD, Mooncake.NoTangent)
            PΔV = V' \ Diagonal(diagview(dD))
            if eltype(∂A) <: Real
                ∂Ac = PΔV * V'
                ∂A .+= real.(∂Ac)
            else
                ∂A = mul!(∂A, PΔV, V', 1, 1)
            end
        end
        dA .= ∂A
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    DV = eig_full!(A, DV, Mooncake.primal(alg_dalg); kwargs...)
    return Mooncake.CoDual(DV, dDV), deig_adjoint
end
#=
@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(MatrixAlgebraKit.eigh_full!), AbstractMatrix, Tuple{<:Diagonal, <:AbstractMatrix}, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(::CoDual{typeof(MatrixAlgebraKit.eigh_full!)}, A_dA::CoDual{<:AbstractMatrix}, DV_dDV::CoDual{<:Tuple{<:Diagonal, <:AbstractMatrix}}, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm};
                 tol::Real=MatrixAlgebraKit.default_pullback_gaugetol(Mooncake.primal(DV_dDV)[1]),
                 degeneracy_atol::Real=tol,
                 gauge_atol::Real=tol,
                 kwargs...)
    A, dA   = arrayify(A_dA)
    ∂A      = zero(A)
    DV      = Mooncake.primal(DV_dDV)
    dDV     = Mooncake.tangent(DV_dDV)
    D, V    = DV
    D, dD   = arrayify(D, dDV[1])
    V, dV   = arrayify(V, dDV[2])
    function deigh_adjoint(::Mooncake.NoRData)
        if !isa(dV, Mooncake.NoTangent)
            VdΔV   = V' * dV
            aVdΔV  = rmul!(VdΔV - VdΔV', 1 / 2)
            mask   = abs.(transpose(D.diag) .- D.diag) .< degeneracy_atol
            Δgauge = norm(view(aVdΔV, mask))
            Δgauge < gauge_atol || @warn "`eigh` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
            VdΔV .*= conj.(inv_safe.(transpose(D.diag) .- D.diag, degeneracy_atol))

            aVdΔV .*= MatrixAlgebraKit.inv_safe.(D.diag' .- D.diag, tol)
            if !isa(dD, Mooncake.NoTangent)
                aVdΔV[diagind(aVdΔV)] .+= real.(diagview(dD))
            end
            dA = mul!(dA, mul!(VdΔV, V, aVdΔV), V', 1, 1)
        elseif !isa(dD, Mooncake.NoTangent)
            dA = mul!(dA, V * Diagonal(real(diagview(d))), V', 1, 1)
        end
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    DV = eigh_full!(A, DV, Mooncake.primal(alg_dalg); kwargs...)
    return Mooncake.CoDual(DV, dDV), deigh_adjoint
end
=#

@from_chainrules DefaultCtx Tuple{typeof(MatrixAlgebraKit.svd_full!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm}
@from_chainrules DefaultCtx Tuple{typeof(MatrixAlgebraKit.svd_compact!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm}
@from_chainrules DefaultCtx Tuple{typeof(MatrixAlgebraKit.svd_vals!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm}
@from_chainrules DefaultCtx Tuple{typeof(MatrixAlgebraKit.svd_trunc!), AbstractMatrix, Any, MatrixAlgebraKit.TruncatedAlgorithm}
end
