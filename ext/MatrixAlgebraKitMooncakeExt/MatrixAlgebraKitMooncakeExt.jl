module MatrixAlgebraKitMooncakeExt

using Mooncake
using Mooncake: DefaultCtx, CoDual, Dual, NoRData, rrule!!, frule!!, arrayify, @is_primitive
using MatrixAlgebraKit
using MatrixAlgebraKit: inv_safe, diagview
using MatrixAlgebraKit: svd_pushforward! 
using MatrixAlgebraKit: qr_pullback!, lq_pullback!, qr_pushforward!, lq_pushforward!
using MatrixAlgebraKit: qr_null_pullback!, lq_null_pullback!, qr_null_pushforward!, lq_null_pushforward!
using MatrixAlgebraKit: eig_pullback!, eigh_pullback!, eig_pushforward!, eigh_pushforward!
using MatrixAlgebraKit: left_polar_pullback!, right_polar_pullback!, left_polar_pushforward!, right_polar_pushforward!
using LinearAlgebra

# two-argument factorizations like LQ, QR, EIG
for (f, pb, pf, adj) in ((qr_full!,    qr_pullback!, qr_pushforward!, :dqr_adjoint), 
                         (qr_compact!, qr_pullback!, qr_pushforward!, :dqr_adjoint),
                         (lq_full!,    lq_pullback!, lq_pushforward!, :dlq_adjoint), 
                         (lq_compact!, lq_pullback!, lq_pushforward!, :dlq_adjoint),
                         (eig_full!,   eig_pullback!, eig_pushforward!, :deig_adjoint),
                         (eigh_full!,  eigh_pullback!, eigh_pushforward!, :deigh_adjoint),
                         (left_polar!, left_polar_pullback!, left_polar_pushforward!, :dleft_polar_adjoint),
                         (right_polar!, right_polar_pullback!, right_polar_pushforward!, :dright_polar_adjoint),
                        )

    @eval begin
        @is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof($f), AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(::CoDual{typeof($f)}, A_dA::CoDual{<:AbstractMatrix}, args_dargs::CoDual, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm}; kwargs...)
            A, dA       = arrayify(A_dA)
            dA         .= zero(eltype(A))
            args        = Mooncake.primal(args_dargs)
            dargs       = Mooncake.tangent(args_dargs)
            arg1, darg1 = arrayify(args[1], dargs[1])
            arg2, darg2 = arrayify(args[2], dargs[2])
            function $adj(::Mooncake.NoRData)
                dA = $pb(dA, A, (arg1, arg2), (darg1, darg2); kwargs...)
                return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
            end
            args   = $f(A, args, Mooncake.primal(alg_dalg); kwargs...)
            darg1 .= zero(eltype(arg1))
            darg2 .= zero(eltype(arg2))
            return Mooncake.CoDual(args, dargs), $adj
        end
        @is_primitive Mooncake.DefaultCtx Mooncake.ForwardMode Tuple{typeof($f), AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.frule!!(::Dual{typeof($f)}, A_dA::Dual{<:AbstractMatrix}, args_dargs::Dual, alg_dalg::Dual{<:MatrixAlgebraKit.AbstractAlgorithm}; kwargs...)
            A, dA        = arrayify(A_dA)
            args         = Mooncake.primal(args_dargs)
            args         = $f(A, args, Mooncake.primal(alg_dalg); kwargs...)
            dargs        = Mooncake.tangent(args_dargs)
            arg1, darg1  = arrayify(args[1], dargs[1])
            arg2, darg2  = arrayify(args[2], dargs[2])
            darg1, darg2 = $pf(dA, A, (arg1, arg2), (darg1, darg2))
            dA          .= zero(eltype(A))
            return Mooncake.Dual(args, dargs)
        end
    end
end

for (f, f_full, pb, pf, adj) in ((qr_null!, qr_full, qr_null_pullback!, qr_null_pushforward!, :dqr_null_adjoint), 
                                 (lq_null!, lq_full, lq_null_pullback!, lq_null_pushforward!, :dlq_null_adjoint), 
                                )
    @eval begin
        @is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof($f), AbstractMatrix, AbstractMatrix, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(f_df::CoDual{typeof($f)}, A_dA::CoDual{<:AbstractMatrix}, arg_darg::CoDual{<:AbstractMatrix}, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm}; kwargs...)
            A, dA     = arrayify(A_dA)
            Ac        = MatrixAlgebraKit.copy_input($f_full, A)
            arg, darg = arrayify(Mooncake.primal(arg_darg), Mooncake.tangent(arg_darg))
            arg       = $f(Ac, arg, Mooncake.primal(alg_dalg))
            function $adj(::Mooncake.NoRData)
                dA   .= zero(eltype(A))
                $pb(dA, A, arg, darg; kwargs...)
                return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
            end
            return arg_darg, $adj 
        end
        @is_primitive Mooncake.DefaultCtx Mooncake.ForwardMode Tuple{typeof($f), AbstractMatrix, AbstractMatrix, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.frule!!(f_df::Dual{typeof($f)}, A_dA::Dual{<:AbstractMatrix}, arg_darg::Dual{<:AbstractMatrix}, alg_dalg::Dual{<:MatrixAlgebraKit.AbstractAlgorithm}; kwargs...)
            A, dA     = arrayify(A_dA)
            Ac        = MatrixAlgebraKit.copy_input($f_full, A)
            arg, darg = arrayify(Mooncake.primal(arg_darg), Mooncake.tangent(arg_darg))
            arg       = $f(Ac, arg, Mooncake.primal(alg_dalg))
            $pf(dA, A, arg, darg; kwargs...)
            dA       .= zero(dA)
            return arg_darg
        end
    end
end

@is_primitive Mooncake.DefaultCtx Mooncake.ForwardMode Tuple{typeof(MatrixAlgebraKit.eig_vals!), AbstractMatrix, AbstractVector, MatrixAlgebraKit.AbstractAlgorithm}
@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(MatrixAlgebraKit.eig_vals!), AbstractMatrix, AbstractVector, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.frule!!(::Dual{<:typeof(MatrixAlgebraKit.eig_vals!)}, A_dA::Dual, D_dD::Dual, alg_dalg::Dual; kwargs...)
    # compute primal
    D_    = Mooncake.primal(D_dD)
    dD_   = Mooncake.tangent(D_dD)
    A_    = Mooncake.primal(A_dA)
    dA_   = Mooncake.tangent(A_dA)
    A, dA = arrayify(A_, dA_)
    D, dD = arrayify(D_, dD_)
    nD, V = eig_full(A, alg_dalg.primal; kwargs...)

    # update tangent
    tmp   = V \ dA
    dD   .= diagview(tmp * V)
    dA   .= zero(eltype(dA))
    return Mooncake.Dual(nD.diag, dD_)
end

function Mooncake.rrule!!(::CoDual{<:typeof(MatrixAlgebraKit.eig_vals!)}, A_dA::CoDual, D_dD::CoDual, alg_dalg::CoDual; kwargs...)
    # compute primal
    D_    = Mooncake.primal(D_dD)
    dD_   = Mooncake.tangent(D_dD)
    A_    = Mooncake.primal(A_dA)
    dA_   = Mooncake.tangent(A_dA)
    A, dA = arrayify(A_, dA_)
    D, dD = arrayify(D_, dD_)
    dA   .= zero(eltype(dA))
    # update primal 
    DV  = eig_full(A, Mooncake.primal(alg_dalg); kwargs...)
    V   = DV[2]
    dD .= zero(eltype(D))
    function deig_vals_adjoint(::Mooncake.NoRData)
        PΔV = V' \ Diagonal(dD)
        if eltype(dA) <: Real
            ΔAc = PΔV * V'
            dA .+= real.(ΔAc)
        else
            mul!(dA, PΔV, V', 1, 0)
        end
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    return Mooncake.CoDual(DV[1].diag, dD_), deig_vals_adjoint
end
#=
@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(MatrixAlgebraKit.eigh_full!), AbstractMatrix, Tuple{<:Diagonal, <:AbstractMatrix}, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(::CoDual{typeof(MatrixAlgebraKit.eigh_full!)}, A_dA::CoDual{<:AbstractMatrix}, DV_dDV::CoDual{<:Tuple{<:Diagonal, <:AbstractMatrix}}, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm}; kwargs...)
    A, dA  = arrayify(A_dA)
    dA    .= zero(eltype(A))
    DV     = Mooncake.primal(DV_dDV)
    dDV    = Mooncake.tangent(DV_dDV)
    D, dD  = arrayify(DV[1], dDV[1])
    V, dV  = arrayify(DV[2], dDV[2])
    function deigh_adjoint(::Mooncake.NoRData)
        dA = MatrixAlgebraKit.eigh_pullback!(dA, A, (D, V), (dD, dV); kwargs...)
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    DV = eigh_full!(A, DV, Mooncake.primal(alg_dalg); kwargs...)
    return Mooncake.CoDual(DV, dDV), deigh_adjoint
end

@is_primitive Mooncake.DefaultCtx Mooncake.ForwardMode Tuple{typeof(MatrixAlgebraKit.eigh_full!), AbstractMatrix, Tuple{<:Diagonal, <:AbstractMatrix}, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.frule!!(::Dual{typeof(MatrixAlgebraKit.eigh_full!)}, A_dA::Dual, DV_dDV::Dual, alg_dalg::Dual; kwargs...)
    A, dA   = arrayify(A_dA)
    DV      = Mooncake.primal(DV_dDV)
    dDV     = Mooncake.tangent(DV_dDV)
    D, dD   = arrayify(DV[1], dDV[1])
    V, dV   = arrayify(DV[2], dDV[2])
    (D, V)  = eigh_full!(A, DV, Mooncake.primal(alg_dalg); kwargs...)
    (dD, dV) = eigh_pushforward!(dA, A, (D, V), (dD, dV); kwargs...)
    return Mooncake.Dual(DV, dDV)
end
=#

@is_primitive Mooncake.DefaultCtx Mooncake.ForwardMode Tuple{typeof(MatrixAlgebraKit.eigh_vals!), AbstractMatrix, AbstractVector, MatrixAlgebraKit.AbstractAlgorithm}
@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(MatrixAlgebraKit.eigh_vals!), AbstractMatrix, AbstractVector, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.frule!!(::Dual{<:typeof(MatrixAlgebraKit.eigh_vals!)}, A_dA::Dual, D_dD::Dual, alg_dalg::Dual; kwargs...)
    # compute primal
    D_    = Mooncake.primal(D_dD)
    dD_   = Mooncake.tangent(D_dD)
    A_    = Mooncake.primal(A_dA)
    dA_   = Mooncake.tangent(A_dA)
    A, dA = arrayify(A_, dA_)
    D, dD = arrayify(D_, dD_)
    nD, V = eigh_full(A, alg_dalg.primal; kwargs...)
    # update tangent
    tmp   = inv(V) * dA * V
    dD   .= real.(diagview(tmp))
    D    .= nD.diag
    dA   .= zero(eltype(dA))
    return D_dD
end

function Mooncake.rrule!!(::CoDual{<:typeof(MatrixAlgebraKit.eigh_vals!)}, A_dA::CoDual, D_dD::CoDual, alg_dalg::CoDual; kwargs...)
    # compute primal
    D_    = Mooncake.primal(D_dD)
    dD_   = Mooncake.tangent(D_dD)
    A_    = Mooncake.primal(A_dA)
    dA_   = Mooncake.tangent(A_dA)
    A, dA = arrayify(A_, dA_)
    D, dD = arrayify(D_, dD_)
    DV    = eigh_full(A, Mooncake.primal(alg_dalg); kwargs...)
    function deigh_vals_adjoint(::Mooncake.NoRData)
        mul!(dA, DV[2] * Diagonal(real(dD)), DV[2]', 1, 0)
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    return Mooncake.CoDual(DV[1].diag, dD_), deigh_vals_adjoint
end


for (f, St) in ((svd_full!, :AbstractMatrix), (svd_compact!, :Diagonal))
    @eval begin
        @is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof($f), AbstractMatrix, Tuple{<:AbstractMatrix, <:$St, <:AbstractMatrix}, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(::CoDual{typeof($f)}, A_dA::CoDual, USVᴴ_dUSVᴴ::CoDual, alg_dalg::CoDual; kwargs...)
            A, dA   = arrayify(A_dA)
            USVᴴ    = Mooncake.primal(USVᴴ_dUSVᴴ)
            dUSVᴴ   = Mooncake.tangent(USVᴴ_dUSVᴴ)
            U, dU   = arrayify(USVᴴ[1], dUSVᴴ[1])
            S, dS   = arrayify(USVᴴ[2], dUSVᴴ[2])
            Vᴴ, dVᴴ = arrayify(USVᴴ[3], dUSVᴴ[3])
            USVᴴ    = $f(A, USVᴴ, Mooncake.primal(alg_dalg); kwargs...)
            function dsvd_adjoint(::Mooncake.NoRData)
                dA   .= zero(eltype(A))
                minmn = min(size(A)...)
                if size(U, 2) == size(Vᴴ, 1) == minmn # compact
                    dA    = MatrixAlgebraKit.svd_pullback!(dA, A, (U, S, Vᴴ), (dU, dS, dVᴴ))
                else # full
                    vU    = view(U, :, 1:minmn)
                    vS    = Diagonal(diagview(S)[1:minmn])
                    vVᴴ   = view(Vᴴ, 1:minmn, :)
                    vdU   = view(dU, :, 1:minmn)
                    vdS   = Diagonal(diagview(dS)[1:minmn])
                    vdVᴴ  = view(dVᴴ, 1:minmn, :)
                    dA    = MatrixAlgebraKit.svd_pullback!(dA, A, (vU, vS, vVᴴ), (vdU, vdS, vdVᴴ))
                end
                return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
            end
            dU  .= zero(dU)
            dS  .= zero(dS)
            dVᴴ .= zero(dVᴴ)
            return Mooncake.CoDual(USVᴴ, dUSVᴴ), dsvd_adjoint
        end
        @is_primitive Mooncake.DefaultCtx Mooncake.ForwardMode Tuple{typeof($f), AbstractMatrix, Tuple{<:AbstractMatrix, <:$St, <:AbstractMatrix}, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.frule!!(::Dual{<:typeof($f)}, A_dA::Dual, USVᴴ_dUSVᴴ::Dual, alg_dalg::Dual; kwargs...)
            # compute primal
            USVᴴ   = Mooncake.primal(USVᴴ_dUSVᴴ)
            dUSVᴴ  = Mooncake.tangent(USVᴴ_dUSVᴴ)
            A_     = Mooncake.primal(A_dA)
            dA_    = Mooncake.tangent(A_dA)
            A, dA  = arrayify(A_, dA_)
            $f(A, USVᴴ, alg_dalg.primal; kwargs...)

            # update tangents
            U_, S_, Vᴴ_    = USVᴴ
            dU_, dS_, dVᴴ_ = dUSVᴴ
            U, dU          = arrayify(U_, dU_) 
            S, dS          = arrayify(S_, dS_) 
            Vᴴ, dVᴴ        = arrayify(Vᴴ_, dVᴴ_) 
            (dU, dS, dVᴴ)  = svd_pushforward!(dA, A, (U, S, Vᴴ), (dU, dS, dVᴴ); kwargs...)
            return USVᴴ_dUSVᴴ
        end
    end
end

@is_primitive Mooncake.DefaultCtx Mooncake.ForwardMode Tuple{typeof(MatrixAlgebraKit.svd_vals!), AbstractMatrix, AbstractVector, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.frule!!(::Dual{<:typeof(MatrixAlgebraKit.svd_vals!)}, A_dA::Dual, S_dS::Dual, alg_dalg::Dual; kwargs...)
    # compute primal
    S_    = Mooncake.primal(S_dS)
    dS_   = Mooncake.tangent(S_dS)
    A_    = Mooncake.primal(A_dA)
    dA_   = Mooncake.tangent(A_dA)
    A, dA = arrayify(A_, dA_)
    U, nS, Vᴴ = svd_compact(A, Mooncake.primal(alg_dalg); kwargs...)

    # update tangent
    S, dS   = arrayify(S_, dS_) 
    copyto!(dS, diag(real.(Vᴴ * dA' * U)))
    copyto!(S, diagview(nS))
    dA .= zero(eltype(dA))
    return Mooncake.Dual(nS.diag, dS)
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(MatrixAlgebraKit.svd_vals!), AbstractMatrix, AbstractVector, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(::CoDual{<:typeof(MatrixAlgebraKit.svd_vals!)}, A_dA::CoDual, S_dS::CoDual, alg_dalg::CoDual; kwargs...)
    # compute primal
    S_    = Mooncake.primal(S_dS)
    dS_   = Mooncake.tangent(S_dS)
    A_    = Mooncake.primal(A_dA)
    dA_   = Mooncake.tangent(A_dA)
    A, dA = arrayify(A_, dA_)
    S, dS = arrayify(S_, dS_)
    U, nS, Vᴴ = svd_compact(A, Mooncake.primal(alg_dalg); kwargs...)
    S    .= diagview(nS)
    dS   .= zero(eltype(S))
    function dsvd_vals_adjoint(::Mooncake.NoRData)
        dA   .= U * Diagonal(dS) * Vᴴ
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    return S_dS, dsvd_vals_adjoint
end

end
