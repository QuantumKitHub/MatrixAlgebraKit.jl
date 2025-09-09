module MatrixAlgebraKitMooncakeExt

using Mooncake
using Mooncake: DefaultCtx, CoDual, Dual, NoRData, rrule!!, frule!!, arrayify, @is_primitive
using MatrixAlgebraKit
using MatrixAlgebraKit: inv_safe, diagview
using MatrixAlgebraKit: qr_pullback!, lq_pullback!
using MatrixAlgebraKit: qr_null_pullback!, lq_null_pullback!
using MatrixAlgebraKit: eig_pullback!, eigh_pullback!
using MatrixAlgebraKit: left_polar_pullback!, right_polar_pullback!
using LinearAlgebra

# two-argument factorizations like LQ, QR, EIG
for (f, pb, adj) in (
        (qr_full!, qr_pullback!, :dqr_adjoint),
        (qr_compact!, qr_pullback!, :dqr_adjoint),
        (lq_full!, lq_pullback!, :dlq_adjoint),
        (lq_compact!, lq_pullback!, :dlq_adjoint),
        (eig_full!, eig_pullback!, :deig_adjoint),
        (eigh_full!, eigh_pullback!, :deigh_adjoint),
        (left_polar!, left_polar_pullback!, :dleft_polar_adjoint),
        (right_polar!, right_polar_pullback!, :dright_polar_adjoint),
    )

    @eval begin
        @is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof($f), AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(::CoDual{typeof($f)}, A_dA::CoDual{<:AbstractMatrix}, args_dargs::CoDual, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm}; kwargs...)
            A, dA = arrayify(A_dA)
            dA .= zero(eltype(A))
            args = Mooncake.primal(args_dargs)
            dargs = Mooncake.tangent(args_dargs)
            arg1, darg1 = arrayify(args[1], dargs[1])
            arg2, darg2 = arrayify(args[2], dargs[2])
            function $adj(::Mooncake.NoRData)
                dA = $pb(dA, A, (arg1, arg2), (darg1, darg2); kwargs...)
                return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
            end
            args = $f(A, args, Mooncake.primal(alg_dalg); kwargs...)
            darg1 .= zero(eltype(arg1))
            darg2 .= zero(eltype(arg2))
            return Mooncake.CoDual(args, dargs), $adj
        end
    end
end

for (f, f_full, pb, adj) in (
        (qr_null!, qr_full, qr_null_pullback!, :dqr_null_adjoint),
        (lq_null!, lq_full, lq_null_pullback!, :dlq_null_adjoint),
    )
    @eval begin
        @is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof($f), AbstractMatrix, AbstractMatrix, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(f_df::CoDual{typeof($f)}, A_dA::CoDual{<:AbstractMatrix}, arg_darg::CoDual{<:AbstractMatrix}, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm}; kwargs...)
            A, dA = arrayify(A_dA)
            Ac = MatrixAlgebraKit.copy_input($f_full, A)
            arg, darg = arrayify(Mooncake.primal(arg_darg), Mooncake.tangent(arg_darg))
            arg = $f(Ac, arg, Mooncake.primal(alg_dalg))
            function $adj(::Mooncake.NoRData)
                dA .= zero(eltype(A))
                $pb(dA, A, arg, darg; kwargs...)
                return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
            end
            return arg_darg, $adj
        end
    end
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(MatrixAlgebraKit.eig_vals!), AbstractMatrix, AbstractVector, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(::CoDual{<:typeof(MatrixAlgebraKit.eig_vals!)}, A_dA::CoDual, D_dD::CoDual, alg_dalg::CoDual; kwargs...)
    # compute primal
    D_ = Mooncake.primal(D_dD)
    dD_ = Mooncake.tangent(D_dD)
    A_ = Mooncake.primal(A_dA)
    dA_ = Mooncake.tangent(A_dA)
    A, dA = arrayify(A_, dA_)
    D, dD = arrayify(D_, dD_)
    dA .= zero(eltype(dA))
    # update primal
    DV = eig_full(A, Mooncake.primal(alg_dalg); kwargs...)
    V = DV[2]
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

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(MatrixAlgebraKit.eigh_vals!), AbstractMatrix, AbstractVector, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(::CoDual{<:typeof(MatrixAlgebraKit.eigh_vals!)}, A_dA::CoDual, D_dD::CoDual, alg_dalg::CoDual; kwargs...)
    # compute primal
    D_ = Mooncake.primal(D_dD)
    dD_ = Mooncake.tangent(D_dD)
    A_ = Mooncake.primal(A_dA)
    dA_ = Mooncake.tangent(A_dA)
    A, dA = arrayify(A_, dA_)
    D, dD = arrayify(D_, dD_)
    DV = eigh_full(A, Mooncake.primal(alg_dalg); kwargs...)
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
            A, dA = arrayify(A_dA)
            USVᴴ = Mooncake.primal(USVᴴ_dUSVᴴ)
            dUSVᴴ = Mooncake.tangent(USVᴴ_dUSVᴴ)
            U, dU = arrayify(USVᴴ[1], dUSVᴴ[1])
            S, dS = arrayify(USVᴴ[2], dUSVᴴ[2])
            Vᴴ, dVᴴ = arrayify(USVᴴ[3], dUSVᴴ[3])
            USVᴴ = $f(A, USVᴴ, Mooncake.primal(alg_dalg); kwargs...)
            function dsvd_adjoint(::Mooncake.NoRData)
                dA .= zero(eltype(A))
                minmn = min(size(A)...)
                if size(U, 2) == size(Vᴴ, 1) == minmn # compact
                    dA = MatrixAlgebraKit.svd_pullback!(dA, A, (U, S, Vᴴ), (dU, dS, dVᴴ))
                else # full
                    vU = view(U, :, 1:minmn)
                    vS = Diagonal(diagview(S)[1:minmn])
                    vVᴴ = view(Vᴴ, 1:minmn, :)
                    vdU = view(dU, :, 1:minmn)
                    vdS = Diagonal(diagview(dS)[1:minmn])
                    vdVᴴ = view(dVᴴ, 1:minmn, :)
                    dA = MatrixAlgebraKit.svd_pullback!(dA, A, (vU, vS, vVᴴ), (vdU, vdS, vdVᴴ))
                end
                return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
            end
            return Mooncake.CoDual(USVᴴ, dUSVᴴ), dsvd_adjoint
        end
    end
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(MatrixAlgebraKit.svd_vals!), AbstractMatrix, AbstractVector, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(::CoDual{<:typeof(MatrixAlgebraKit.svd_vals!)}, A_dA::CoDual, S_dS::CoDual, alg_dalg::CoDual; kwargs...)
    # compute primal
    S_ = Mooncake.primal(S_dS)
    dS_ = Mooncake.tangent(S_dS)
    A_ = Mooncake.primal(A_dA)
    dA_ = Mooncake.tangent(A_dA)
    A, dA = arrayify(A_, dA_)
    S, dS = arrayify(S_, dS_)
    U, nS, Vᴴ = svd_compact(A, Mooncake.primal(alg_dalg); kwargs...)
    S .= diagview(nS)
    dS .= zero(eltype(S))
    function dsvd_vals_adjoint(::Mooncake.NoRData)
        dA .= U * Diagonal(dS) * Vᴴ
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    return S_dS, dsvd_vals_adjoint
end

end
