module MatrixAlgebraKitMooncakeExt

using Mooncake
using Mooncake: @from_chainrules, DefaultCtx, CoDual, Dual, NoRData, rrule!!, frule!!, arrayify, @is_primitive
using MatrixAlgebraKit
using MatrixAlgebraKit: inv_safe, diagview
using MatrixAlgebraKit.YALAPACK
using ChainRulesCore
using LinearAlgebra
using LinearAlgebra: BlasFloat, BlasComplex, diagind

#@from_chainrules DefaultCtx Tuple{typeof(MatrixAlgebraKit.eigh_full!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm}
#@from_chainrules DefaultCtx Tuple{typeof(MatrixAlgebraKit.eigh_vals!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm}

#@from_chainrules DefaultCtx Tuple{typeof(MatrixAlgebraKit.eig_full!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm}
#@from_chainrules DefaultCtx Tuple{typeof(MatrixAlgebraKit.eig_vals!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm}

@from_chainrules DefaultCtx Tuple{typeof(MatrixAlgebraKit.copy_input), Any, AbstractMatrix}

Base.one(::Type{Tangent{Any, @NamedTuple{re::ComplexF64, im::ComplexF64}}}) = one(ComplexF64)

for f in (qr_full!, qr_compact!)
    @eval begin
        @is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof($f), AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(::CoDual{typeof($f)}, A_dA::CoDual{<:AbstractMatrix}, QR_dQR::CoDual, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm}; kwargs...)
            A, dA  = arrayify(A_dA)
            dA    .= zero(eltype(A))
            QR     = Mooncake.primal(QR_dQR)
            dQR    = Mooncake.tangent(QR_dQR)
            Q, dQ  = arrayify(QR[1], dQR[1])
            R, dR  = arrayify(QR[2], dQR[2])
            function dqr_adjoint(::Mooncake.NoRData)
                dA = MatrixAlgebraKit.qr_compact_pullback!(dA, (Q, R), (dQ, dR); kwargs...)
                return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
            end
            QR  = $f(A, QR, Mooncake.primal(alg_dalg); kwargs...)
            dQ .= zero(eltype(Q))
            dR .= zero(eltype(R))
            return Mooncake.CoDual(QR, dQR), dqr_adjoint
        end
    end
end

for f in (lq_full!, lq_compact!)
    @eval begin
        @is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof($f), AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(::CoDual{typeof($f)}, A_dA::CoDual{<:AbstractMatrix}, LQ_dLQ::CoDual, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm}; kwargs...)
            A, dA  = arrayify(A_dA)
            dA    .= zero(eltype(A))
            LQ     = Mooncake.primal(LQ_dLQ)
            dLQ    = Mooncake.tangent(LQ_dLQ)
            L, dL  = arrayify(LQ[1], dLQ[1])
            Q, dQ  = arrayify(LQ[2], dLQ[2])
            function dlq_adjoint(::Mooncake.NoRData)
                dA = MatrixAlgebraKit.lq_compact_pullback!(dA, (L, Q), (dL, dQ); kwargs...)
                return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
            end
            LQ  = $f(A, LQ, Mooncake.primal(alg_dalg); kwargs...)
            dL .= zero(eltype(L))
            dQ .= zero(eltype(Q))
            return Mooncake.CoDual(LQ, dLQ), dlq_adjoint
        end
    end
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(lq_null!), AbstractMatrix, AbstractMatrix, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(f_df::CoDual{typeof(lq_null!)}, A_dA::CoDual{<:AbstractMatrix}, Nᴴ_dNᴴ::CoDual{<:AbstractMatrix}, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm}; kwargs...)
    A, dA  = arrayify(A_dA)
    dA    .= zero(eltype(A))
    Ac     = MatrixAlgebraKit.copy_input(lq_full, A)
    LQ     = MatrixAlgebraKit.initialize_output(lq_full!, A, Mooncake.primal(alg_dalg))
    L, Q   = lq_full!(Ac, LQ, Mooncake.primal(alg_dalg))
    Nᴴ, dNᴴ  = arrayify(Mooncake.primal(Nᴴ_dNᴴ), Mooncake.tangent(Nᴴ_dNᴴ))
    copy!(Nᴴ, view(Q, (size(A, 1) + 1):size(A, 2), 1:size(A, 2)))
    function dlq_null_adjoint(::Mooncake.NoRData)
        m, n = size(A)
        minmn = min(m, n)
        dQ = zeros(eltype(A), (n, n))
        view(dQ, (minmn + 1):n, 1:n) .= dNᴴ
        dL = zeros(eltype(A), (m, n))
        MatrixAlgebraKit.lq_compact_pullback!(dA, (L, Q), (dL, dQ); kwargs...)
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    return Nᴴ_dNᴴ, dlq_null_adjoint
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(qr_null!), AbstractMatrix, AbstractMatrix, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(f_df::CoDual{typeof(qr_null!)}, A_dA::CoDual{<:AbstractMatrix}, N_dN::CoDual{<:AbstractMatrix}, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm}; kwargs...)
    A, dA  = arrayify(A_dA)
    dA    .= zero(eltype(A))
    Ac     = MatrixAlgebraKit.copy_input(qr_full, A)
    QR     = MatrixAlgebraKit.initialize_output(qr_full!, A, Mooncake.primal(alg_dalg))
    Q, R   = qr_full!(Ac, QR, Mooncake.primal(alg_dalg))
    N, dN  = arrayify(Mooncake.primal(N_dN), Mooncake.tangent(N_dN))
    copy!(N, view(Q, 1:size(A, 1), (size(A, 2) + 1):size(A, 1)))
    function dqr_null_adjoint(::Mooncake.NoRData)
        m, n = size(A)
        minmn = min(m, n)
        dQ = zeros(eltype(A), (m, m))
        view(dQ, 1:m, (minmn + 1):m) .= dN
        dR = zeros(eltype(A), (m, n))
        MatrixAlgebraKit.qr_compact_pullback!(dA, (Q, R), (dQ, dR); kwargs...)
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    return N_dN, dqr_null_adjoint
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(MatrixAlgebraKit.eig_full!), AbstractMatrix, Tuple{<:Diagonal, <:AbstractMatrix}, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(::CoDual{typeof(MatrixAlgebraKit.eig_full!)}, A_dA::CoDual, DV_dDV::CoDual, alg_dalg::CoDual; kwargs...)
    A, dA  = arrayify(A_dA)
    dA    .= zero(eltype(A))
    DV     = Mooncake.primal(DV_dDV)
    dDV    = Mooncake.tangent(DV_dDV)
    D, dD  = arrayify(DV[1], dDV[1])
    V, dV  = arrayify(DV[2], dDV[2])
    function deig_adjoint(::Mooncake.NoRData)
        dA = MatrixAlgebraKit.eig_full_pullback!(dA, (D, V), (dD, dV); kwargs...)
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    A_copy = copy(A)
    DV = eig_full!(A, DV, Mooncake.primal(alg_dalg); kwargs...)
    copyto!(A, A_copy)
    return Mooncake.CoDual(DV, dDV), deig_adjoint
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(MatrixAlgebraKit.eigh_full!), AbstractMatrix, Tuple{<:Diagonal, <:AbstractMatrix}, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(::CoDual{typeof(MatrixAlgebraKit.eigh_full!)}, A_dA::CoDual{<:AbstractMatrix}, DV_dDV::CoDual{<:Tuple{<:Diagonal, <:AbstractMatrix}}, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm}; kwargs...)
    A, dA  = arrayify(A_dA)
    dA    .= zero(eltype(A))
    DV     = Mooncake.primal(DV_dDV)
    dDV    = Mooncake.tangent(DV_dDV)
    D, dD  = arrayify(DV[1], dDV[1])
    V, dV  = arrayify(DV[2], dDV[2])
    function deigh_adjoint(::Mooncake.NoRData)
        dA = MatrixAlgebraKit.eigh_full_pullback!(dA, (D, V), (dD, dV); kwargs...)
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    A_copy = copy(A)
    DV = eigh_full!(A, DV, Mooncake.primal(alg_dalg); kwargs...)
    copyto!(A, A_copy)
    return Mooncake.CoDual(DV, dDV), deigh_adjoint
end

for (f, St) in ((svd_full!, :AbstractMatrix), (svd_compact!, :Diagonal))
    @eval begin
        @is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof($f), AbstractMatrix, Tuple{<:AbstractMatrix, <:$St, <:AbstractMatrix}, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(::CoDual{typeof($f)}, A_dA::CoDual, USVᴴ_dUSVᴴ::CoDual, alg_dalg::CoDual; kwargs...)
            A, dA  = arrayify(A_dA)
            dA    .= zero(eltype(A))
            USVᴴ   = Mooncake.primal(USVᴴ_dUSVᴴ)
            dUSVᴴ  = Mooncake.tangent(USVᴴ_dUSVᴴ)
            U, dU  = arrayify(USVᴴ[1], dUSVᴴ[1])
            S, dS  = arrayify(USVᴴ[2], dUSVᴴ[2])
            Vᴴ, dVᴴ = arrayify(USVᴴ[3], dUSVᴴ[3])
            function dsvd_adjoint(::Mooncake.NoRData)
                dA = MatrixAlgebraKit.svd_compact_pullback!(dA, (U, S, Vᴴ), (dU, dS, dVᴴ); kwargs...)
                return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
            end
            USVᴴ = $f(A, USVᴴ, Mooncake.primal(alg_dalg); kwargs...)
            return Mooncake.CoDual(USVᴴ, dUSVᴴ), dsvd_adjoint
        end
    end
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(MatrixAlgebraKit.left_polar!), AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(::CoDual{typeof(MatrixAlgebraKit.left_polar!)}, A_dA::CoDual{<:AbstractMatrix}, WP_dWP::CoDual{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}}, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm}; kwargs...)
    A, dA  = arrayify(A_dA)
    dA    .= zero(eltype(A))
    WP     = Mooncake.primal(WP_dWP)
    dWP    = Mooncake.tangent(WP_dWP)
    W, dW  = arrayify(WP[1], dWP[1])
    P, dP  = arrayify(WP[2], dWP[2])
    function dleft_polar_adjoint(::Mooncake.NoRData)
        dA = MatrixAlgebraKit.left_polar_pullback!(dA, (W, P), (dW, dP); kwargs...)
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    WP = left_polar!(A, WP, Mooncake.primal(alg_dalg); kwargs...)
    return Mooncake.CoDual(WP, dWP), dleft_polar_adjoint
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(MatrixAlgebraKit.right_polar!), AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(::CoDual{typeof(MatrixAlgebraKit.right_polar!)}, A_dA::CoDual{<:AbstractMatrix}, PWᴴ_dPWᴴ::CoDual{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}}, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm}; kwargs...)
    A, dA   = arrayify(A_dA)
    dA     .= zero(eltype(A))
    PWᴴ     = Mooncake.primal(PWᴴ_dPWᴴ)
    dPWᴴ    = Mooncake.tangent(PWᴴ_dPWᴴ)
    P, dP   = arrayify(PWᴴ[1], dPWᴴ[1])
    Wᴴ, dWᴴ = arrayify(PWᴴ[2], dPWᴴ[2])
    function dright_polar_adjoint(::Mooncake.NoRData)
        dA = MatrixAlgebraKit.right_polar_pullback!(dA, (P, Wᴴ), (dP, dWᴴ); kwargs...)
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    PWᴴ = right_polar!(A, PWᴴ, Mooncake.primal(alg_dalg); kwargs...)
    return Mooncake.CoDual(PWᴴ, dPWᴴ), dright_polar_adjoint
end

@from_chainrules DefaultCtx Tuple{typeof(MatrixAlgebraKit.svd_vals!), AbstractMatrix, Any, MatrixAlgebraKit.AbstractAlgorithm}
@from_chainrules DefaultCtx Tuple{typeof(MatrixAlgebraKit.svd_trunc!), AbstractMatrix, Any, MatrixAlgebraKit.TruncatedAlgorithm}
end
