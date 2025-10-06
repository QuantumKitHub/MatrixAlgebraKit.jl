module MatrixAlgebraKitMooncakeExt

using Mooncake
using Mooncake: DefaultCtx, CoDual, Dual, NoRData, rrule!!, frule!!, arrayify, @is_primitive
using MatrixAlgebraKit
using MatrixAlgebraKit: inv_safe, diagview
using LinearAlgebra

function lq_compact_fwd(dA, LQ, dLQ; tol::Real=MatrixAlgebraKit.default_pullback_gaugetol(LQ[1]), rank_atol::Real=tol, gauge_atol::Real=tol)
    L, Q  = LQ
    m     = size(L, 1)
    n     = size(Q, 2)
    minmn = min(m, n)
    Ld    = diagview(L)
    p     = findlast(>=(rank_atol) ∘ abs, Ld)

    n1 = p
    n2 = minmn - p
    n3 = n - minmn
    m1 = p
    m2 = m - p

    #####
    Q1  = view(Q, 1:n1, 1:n) # full rank portion
    Q2  = view(Q, 1:n1+1:n2+n1, 1:n)
    L11 = view(L, 1:m, 1:n1)
    L12 = view(L, 1:m1, n1+1:n)

    dA1 = view(dA, 1:m, 1:n1)
    dA2 = view(dA, 1:m, (n1 + 1):n)

    dQ, dR = dQR
    dQ1    = view(dQ, 1:m, 1:m1)
    dQ2    = view(dQ, 1:m, m1+1:m2+m1)
    dR11   = view(dR, 1:m1, 1:n1)
    dR12   = view(dR, 1:m1, n1+1:n)
    dR22   = view(dR, m1+1:m1+m2, n1+1:n)

    # fwd rule for Q1 and R11 -- for a non-rank redeficient QR, this is all we need
    invR11  = inv(R11)
    tmp     = Q1' * dA1 * invR11
    Rtmp    = tmp + tmp'
    diagview(Rtmp) ./= 2
    ltRtmp  = view(Rtmp, MatrixAlgebraKit.lowertriangularind(Rtmp))
    #ltRtmp .= zero(eltype(Rtmp))
    dR11   .= Rtmp * R11
    dQ1    .= dA1 * invR11 - Q1 * dR11 * invR11

    dR12  .= adjoint(Q1) * (dA2 - dQ1 * R12)
    dQ2   .= Q1 * (Q1' * dQ2)
    if size(Q2, 2) > 0
        dQ2  .+= Q2 * (Q2' * dQ2)
    end
    if m3 > 0 && size(dQ2, 2) > 0
        # only present for qr_full or rank-deficient qr_compact
        Q3    = view(Q, 1:m, m1+m2+1:size(Q, 2))
        dQ2 .+= Q3 * (Q3' * dQ2)
    end
    if !isempty(dR22)
        _, r22 = qr_full(dA2 - dQ1*R12 - Q1*dR12, MatrixAlgebraKit.LAPACK_HouseholderQR(; positive=true))
        dR22  .= view(r22, 1:size(dR22, 1), 1:size(dR22, 2))
    end
    return (dQ, dR)
end


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
                dA = MatrixAlgebraKit.qr_pullback!(dA, A, (Q, R), (dQ, dR); kwargs...)
                return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
            end
            QR  = $f(A, QR, Mooncake.primal(alg_dalg); kwargs...)
            dQ .= zero(eltype(Q))
            dR .= zero(eltype(R))
            return Mooncake.CoDual(QR, dQR), dqr_adjoint
        end
        @is_primitive Mooncake.DefaultCtx Mooncake.ForwardMode Tuple{typeof($f), AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.frule!!(::Dual{typeof($f)}, A_dA::Dual{<:AbstractMatrix}, QR_dQR::Dual, alg_dalg::Dual{<:MatrixAlgebraKit.AbstractAlgorithm}; kwargs...)
            A, dA = arrayify(A_dA)
            QR    = Mooncake.primal(QR_dQR)
            QR    = $f(A, QR, Mooncake.primal(alg_dalg); kwargs...)
            dQR   = Mooncake.tangent(QR_dQR)
            Q, dQ = arrayify(QR[1], dQR[1])
            R, dR = arrayify(QR[2], dQR[2])
            dQ, dR = MatrixAlgebraKit.qr_compact_fwd(dA, (Q, R), (dQ, dR))
            dA   .= zero(eltype(A))
            return Mooncake.Dual(QR, dQR)
        end
    end
end

for f in (lq_full!, lq_compact!)
    @eval begin
        @is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof($f), AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.rrule!!(::CoDual{typeof($f)}, A_dA::CoDual{<:AbstractMatrix}, LQ_dLQ::CoDual, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm}; kwargs...)
            A, dA  = arrayify(A_dA)
            LQ     = Mooncake.primal(LQ_dLQ)
            dLQ    = Mooncake.tangent(LQ_dLQ)
            L, dL  = arrayify(LQ[1], dLQ[1])
            Q, dQ  = arrayify(LQ[2], dLQ[2])
            function dlq_adjoint(::Mooncake.NoRData)
                dA .= zero(eltype(A))
                dA  = MatrixAlgebraKit.lq_pullback!(dA, A, (L, Q), (dL, dQ); kwargs...)
                return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
            end
            LQ  = $f(copy(A), LQ, Mooncake.primal(alg_dalg); kwargs...)
            dL .= zero(eltype(L))
            dQ .= zero(eltype(Q))
            return Mooncake.CoDual(LQ, dLQ), dlq_adjoint
        end
        @is_primitive Mooncake.DefaultCtx Mooncake.ForwardMode Tuple{typeof($f), AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}, MatrixAlgebraKit.AbstractAlgorithm}
        function Mooncake.frule!!(::Dual{typeof($f)}, A_dA::Dual{<:AbstractMatrix}, LQ_dLQ::Dual, alg_dalg::Dual{<:MatrixAlgebraKit.AbstractAlgorithm}; kwargs...)
            A, dA = arrayify(A_dA)
            LQ    = Mooncake.primal(LQ_dLQ)
            LQ    = $f(A, LQ, Mooncake.primal(alg_dalg); kwargs...)
            dLQ   = Mooncake.tangent(LQ_dLQ)
            L, dL = arrayify(LQ[1], dLQ[1])
            Q, dQ = arrayify(LQ[2], dLQ[2])
            invL  = inv(L)
            ∂K    = invL * dA * Q'
            ∂K    = ∂K + ∂K'
            diagview(∂K) ./= 2
            ∂K[MatrixAlgebraKit.uppertriangularind(∂K)] .= zero(eltype(∂K))
            dL   .= L * ∂K
            dQ   .= invL * dA - invL * dL * Q
            dA   .= zero(eltype(A))
            return Mooncake.Dual(LQ, dLQ)
        end
    end
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(lq_null!), AbstractMatrix, AbstractMatrix, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(f_df::CoDual{typeof(lq_null!)}, A_dA::CoDual{<:AbstractMatrix}, Nᴴ_dNᴴ::CoDual{<:AbstractMatrix}, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm}; kwargs...)
    A, dA   = arrayify(A_dA)
    Ac      = MatrixAlgebraKit.copy_input(lq_full, A)
    Nᴴ, dNᴴ = arrayify(Mooncake.primal(Nᴴ_dNᴴ), Mooncake.tangent(Nᴴ_dNᴴ))
    Nᴴ      = lq_null!(Ac, Nᴴ, Mooncake.primal(alg_dalg))
    function dlq_null_adjoint(::Mooncake.NoRData)
        dA    .= zero(eltype(A))
        MatrixAlgebraKit.lq_null_pullback!(dA, A, Nᴴ, dNᴴ; kwargs...)
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    return Nᴴ_dNᴴ, dlq_null_adjoint
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(qr_null!), AbstractMatrix, AbstractMatrix, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(f_df::CoDual{typeof(qr_null!)}, A_dA::CoDual{<:AbstractMatrix}, N_dN::CoDual{<:AbstractMatrix}, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm}; kwargs...)
    A, dA  = arrayify(A_dA)
    N, dN  = arrayify(Mooncake.primal(N_dN), Mooncake.tangent(N_dN))
    Ac     = MatrixAlgebraKit.copy_input(qr_full, A)
    N      = qr_null!(Ac, N, Mooncake.primal(alg_dalg))
    function dqr_null_adjoint(::Mooncake.NoRData)
        dA .= zero(eltype(A))
        MatrixAlgebraKit.qr_null_pullback!(dA, A, N, dN; kwargs...)
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    return N_dN, dqr_null_adjoint
end

@is_primitive Mooncake.DefaultCtx Mooncake.ForwardMode Tuple{typeof(qr_null!), AbstractMatrix, AbstractMatrix, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.frule!!(f_df::Dual{typeof(qr_null!)}, A_dA::Dual{<:AbstractMatrix}, N_dN::Dual{<:AbstractMatrix}, alg_dalg::Dual{<:MatrixAlgebraKit.AbstractAlgorithm}; kwargs...)
    N, dN  = arrayify(Mooncake.primal(N_dN), Mooncake.tangent(N_dN))
    A, dA  = arrayify(A_dA)
    N = qr_null!(A, N, Mooncake.primal(alg_dalg))
    m, n = size(A)
    Ac = MatrixAlgebraKit.copy_input(qr_full, A)
    QR = MatrixAlgebraKit.initialize_output(qr_full!, Ac, Mooncake.primal(alg_dalg))
    Q, R = qr_full!(Ac, QR, Mooncake.primal(alg_dalg))
    copy!(N, view(Q, 1:size(A, 1), (size(A, 2) + 1):size(A, 1)))
    minmn  = min(m, n)
    dQ     = zeros(eltype(A), (m, m))
    view(dQ, 1:m, (minmn + 1):m) .= dN
    MatrixAlgebraKit.qr_compact_fwd(dA, (Q, R), (dQ, zeros(eltype(R), size(R))))
    dN .= view(dQ, 1:m, (minmn + 1):m)
    dA .= zero(eltype(A))
    return N_dN
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
        dA = MatrixAlgebraKit.eig_pullback!(dA, A, (D, V), (dD, dV); kwargs...)
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    DV = eig_full!(A, DV, Mooncake.primal(alg_dalg); kwargs...)
    return Mooncake.CoDual(DV, dDV), deig_adjoint
end

@is_primitive Mooncake.DefaultCtx Mooncake.ForwardMode Tuple{typeof(MatrixAlgebraKit.eig_full!), AbstractMatrix, Tuple{<:Diagonal, <:AbstractMatrix}, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.frule!!(::Dual{typeof(MatrixAlgebraKit.eig_full!)}, A_dA::Dual, DV_dDV::Dual, alg_dalg::Dual; kwargs...)
    A, dA  = arrayify(A_dA)
    DV     = Mooncake.primal(DV_dDV)
    dDV    = Mooncake.tangent(DV_dDV)
    D, dD  = arrayify(DV[1], dDV[1])
    V, dV  = arrayify(DV[2], dDV[2])
    (D, V) = eig_full!(A, DV, Mooncake.primal(alg_dalg); kwargs...)
    MatrixAlgebraKit.eig_full_fwd(dA, DV, (dD, dV))
    return Mooncake.Dual(DV, dDV)
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(MatrixAlgebraKit.eig_trunc!), AbstractMatrix, Tuple{<:Diagonal, <:AbstractMatrix}, MatrixAlgebraKit.TruncatedAlgorithm}
function Mooncake.rrule!!(::CoDual{typeof(MatrixAlgebraKit.eig_trunc!)}, A_dA::CoDual, DV_dDV::CoDual, alg_dalg::CoDual; kwargs...)
    A, dA  = arrayify(A_dA)
    DV     = Mooncake.primal(DV_dDV)
    dDV    = Mooncake.tangent(DV_dDV)
    D, dD  = arrayify(DV[1], dDV[1])
    V, dV  = arrayify(DV[2], dDV[2])
    DV′, ind = eig_trunc(A, DV, Mooncake.primal(alg_dalg); kwargs...)
    function deig_trunc_adjoint(::Mooncake.NoRData)
        dA .= zero(eltype(A))
        dA  = MatrixAlgebraKit.eig_pullback!(dA, A, (D, V), (dD, dV), ind; kwargs...)
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    return Mooncake.CoDual(DV′, dDV), deig_trunc_adjoint
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
        # Add lower triangle to upper triangle and zero out.
        dA .*= 2
        dA[diagind(dA)] ./= 2
        for i in 1:size(dA, 1), j in 1:size(dA, 2)
            if i > j
                dA[i, j] = zero(eltype(dA)) 
            end
        end
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
    dA .*= 2
    dA[diagind(dA)] ./= 2
    for i in 1:size(dA, 1), j in 1:size(dA, 2)
        if i > j
            dA[i, j] = zero(eltype(dA))
        end
    end
    tmpV         = V \ dA
    ∂K           = tmpV * V
    ∂Kdiag       = diag(∂K)
    dD.diag     .= real.(∂Kdiag)
    dDD          = transpose(diagview(D)) .- diagview(D)
    F            = one(eltype(dDD)) ./ dDD
    diagview(F) .= zero(eltype(F))
    ∂K         .*= F
    ∂V           = mul!(tmpV, V, ∂K) 
    copyto!(dV, ∂V)
    dA          .= zero(eltype(A))
    return Mooncake.Dual(DV, dDV)
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(MatrixAlgebraKit.eigh_trunc!), AbstractMatrix, Tuple{<:Diagonal, <:AbstractMatrix}, MatrixAlgebraKit.TruncatedAlgorithm}
function Mooncake.rrule!!(::CoDual{typeof(MatrixAlgebraKit.eigh_trunc!)}, A_dA::CoDual, DV_dDV::CoDual, alg_dalg::CoDual; kwargs...)
    A, dA  = arrayify(A_dA)
    DV     = Mooncake.primal(DV_dDV)
    dDV    = Mooncake.tangent(DV_dDV)
    D, dD  = arrayify(DV[1], dDV[1])
    V, dV  = arrayify(DV[2], dDV[2])
    DV′, ind = eig_trunc(A, DV, Mooncake.primal(alg_dalg); kwargs...)
    function deigh_trunc_adjoint(::Mooncake.NoRData)
        dA .= zero(eltype(A))
        dA  = MatrixAlgebraKit.eig_pullback!(dA, A, (D, V), (dD, dV), ind; kwargs...)
        dA .*= 2
        dA[diagind(dA)] ./= 2
        for i in 1:size(dA, 1), j in 1:size(dA, 2)
            if i > j
                dA[i, j] = zero(eltype(dA)) 
            end
        end
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    return Mooncake.CoDual(DV′, dDV), deig_trunc_adjoint
end

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

    dA .*= 2
    dA[diagind(dA)] ./= 2
    for i in 1:size(dA, 1), j in 1:size(dA, 2)
        if i > j
            dA[i, j] = zero(eltype(dA))
        end
    end
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
        # Add lower triangle to upper triangle and zero out.
        dA .*= 2
        dA[diagind(dA)] ./= 2
        for i in 1:size(dA, 1), j in 1:size(dA, 2)
            if i > j
                dA[i, j] = zero(eltype(dA))
            end
        end
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    return Mooncake.CoDual(DV[1].diag, dD_), deigh_vals_adjoint
end

@is_primitive Mooncake.DefaultCtx Mooncake.ForwardMode Tuple{typeof(svd_compact!), AbstractMatrix, Tuple{<:AbstractMatrix, <:Diagonal, <:AbstractMatrix}, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.frule!!(::Dual{<:typeof(svd_compact!)}, A_dA::Dual, USVᴴ_dUSVᴴ::Dual, alg_dalg::Dual; kwargs...)
    # compute primal
    USVᴴ   = Mooncake.primal(USVᴴ_dUSVᴴ)
    dUSVᴴ  = Mooncake.tangent(USVᴴ_dUSVᴴ)
    A_     = Mooncake.primal(A_dA)
    dA_    = Mooncake.tangent(A_dA)
    A, dA  = arrayify(A_, dA_)
    svd_compact!(A, USVᴴ, alg_dalg.primal; kwargs...)
    
    # update tangents
    U_, S_, Vᴴ_    = USVᴴ
    dU_, dS_, dVᴴ_ = dUSVᴴ
    U, dU   = arrayify(U_, dU_) 
    S, dS   = arrayify(S_, dS_) 
    Vᴴ, dVᴴ = arrayify(Vᴴ_, dVᴴ_) 
    V       = adjoint(Vᴴ)

    copyto!(dS.diag, diag(real.(U' * dA * V)))
    m, n    = size(A)
    F       = one(eltype(S)) ./ (diagview(S)' .- diagview(S))
    G       = one(eltype(S)) ./ (diagview(S)' .+ diagview(S))
    diagview(F) .= zero(eltype(F))
    invSdiag = zeros(eltype(S), length(S.diag))
    for i in 1:length(S.diag)
        @inbounds invSdiag[i] = inv(diagview(S)[i])
    end
    invS = Diagonal(invSdiag)
    ∂U = U * (F .* (U' * dA * V * S + S * Vᴴ * dA' * U)) + (diagm(ones(eltype(U), m)) - U*U') * dA * V * invS
    ∂V = V * (F .* (S * U' * dA * V + Vᴴ * dA' * U * S)) + (diagm(ones(eltype(V), n)) - V*Vᴴ) * dA' * U * invS
    copyto!(dU, ∂U)
    adjoint!(dVᴴ, ∂V)
    dA .= zero(eltype(A))
    return USVᴴ_dUSVᴴ
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
                dA .= zero(eltype(A))
                minmn = min(size(A)...)
                dA  = MatrixAlgebraKit.svd_pullback!(dA, A, (U, S, Vᴴ), (dU, dS, dVᴴ))
                return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
            end
            return Mooncake.CoDual(USVᴴ, dUSVᴴ), dsvd_adjoint
        end
    end
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(svd_trunc!), AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix, <:AbstractMatrix}, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(::CoDual{typeof(svd_trunc!)}, A_dA::CoDual, USVᴴ_dUSVᴴ::CoDual, alg_dalg::CoDual; kwargs...)
    A, dA   = arrayify(A_dA)
    USVᴴ    = Mooncake.primal(USVᴴ_dUSVᴴ)
    dUSVᴴ   = Mooncake.tangent(USVᴴ_dUSVᴴ)
    alg     = Mooncake.primal(alg_dalg)
    U, dU   = arrayify(USVᴴ[1], dUSVᴴ[1])
    S, dS   = arrayify(USVᴴ[2], dUSVᴴ[2])
    Vᴴ, dVᴴ = arrayify(USVᴴ[3], dUSVᴴ[3])
    Ac      = MatrixAlgebraKit.copy_input(svd_compact, A)
    USVᴴ    = svd_compact!(Ac, USVᴴ, alg.alg)
    function dsvd_trunc_adjoint(::Mooncake.NoRData)
        dA .= zero(eltype(A))
        dA  = MatrixAlgebraKit.svd_pullback!(dA, A, (U, S, Vᴴ), (dU, dS, dVᴴ), ind)
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    USVᴴ′, ind = MatrixAlgebraKit.truncate(svd_trunc!, USVᴴ, alg.trunc)
    ΔU′  = Matrix(dU[:, ind])
    ΔS′  = Diagonal(dS[ind, ind])
    ΔVᴴ′ = Matrix(dVᴴ[ind, :])
    dS′  = Mooncake.build_tangent(typeof(ΔS′), ΔS′.diag)
    if eltype(A) <: Real
        dU′   = ΔU′
        dVᴴ′  = ΔVᴴ′
    else
        dU′   = [Mooncake.build_tangent(typeof(ΔU′[i,j]), real(ΔU′[i,j]), imag(ΔU′[i,j])) for i in 1:size(ΔU′, 1), j in 1:size(ΔU′, 2)]
        dVᴴ′  = [Mooncake.build_tangent(typeof(ΔVᴴ′[i,j]), real(ΔVᴴ′[i,j]), imag(ΔVᴴ′[i,j])) for i in 1:size(ΔVᴴ′, 1), j in 1:size(ΔVᴴ′, 2)]
    end
    dUSVᴴ′ = Mooncake.build_tangent(typeof((ΔU′,ΔS′,ΔVᴴ′)), dU′, dS′, dVᴴ′)
    return Mooncake.CoDual(USVᴴ′, dUSVᴴ′), dsvd_trunc_adjoint
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

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(MatrixAlgebraKit.left_polar!), AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}, MatrixAlgebraKit.AbstractAlgorithm}
function Mooncake.rrule!!(::CoDual{typeof(MatrixAlgebraKit.left_polar!)}, A_dA::CoDual{<:AbstractMatrix}, WP_dWP::CoDual{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}}, alg_dalg::CoDual{<:MatrixAlgebraKit.AbstractAlgorithm}; kwargs...)
    A, dA  = arrayify(A_dA)
    dA    .= zero(eltype(A))
    WP     = Mooncake.primal(WP_dWP)
    dWP    = Mooncake.tangent(WP_dWP)
    W, dW  = arrayify(WP[1], dWP[1])
    P, dP  = arrayify(WP[2], dWP[2])
    function dleft_polar_adjoint(::Mooncake.NoRData)
        dA = MatrixAlgebraKit.left_polar_pullback!(dA, A, (W, P), (dW, dP); kwargs...)
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
        dA = MatrixAlgebraKit.right_polar_pullback!(dA, A, (P, Wᴴ), (dP, dWᴴ); kwargs...)
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData()
    end
    PWᴴ = right_polar!(A, PWᴴ, Mooncake.primal(alg_dalg); kwargs...)
    return Mooncake.CoDual(PWᴴ, dPWᴴ), dright_polar_adjoint
end

end
