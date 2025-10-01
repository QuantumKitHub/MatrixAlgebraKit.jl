module MatrixAlgebraKitChainRulesCoreExt

using MatrixAlgebraKit
using MatrixAlgebraKit: copy_input, initialize_output, zero!, diagview,
    TruncatedAlgorithm, findtruncated, findtruncated_svd
using ChainRulesCore
using LinearAlgebra

# TODO: Decide on an interface to pass on the kwargs for the pullback functions
# from the primal function calls

MatrixAlgebraKit.iszerotangent(::AbstractZero) = true

function ChainRulesCore.rrule(::typeof(copy_input), f, A::AbstractMatrix)
    project = ProjectTo(A)
    copy_input_pullback(ΔA) = (NoTangent(), NoTangent(), project(unthunk(ΔA)))
    return copy_input(f, A), copy_input_pullback
end

for qr_f in (:qr_compact, :qr_full)
    qr_f! = Symbol(qr_f, '!')
    @eval begin
        function ChainRulesCore.rrule(::typeof($qr_f!), A::AbstractMatrix, QR, alg)
            Ac = copy_input($qr_f, A)
            QR = $(qr_f!)(Ac, QR, alg)
            function qr_pullback(ΔQR)
                ΔA = zero(A)
                MatrixAlgebraKit.qr_pullback!(ΔA, A, QR, unthunk.(ΔQR))
                return NoTangent(), ΔA, ZeroTangent(), NoTangent()
            end
            function qr_pullback(::Tuple{ZeroTangent, ZeroTangent}) # is this extra definition useful?
                return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent()
            end
            return QR, qr_pullback
        end
    end
end
function ChainRulesCore.rrule(::typeof(qr_null!), A::AbstractMatrix, N, alg)
    Ac = copy_input(qr_full, A)
    QR = initialize_output(qr_full!, A, alg)
    Q, R = qr_full!(Ac, QR, alg)
    N = copy!(N, view(Q, 1:size(A, 1), (size(A, 2) + 1):size(A, 1)))
    function qr_null_pullback(ΔN)
        ΔA = zero(A)
        (m, n) = size(A)
        minmn = min(m, n)
        ΔQ = zero!(similar(A, (m, m)))
        view(ΔQ, 1:m, (minmn + 1):m) .= unthunk.(ΔN)
        MatrixAlgebraKit.qr_pullback!(ΔA, A, (Q, R), (ΔQ, ZeroTangent()))
        return NoTangent(), ΔA, ZeroTangent(), NoTangent()
    end
    function qr_null_pullback(::ZeroTangent) # is this extra definition useful?
        return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent()
    end
    return N, qr_null_pullback
end

for lq_f in (:lq_compact, :lq_full)
    lq_f! = Symbol(lq_f, '!')
    @eval begin
        function ChainRulesCore.rrule(::typeof($lq_f!), A::AbstractMatrix, LQ, alg)
            Ac = copy_input($lq_f, A)
            LQ = $(lq_f!)(Ac, LQ, alg)
            function lq_pullback(ΔLQ)
                ΔA = zero(A)
                MatrixAlgebraKit.lq_pullback!(ΔA, A, LQ, unthunk.(ΔLQ))
                return NoTangent(), ΔA, ZeroTangent(), NoTangent()
            end
            function lq_pullback(::Tuple{ZeroTangent, ZeroTangent}) # is this extra definition useful?
                return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent()
            end
            return LQ, lq_pullback
        end
    end
end
function ChainRulesCore.rrule(::typeof(lq_null!), A::AbstractMatrix, Nᴴ, alg)
    Ac = copy_input(lq_full, A)
    LQ = initialize_output(lq_full!, A, alg)
    L, Q = lq_full!(Ac, LQ, alg)
    Nᴴ = copy!(Nᴴ, view(Q, (size(A, 1) + 1):size(A, 2), 1:size(A, 2)))
    function lq_null_pullback(ΔNᴴ)
        ΔA = zero(A)
        (m, n) = size(A)
        minmn = min(m, n)
        ΔQ = zero!(similar(A, (n, n)))
        view(ΔQ, (minmn + 1):n, 1:n) .= unthunk.(ΔNᴴ)
        MatrixAlgebraKit.lq_pullback!(ΔA, A, (L, Q), (ZeroTangent(), ΔQ))
        return NoTangent(), ΔA, ZeroTangent(), NoTangent()
    end
    function lq_null_pullback(::ZeroTangent) # is this extra definition useful?
        return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent()
    end
    return Nᴴ, lq_null_pullback
end

for eig in (:eig, :eigh)
    eig_f = Symbol(eig, "_full")
    eig_f! = Symbol(eig_f, "!")
    eig_pb! = Symbol(eig, "_pullback!")
    eig_pb = Symbol(eig, "_pullback")
    eig_t! = Symbol(eig, "_trunc!")
    eig_t_pb = Symbol(eig, "_trunc_pullback")
    _make_eig_t_pb = Symbol("_make_", eig_t_pb)
    @eval begin
        function ChainRulesCore.rrule(::typeof($eig_f!), A::AbstractMatrix, DV, alg)
            Ac = copy_input($eig_f, A)
            DV = $(eig_f!)(Ac, DV, alg)
            function $eig_pb(ΔDV)
                ΔA = zero(A)
                MatrixAlgebraKit.$eig_pb!(ΔA, A, DV, unthunk.(ΔDV))
                return NoTangent(), ΔA, ZeroTangent(), NoTangent()
            end
            function $eig_pb(::Tuple{ZeroTangent, ZeroTangent}) # is this extra definition useful?
                return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent()
            end
            return DV, $eig_pb
        end
        function ChainRulesCore.rrule(
                ::typeof($eig_t!), A::AbstractMatrix, DV,
                alg::TruncatedAlgorithm
            )
            Ac = copy_input($eig_f, A)
            DV = $(eig_f!)(Ac, DV, alg.alg)
            DV′, ind = MatrixAlgebraKit.truncate($eig_t!, DV, alg.trunc)
            return DV′, $(_make_eig_t_pb)(A, DV, ind)
        end
        function $(_make_eig_t_pb)(A, DV, ind)
            function $eig_t_pb(ΔDV)
                ΔA = zero(A)
                MatrixAlgebraKit.$eig_pb!(ΔA, A, DV, unthunk.(ΔDV), ind)
                return NoTangent(), ΔA, ZeroTangent(), NoTangent()
            end
            function $eig_t_pb(::Tuple{ZeroTangent, ZeroTangent}) # is this extra definition useful?
                return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent()
            end
            return $eig_t_pb
        end
    end
end

for svd_f in (:svd_compact, :svd_full)
    svd_f! = Symbol(svd_f, "!")
    @eval begin
        function ChainRulesCore.rrule(::typeof($svd_f!), A::AbstractMatrix, USVᴴ, alg)
            Ac = copy_input($svd_f, A)
            USVᴴ = $(svd_f!)(Ac, USVᴴ, alg)
            function svd_pullback(ΔUSVᴴ)
                ΔA = zero(A)
                MatrixAlgebraKit.svd_pullback!(ΔA, A, USVᴴ, unthunk.(ΔUSVᴴ))
                return NoTangent(), ΔA, ZeroTangent(), NoTangent()
            end
            function svd_pullback(::Tuple{ZeroTangent, ZeroTangent, ZeroTangent}) # is this extra definition useful?
                return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent()
            end
            return USVᴴ, svd_pullback
        end
    end
end

function ChainRulesCore.rrule(
        ::typeof(svd_trunc!), A::AbstractMatrix, USVᴴ,
        alg::TruncatedAlgorithm
    )
    Ac = copy_input(svd_compact, A)
    USVᴴ = svd_compact!(Ac, USVᴴ, alg.alg)
    USVᴴ′, ind = MatrixAlgebraKit.truncate(svd_trunc!, USVᴴ, alg.trunc)
    return USVᴴ′, _make_svd_trunc_pullback(A, USVᴴ, ind)
end
function _make_svd_trunc_pullback(A, USVᴴ, ind)
    function svd_trunc_pullback(ΔUSVᴴ)
        ΔA = zero(A)
        MatrixAlgebraKit.svd_pullback!(ΔA, A, USVᴴ, unthunk.(ΔUSVᴴ), ind)
        return NoTangent(), ΔA, ZeroTangent(), NoTangent()
    end
    function svd_trunc_pullback(::Tuple{ZeroTangent, ZeroTangent, ZeroTangent}) # is this extra definition useful?
        return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent()
    end
    return svd_trunc_pullback
end

function ChainRulesCore.rrule(::typeof(left_polar!), A::AbstractMatrix, WP, alg)
    Ac = copy_input(left_polar, A)
    WP = left_polar!(Ac, WP, alg)
    function left_polar_pullback(ΔWP)
        ΔA = zero(A)
        MatrixAlgebraKit.left_polar_pullback!(ΔA, A, WP, unthunk.(ΔWP))
        return NoTangent(), ΔA, ZeroTangent(), NoTangent()
    end
    function left_polar_pullback(::Tuple{ZeroTangent, ZeroTangent}) # is this extra definition useful?
        return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent()
    end
    return WP, left_polar_pullback
end

function ChainRulesCore.rrule(::typeof(right_polar!), A::AbstractMatrix, PWᴴ, alg)
    Ac = copy_input(left_polar, A)
    PWᴴ = right_polar!(Ac, PWᴴ, alg)
    function right_polar_pullback(ΔPWᴴ)
        ΔA = zero(A)
        MatrixAlgebraKit.right_polar_pullback!(ΔA, A, PWᴴ, unthunk.(ΔPWᴴ))
        return NoTangent(), ΔA, ZeroTangent(), NoTangent()
    end
    function right_polar_pullback(::Tuple{ZeroTangent, ZeroTangent}) # is this extra definition useful?
        return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent()
    end
    return PWᴴ, right_polar_pullback
end

end
