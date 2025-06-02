module MatrixAlgebraKitFillArraysExt

using LinearAlgebra
using MatrixAlgebraKit
using MatrixAlgebraKit: AbstractAlgorithm, TruncatedAlgorithm, TruncationStrategy,
                        check_input, diagview,
                        findtruncated, select_algorithm, select_truncation
using FillArrays
using FillArrays: AbstractZerosMatrix, OnesVector, RectDiagonal, SquareEye

function MatrixAlgebraKit.diagview(A::RectDiagonal{<:Any,<:OnesVector})
    return A.diag
end

struct ZerosAlgorithm <: AbstractAlgorithm end

for f in [:eig, :eigh, :lq, :qr, :svd]
    ff = Symbol("default_", f, "_algorithm")
    @eval begin
        function MatrixAlgebraKit.$ff(::Type{<:AbstractZerosMatrix}; kwargs...)
            return ZerosAlgorithm()
        end
    end
end

for f in [:eig_full,
          :eigh_full,
          :eig_vals,
          :eigh_vals,
          :qr_compact,
          :qr_full,
          :left_polar,
          :lq_compact,
          :lq_full,
          :right_polar,
          :svd_compact,
          :svd_full,
          :svd_vals]
    f! = Symbol(f, "!")
    @eval begin
        MatrixAlgebraKit.copy_input(::typeof($f), A::AbstractZerosMatrix) = A
        function MatrixAlgebraKit.initialize_output(::typeof($f!), A::AbstractZerosMatrix,
                                                    alg::ZerosAlgorithm)
            return nothing
        end
    end
end

for f in [:eig_full!, :eigh_full!]
    @eval begin
        function MatrixAlgebraKit.check_input(::typeof($f), A::AbstractZerosMatrix, F)
            LinearAlgebra.checksquare(A)
            return nothing
        end
        function MatrixAlgebraKit.$f(A::AbstractZerosMatrix, F, alg::ZerosAlgorithm;
                                     kwargs...)
            check_input($f, A, F)
            return (A, Eye(axes(A)))
        end
    end
end

for f in [:eig_vals!, :eigh_vals!]
    @eval begin
        function MatrixAlgebraKit.check_input(::typeof($f), A::AbstractZerosMatrix, F)
            LinearAlgebra.checksquare(A)
            return nothing
        end
        function MatrixAlgebraKit.$f(A::AbstractZerosMatrix, F, alg::ZerosAlgorithm;
                                     kwargs...)
            check_input($f, A, F)
            return diagview(A)
        end
    end
end

function MatrixAlgebraKit.qr_compact!(A::AbstractZerosMatrix, F, alg::ZerosAlgorithm)
    m, n = size(A)
    ax = axes(A)
    if m > n
        r_ax = (ax[2], ax[2])
        return (Eye(ax), Zeros(r_ax))
    else
        q_ax = (ax[1], ax[1])
        return (Eye(q_ax), Zeros(ax))
    end
end

function MatrixAlgebraKit.qr_full!(A::AbstractZerosMatrix, F, alg::ZerosAlgorithm)
    ax = axes(A)
    q_ax = (ax[1], ax[1])
    return (Eye(q_ax), Zeros(ax))
end

function MatrixAlgebraKit.lq_compact!(A::AbstractZerosMatrix, F, alg::ZerosAlgorithm)
    m, n = size(A)
    ax = axes(A)
    if m < n
        l_ax = (ax[1], ax[1])
        return (Zeros(l_ax), Eye(ax))
    else
        q_ax = (ax[2], ax[2])
        return (Zeros(ax), Eye(q_ax))
    end
end

function MatrixAlgebraKit.lq_full!(A::AbstractZerosMatrix, F, alg::ZerosAlgorithm)
    ax = axes(A)
    q_ax = (ax[2], ax[2])
    return (Zeros(ax), Eye(q_ax))
end

function MatrixAlgebraKit.svd_compact!(A::AbstractZerosMatrix, F, alg::ZerosAlgorithm)
    m, n = size(A)
    ax = axes(A)
    if m > n
        s_ax = (ax[2], ax[2])
        return (Eye(ax), Zeros(s_ax), Eye(s_ax))
    else
        s_ax = (ax[1], ax[1])
        return (Eye(s_ax), Zeros(s_ax), Eye(ax))
    end
end

function MatrixAlgebraKit.svd_full!(A::AbstractZerosMatrix, F, alg::ZerosAlgorithm)
    ax = axes(A)
    return (Eye((ax[1], ax[1])), Zeros(ax), Eye((ax[2], ax[2])))
end

function MatrixAlgebraKit.svd_vals!(A::AbstractZerosMatrix, F, alg::ZerosAlgorithm)
    return diagview(A)
end

struct EyeAlgorithm <: AbstractAlgorithm end

for f in [:eig, :eigh, :lq, :qr, :polar, :svd]
    ff = Symbol("default_", f, "_algorithm")
    @eval begin
        function MatrixAlgebraKit.$ff(A::Type{<:Eye}; kwargs...)
            return EyeAlgorithm()
        end
    end
end

for f in [:eig_full,
          :eigh_full,
          :eig_vals,
          :eigh_vals,
          :qr_compact,
          :qr_full,
          :lq_compact,
          :lq_full,
          :left_polar,
          :right_polar,
          :svd_compact,
          :svd_full,
          :svd_vals]
    f! = Symbol(f, "!")
    @eval begin
        MatrixAlgebraKit.copy_input(::typeof($f), A::Eye) = A
        function MatrixAlgebraKit.initialize_output(::typeof($f!), A::Eye,
                                                    alg::EyeAlgorithm)
            return nothing
        end
    end
end

for f in [:eig_full!, :eigh_full!]
    @eval begin
        function MatrixAlgebraKit.check_input(::typeof($f), A::Eye, F)
            LinearAlgebra.checksquare(A)
            return nothing
        end
        function MatrixAlgebraKit.$f(A::Eye, F, alg::EyeAlgorithm;
                                     kwargs...)
            check_input($f, A, F)
            return (A, A)
        end
    end
end

for f in [:eig_trunc!, :eigh_trunc!]
    @eval begin
        # TODO: Delete this when `select_algorithm` is generalized.
        function MatrixAlgebraKit.select_algorithm(::typeof($f), ::Type{A}, alg;
                                                   trunc=nothing,
                                                   kwargs...) where {A<:Eye}
            alg_eig = select_algorithm(eig_full!, A, alg; kwargs...)
            return TruncatedAlgorithm(alg_eig, select_truncation(trunc))
        end
        # TODO: I think it would be better to dispatch on the algorithm here,
        # rather than the output types.
        function MatrixAlgebraKit.truncate!(::typeof($f), (D, V)::Tuple{Eye,Eye},
                                            strategy::TruncationStrategy)
            ind = findtruncated(diagview(D), strategy)
            return Diagonal(diagview(D)[ind]),
                   Eye((axes(V, 1), only(axes(axes(V, 2)[ind]))))
        end
    end
end

for f in [:eig_vals!, :eigh_vals!]
    @eval begin
        function MatrixAlgebraKit.check_input(::typeof($f), A::Eye, F)
            LinearAlgebra.checksquare(A)
            return nothing
        end
        function MatrixAlgebraKit.$f(A::Eye, F, alg::EyeAlgorithm;
                                     kwargs...)
            check_input($f, A, F)
            return diagview(A)
        end
    end
end

function MatrixAlgebraKit.qr_compact!(A::Eye, F, alg::EyeAlgorithm)
    m, n = size(A)
    ax = axes(A)
    if m > n
        r_ax = (ax[2], ax[2])
        return (Eye(ax), Eye(r_ax))
    else
        q_ax = (ax[1], ax[1])
        return (Eye(q_ax), Eye(ax))
    end
end
function MatrixAlgebraKit.qr_compact!(A::SquareEye, F, alg::EyeAlgorithm)
    return (A, A)
end

function MatrixAlgebraKit.qr_full!(A::Eye, F, alg::EyeAlgorithm)
    ax = axes(A)
    q_ax = (ax[1], ax[1])
    return (Eye(q_ax), A)
end
function MatrixAlgebraKit.qr_full!(A::SquareEye, F, alg::EyeAlgorithm)
    return (A, A)
end

function MatrixAlgebraKit.lq_compact!(A::Eye, F, alg::EyeAlgorithm)
    m, n = size(A)
    ax = axes(A)
    if m < n
        l_ax = (ax[1], ax[1])
        return (Eye(l_ax), Eye(ax))
    else
        q_ax = (ax[2], ax[2])
        return (Eye(ax), Eye(q_ax))
    end
end
function MatrixAlgebraKit.lq_compact!(A::SquareEye, F, alg::EyeAlgorithm)
    return (A, A)
end

function MatrixAlgebraKit.lq_full!(A::Eye, F, alg::EyeAlgorithm)
    ax = axes(A)
    q_ax = (ax[2], ax[2])
    return (A, Eye(q_ax))
end
function MatrixAlgebraKit.lq_full!(A::SquareEye, F, alg::EyeAlgorithm)
    return (A, A)
end

function MatrixAlgebraKit.svd_compact!(A::Eye, F, alg::EyeAlgorithm)
    m, n = size(A)
    ax = axes(A)
    if m > n
        s_ax = (ax[2], ax[2])
        return (Eye(ax), Eye(s_ax), Eye(s_ax))
    else
        s_ax = (ax[1], ax[1])
        return (Eye(s_ax), Eye(s_ax), Eye(ax))
    end
end
function MatrixAlgebraKit.svd_compact!(A::SquareEye, F, alg::EyeAlgorithm)
    return (A, A, A)
end

function MatrixAlgebraKit.svd_full!(A::Eye, F, alg::EyeAlgorithm)
    ax = axes(A)
    return (Eye((ax[1],)), A, Eye((ax[2],)))
end
function MatrixAlgebraKit.svd_full!(A::SquareEye, F, alg::EyeAlgorithm)
    return (A, A, A)
end

# TODO: Delete this when `select_algorithm` is generalized.
function MatrixAlgebraKit.select_algorithm(::typeof(svd_trunc!), ::Type{A}, alg;
                                           trunc=nothing,
                                           kwargs...) where {A<:Eye}
    alg_eig = select_algorithm(eig_full!, A, alg; kwargs...)
    return TruncatedAlgorithm(alg_eig, select_truncation(trunc))
end
# TODO: I think it would be better to dispatch on the algorithm here,
# rather than the output types.
function MatrixAlgebraKit.truncate!(::typeof(svd_trunc!), (U, S, V)::Tuple{Eye,Eye,Eye},
                                    strategy::TruncationStrategy)
    ind = findtruncated(diagview(S), strategy)
    U′ = Eye((axes(U, 1), only(axes(axes(U, 2)[ind]))))
    S′ = Diagonal(diagview(S)[ind])
    V′ = Eye((only(axes(axes(V, 1)[ind])), axes(V, 2)))
    return (U′, S′, V′)
end

function MatrixAlgebraKit.svd_vals!(A::Eye, F, alg::EyeAlgorithm)
    return diagview(A)
end

end
