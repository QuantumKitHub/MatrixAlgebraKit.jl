module MatrixAlgebraKitFillArraysExt

using LinearAlgebra
using MatrixAlgebraKit
using MatrixAlgebraKit: AbstractAlgorithm, check_input, diagview
using FillArrays
using FillArrays: AbstractZerosMatrix

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
            return Zeros(axes(A, 1))
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

function MatrixAlgebraKit.qr_full!(A::Eye, F, alg::EyeAlgorithm)
    ax = axes(A)
    q_ax = (ax[1], ax[1])
    return (Eye(q_ax), A)
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

function MatrixAlgebraKit.lq_full!(A::Eye, F, alg::EyeAlgorithm)
    ax = axes(A)
    q_ax = (ax[2], ax[2])
    return (A, Eye(q_ax))
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

function MatrixAlgebraKit.svd_full!(A::Eye, F, alg::EyeAlgorithm)
    ax = axes(A)
    return (Eye((ax[1], ax[1])), A, Eye((ax[2], ax[2])))
end

end
