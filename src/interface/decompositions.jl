# TODO: module Decompositions?

# =================
# LAPACK ALGORITHMS
# =================

# reference for naming LAPACK algorithms:
# https://www.netlib.org/lapack/explore-html/topics.html

# QR, LQ, QL, RQ Decomposition
# ----------------------------
"""
    LAPACK_HouseholderQR(; blocksize, positive = false, pivoted = false)

Algorithm type to denote the standard LAPACK algorithm for computing the QR decomposition of
a matrix using Householder reflectors. The specific LAPACK function can be controlled using
the keyword arugments, i.e.  `?geqrt` will be chosen if `blocksize > 1`. With
`blocksize == 1`, `?geqrf` will be chosen if `pivoted == false` and `?geqp3` will be chosen
if `pivoted == true`. The keyword `positive=true` can be used to ensure that the diagonal
elements of `R` are non-negative.
"""
@algdef LAPACK_HouseholderQR

"""
    LAPACK_HouseholderLQ(; blocksize, positive = false)

Algorithm type to denote the standard LAPACK algorithm for computing the LQ decomposition of
a matrix using Householder reflectors. The specific LAPACK function can be controlled using
the keyword arugments, i.e. `?gelqt` will be chosen if `blocksize > 1` or `?gelqf` will be
chosen if `blocksize == 1`. The keyword `positive=true` can be used to ensure that the diagonal
elements of `L` are non-negative.
"""
@algdef LAPACK_HouseholderLQ

# TODO:
@algdef LAPACK_HouseholderQL
@algdef LAPACK_HouseholderRQ

# General Eigenvalue Decomposition
# -------------------------------
"""
    LAPACK_Simple()

Algorithm type to denote the simple LAPACK driver for computing the Schur or non-Hermitian
eigenvalue decomposition of a matrix.
"""
@algdef LAPACK_Simple

"""
    LAPACK_Expert()

Algorithm type to denote the expert LAPACK driver for computing the Schur or non-Hermitian
eigenvalue decomposition of a matrix.
"""
@algdef LAPACK_Expert

const LAPACK_EigAlgorithm = Union{LAPACK_Simple, LAPACK_Expert}

# Hermitian Eigenvalue Decomposition
# ----------------------------------
"""
    LAPACK_QRIteration()

Algorithm type to denote the LAPACK driver for computing the eigenvalue decomposition of a
Hermitian matrix, or the singular value decomposition of a general matrix using the
QR Iteration algorithm.
"""
@algdef LAPACK_QRIteration

"""
    LAPACK_Bisection()

Algorithm type to denote the LAPACK driver for computing the eigenvalue decomposition of a
Hermitian matrix, or the singular value decomposition of a general matrix using the
Bisection algorithm.
"""
@algdef LAPACK_Bisection

"""
    LAPACK_DivideAndConquer()

Algorithm type to denote the LAPACK driver for computing the eigenvalue decomposition of a
Hermitian matrix, or the singular value decomposition of a general matrix using the
Divide and Conquer algorithm.
"""
@algdef LAPACK_DivideAndConquer

"""
    LAPACK_MultipleRelativelyRobustRepresentations()

Algorithm type to denote the LAPACK driver for computing the eigenvalue decomposition of a
Hermitian matrix using the Multiple Relatively Robust Representations algorithm.
"""
@algdef LAPACK_MultipleRelativelyRobustRepresentations

const LAPACK_EighAlgorithm = Union{
    LAPACK_QRIteration,
    LAPACK_Bisection,
    LAPACK_DivideAndConquer,
    LAPACK_MultipleRelativelyRobustRepresentations,
}

# Singular Value Decomposition
# ----------------------------
"""
    LAPACK_Jacobi()

Algorithm type to denote the LAPACK driver for computing the singular value decomposition of
a general matrix using the Jacobi algorithm.
"""
@algdef LAPACK_Jacobi

const LAPACK_SVDAlgorithm = Union{
    LAPACK_QRIteration,
    LAPACK_Bisection,
    LAPACK_DivideAndConquer,
    LAPACK_Jacobi,
}

# =========================
# Polar decompositions
# =========================
"""
    PolarViaSVD(svd_alg)

Algorithm for computing the polar decomposition of a matrix `A` via the singular value
decomposition (SVD) of `A`. The `svd_alg` argument specifies the SVD algorithm to use.
"""
struct PolarViaSVD{SVDAlg} <: AbstractAlgorithm
    svd_alg::SVDAlg
end

"""
    PolarNewton(; maxiter = 10, tol = defaulttol(A))

Algorithm for computing the polar decomposition of a matrix `A` via
scaled Newton iteration, with a maximum of `maxiter` iterations and
until convergence up to tolerance `tol`.
"""
@algdef PolarNewton

# =========================
# Varia
# =========================
"""
    DiagonalAlgorithm(; kwargs...)

Algorithm type to denote a native Julia implementation of the decompositions making use of
the diagonal structure of the input and outputs.
"""
@algdef DiagonalAlgorithm

"""
    LQViaTransposedQR(qr_alg)

Algorithm type to denote finding the LQ decomposition of `A` by computing the QR decomposition of `Aᵀ`.
The `qr_alg` specifies which QR-decomposition implementation to use.
"""
struct LQViaTransposedQR{A <: AbstractAlgorithm} <: AbstractAlgorithm
    qr_alg::A
end
function Base.show(io::IO, alg::LQViaTransposedQR)
    print(io, "LQViaTransposedQR(")
    _show_alg(io, alg.qr_alg)
    return print(io, ")")
end

# =========================
# CUSOLVER ALGORITHMS
# =========================
"""
    CUSOLVER_HouseholderQR(; positive = false)

Algorithm type to denote the standard CUSOLVER algorithm for computing the QR decomposition of
a matrix using Householder reflectors. The keyword `positive=true` can be used to ensure that
the diagonal elements of `R` are non-negative.
"""
@algdef CUSOLVER_HouseholderQR

"""
    CUSOLVER_QRIteration()

Algorithm type to denote the CUSOLVER driver for computing the eigenvalue decomposition of a
Hermitian matrix, or the singular value decomposition of a general matrix using the
QR Iteration algorithm.
"""
@algdef CUSOLVER_QRIteration

"""
    CUSOLVER_SVDPolar()

Algorithm type to denote the CUSOLVER driver for computing the singular value decomposition of
a general matrix by using Halley's iterative algorithm to compute the polar decompositon,
followed by the hermitian eigenvalue decomposition of the positive definite factor.
"""
@algdef CUSOLVER_SVDPolar

"""
    CUSOLVER_Jacobi()

Algorithm type to denote the CUSOLVER driver for computing the singular value decomposition of
a general matrix using the Jacobi algorithm.
"""
@algdef CUSOLVER_Jacobi

"""
    CUSOLVER_Randomized(; k, p, niters)

Algorithm type to denote the CUSOLVER driver for computing the singular value decomposition of
a general matrix using the randomized SVD algorithm. Here, `k` denotes the number of singular
values that should be computed, therefore requiring `k <= min(size(A))`. This method is accurate
for small values of `k` compared to the size of the input matrix, where the accuracy can be
improved by increasing `p`, the number of additional values used for oversampling,
and `niters`, the number of iterations the solver uses, at the cost of increasing the runtime.

See also the [CUSOLVER documentation](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnxgesvdr)
for more information.
"""
@algdef CUSOLVER_Randomized

does_truncate(::TruncatedAlgorithm{<:CUSOLVER_Randomized}) = true

"""
    CUSOLVER_Simple()

Algorithm type to denote the simple CUSOLVER driver for computing the non-Hermitian
eigenvalue decomposition of a matrix.
"""
@algdef CUSOLVER_Simple

const CUSOLVER_EigAlgorithm = Union{CUSOLVER_Simple}

"""
    CUSOLVER_DivideAndConquer()

Algorithm type to denote the CUSOLVER driver for computing the eigenvalue decomposition of a
Hermitian matrix, or the singular value decomposition of a general matrix using the
Divide and Conquer algorithm.
"""
@algdef CUSOLVER_DivideAndConquer

const CUSOLVER_SVDAlgorithm = Union{
    CUSOLVER_QRIteration, CUSOLVER_SVDPolar, CUSOLVER_Jacobi, CUSOLVER_Randomized,
}

# =========================
# ROCSOLVER ALGORITHMS
# =========================
"""
    ROCSOLVER_HouseholderQR(; positive = false)

Algorithm type to denote the standard ROCSOLVER algorithm for computing the QR decomposition of
a matrix using Householder reflectors. The keyword `positive=true` can be used to ensure that
the diagonal elements of `R` are non-negative.
"""
@algdef ROCSOLVER_HouseholderQR

"""
    ROCSOLVER_QRIteration()

Algorithm type to denote the ROCSOLVER driver for computing the eigenvalue decomposition of a
Hermitian matrix, or the singular value decomposition of a general matrix using the
QR Iteration algorithm.
"""
@algdef ROCSOLVER_QRIteration

"""
    ROCSOLVER_Jacobi()

Algorithm type to denote the ROCSOLVER driver for computing the singular value decomposition of
a general matrix using the Jacobi algorithm.
"""
@algdef ROCSOLVER_Jacobi

"""
    ROCSOLVER_Bisection()

Algorithm type to denote the ROCSOLVER driver for computing the eigenvalue decomposition of a
Hermitian matrix, or the singular value decomposition of a general matrix using the
Bisection algorithm.
"""
@algdef ROCSOLVER_Bisection

"""
    ROCSOLVER_DivideAndConquer()

Algorithm type to denote the ROCSOLVER driver for computing the eigenvalue decomposition of a
Hermitian matrix, or the singular value decomposition of a general matrix using the
Divide and Conquer algorithm.
"""
@algdef ROCSOLVER_DivideAndConquer

const ROCSOLVER_SVDAlgorithm = Union{ROCSOLVER_QRIteration, ROCSOLVER_Jacobi}

# Various consts and unions
# -------------------------

const GPU_Simple = Union{CUSOLVER_Simple}
const GPU_EigAlgorithm = Union{GPU_Simple}
const GPU_QRIteration = Union{CUSOLVER_QRIteration, ROCSOLVER_QRIteration}
const GPU_Jacobi = Union{CUSOLVER_Jacobi, ROCSOLVER_Jacobi}
const GPU_DivideAndConquer = Union{CUSOLVER_DivideAndConquer, ROCSOLVER_DivideAndConquer}
const GPU_Bisection = Union{ROCSOLVER_Bisection}
const GPU_EighAlgorithm = Union{
    GPU_QRIteration, GPU_Jacobi, GPU_DivideAndConquer, GPU_Bisection,
}
const GPU_SVDAlgorithm = Union{CUSOLVER_SVDAlgorithm, ROCSOLVER_SVDAlgorithm}

const GPU_SVDPolar = Union{CUSOLVER_SVDPolar}
const GPU_Randomized = Union{CUSOLVER_Randomized}

const QRAlgorithms = Union{LAPACK_HouseholderQR, CUSOLVER_HouseholderQR, ROCSOLVER_HouseholderQR}
const LQAlgorithms = Union{LAPACK_HouseholderLQ, LQViaTransposedQR}
const SVDAlgorithms = Union{LAPACK_SVDAlgorithm, GPU_SVDAlgorithm}
const PolarAlgorithms = Union{PolarViaSVD, PolarNewton}

# ================================
# ORTHOGONALIZATION ALGORITHMS
# ================================

"""
    LeftOrthAlgorithm{Kind, Alg <: AbstractAlgorithm}(alg)

Wrapper type to denote the `Kind` of factorization that is used as a backend for [`left_orth`](@ref).
By default `Kind` is a symbol, which can be either `:qr`, `:polar` or `:svd`.
"""
struct LeftOrthAlgorithm{Kind, Alg <: AbstractAlgorithm} <: AbstractAlgorithm
    alg::Alg
end
LeftOrthAlgorithm{Kind}(alg::Alg) where {Kind, Alg <: AbstractAlgorithm} = LeftOrthAlgorithm{Kind, Alg}(alg)

# Note: specific algorithm selection is handled by `left_orth_alg` in orthnull.jl
LeftOrthAlgorithm(alg::AbstractAlgorithm) = error(
    """
    Unkown or invalid `left_orth` algorithm type `$(typeof(alg))`.
    To register the algorithm type for `left_orth`, define

        MatrixAlgebraKit.left_orth_alg(alg::CustomAlgorithm) = LeftOrthAlgorithm{kind}(alg)

    where `kind` selects the factorization type that will be used.
    By default, this is either `:qr`, `:polar` or `:svd`, to select [`qr_compact!`](@ref),
    [`left_polar!`](@ref), [`svd_compact!`](@ref) or [`svd_trunc!`](@ref) respectively.
    """
)

const LeftOrthViaQR = LeftOrthAlgorithm{:qr}
const LeftOrthViaPolar = LeftOrthAlgorithm{:polar}
const LeftOrthViaSVD = LeftOrthAlgorithm{:svd}

"""
    RightOrthAlgorithm{Kind, Alg <: AbstractAlgorithm}(alg)

Wrapper type to denote the `Kind` of factorization that is used as a backend for [`right_orth`](@ref).
By default `Kind` is a symbol, which can be either `:lq`, `:polar` or `:svd`.
"""
struct RightOrthAlgorithm{Kind, Alg <: AbstractAlgorithm} <: AbstractAlgorithm
    alg::Alg
end
RightOrthAlgorithm{Kind}(alg::Alg) where {Kind, Alg <: AbstractAlgorithm} = RightOrthAlgorithm{Kind, Alg}(alg)

# Note: specific algorithm selection is handled by `right_orth_alg` in orthnull.jl
RightOrthAlgorithm(alg::AbstractAlgorithm) = error(
    """
    Unkown or invalid `right_orth` algorithm type `$(typeof(alg))`.
    To register the algorithm type for `right_orth`, define

        MatrixAlgebraKit.right_orth_alg(alg::CustomAlgorithm) = RightOrthAlgorithm{kind}(alg)

    where `kind` selects the factorization type that will be used.
    By default, this is either `:lq`, `:polar` or `:svd`, to select [`lq_compact!`](@ref),
    [`right_polar!`](@ref), [`svd_compact!`](@ref) or [`svd_trunc!`](@ref) respectively.
    """
)

const RightOrthViaLQ = RightOrthAlgorithm{:lq}
const RightOrthViaPolar = RightOrthAlgorithm{:polar}
const RightOrthViaSVD = RightOrthAlgorithm{:svd}

"""
    LeftNullAlgorithm{Kind, Alg <: AbstractAlgorithm}(alg)

Wrapper type to denote the `Kind` of factorization that is used as a backend for [`left_null`](@ref).
By default `Kind` is a symbol, which can be either `:qr` or `:svd`.
"""
struct LeftNullAlgorithm{Kind, Alg <: AbstractAlgorithm} <: AbstractAlgorithm
    alg::Alg
end
LeftNullAlgorithm{Kind}(alg::Alg) where {Kind, Alg <: AbstractAlgorithm} = LeftNullAlgorithm{Kind, Alg}(alg)

# Note: specific algorithm selection is handled by `left_null_alg` in orthnull.jl
LeftNullAlgorithm(alg::AbstractAlgorithm) = error(
    """
    Unkown or invalid `left_null` algorithm type `$(typeof(alg))`.
    To register the algorithm type for `left_null`, define

        MatrixAlgebraKit.left_null_alg(alg::CustomAlgorithm) = LeftNullAlgorithm{kind}(alg)

    where `kind` selects the factorization type that will be used.
    By default, this is either `:qr` or `:svd`, to select [`qr_null!`](@ref),
    [`svd_compact!`](@ref) or [`svd_trunc!`](@ref) respectively.
    """
)

const LeftNullViaQR = LeftNullAlgorithm{:qr}
const LeftNullViaSVD = LeftNullAlgorithm{:svd}

"""
    RightNullAlgorithm{Kind, Alg <: AbstractAlgorithm}(alg)

Wrapper type to denote the `Kind` of factorization that is used as a backend for [`right_null`](@ref).
By default `Kind` is a symbol, which can be either `:lq` or `:svd`.
"""
struct RightNullAlgorithm{Kind, Alg <: AbstractAlgorithm} <: AbstractAlgorithm
    alg::Alg
end
RightNullAlgorithm{Kind}(alg::Alg) where {Kind, Alg <: AbstractAlgorithm} = RightNullAlgorithm{Kind, Alg}(alg)

# Note: specific algorithm selection is handled by `right_null_alg` in orthnull.jl
RightNullAlgorithm(alg::AbstractAlgorithm) = error(
    """
    Unkown or invalid `right_null` algorithm type `$(typeof(alg))`.
    To register the algorithm type for `right_null`, define

        MatrixAlgebraKit.right_null_alg(alg::CustomAlgorithm) = RightNullAlgorithm{kind}(alg)

    where `kind` selects the factorization type that will be used.
    By default, this is either `:lq` or `:svd`, to select [`lq_null!`](@ref),
    [`svd_compact!`](@ref) or [`svd_trunc!`](@ref) respectively.
    """
)

const RightNullViaLQ = RightNullAlgorithm{:lq}
const RightNullViaSVD = RightNullAlgorithm{:svd}
