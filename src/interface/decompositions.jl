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

const LAPACK_EigAlgorithm = Union{LAPACK_Simple,LAPACK_Expert}

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

const LAPACK_EighAlgorithm = Union{LAPACK_QRIteration,
                                   LAPACK_Bisection,
                                   LAPACK_DivideAndConquer,
                                   LAPACK_MultipleRelativelyRobustRepresentations}

# Singular Value Decomposition
# ----------------------------
"""
    LAPACK_Jacobi()

Algorithm type to denote the LAPACK driver for computing the singular value decomposition of
a general matrix using the Jacobi algorithm.
"""
@algdef LAPACK_Jacobi

const LAPACK_SVDAlgorithm = Union{LAPACK_QRIteration,
                                  LAPACK_Bisection,
                                  LAPACK_DivideAndConquer,
                                  LAPACK_Jacobi}

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
    CUSOLVER_Randomized(; p, niters)

Algorithm type to denote the CUSOLVER driver for computing the singular value decomposition of
a general matrix using the randomized SVD algorithm.

!!! note
  Randomized SVD cannot compute all singular values of the input matrix `A`, only the first `k` where
  `k < min(m, n)`. The remainder are used for oversampling. See the [CUSOLVER documentation](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnxgesvdr)
  for more information.
"""
@algdef CUSOLVER_Randomized

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


const GPU_Simple = Union{CUSOLVER_Simple}
const GPU_EigAlgorithm = Union{GPU_Simple}
const GPU_QRIteration = Union{CUSOLVER_QRIteration, ROCSOLVER_QRIteration}
const GPU_Jacobi = Union{CUSOLVER_Jacobi, ROCSOLVER_Jacobi}
const GPU_DivideAndConquer = Union{CUSOLVER_DivideAndConquer, ROCSOLVER_DivideAndConquer}
const GPU_Bisection = Union{ROCSOLVER_Bisection}
const GPU_EighAlgorithm = Union{GPU_QRIteration,
                                GPU_Jacobi,
                                GPU_DivideAndConquer,
                                GPU_Bisection}

