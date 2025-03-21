# TODO: module Decompositions?

# ==========
# ALGORITHMS
# ==========

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
