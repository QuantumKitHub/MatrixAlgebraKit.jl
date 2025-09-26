module MatrixAlgebraKit

using LinearAlgebra: LinearAlgebra
using LinearAlgebra: norm # TODO: eleminate if we use VectorInterface.jl?
using LinearAlgebra: mul!, rmul!, lmul!, adjoint!, rdiv!, ldiv!
using LinearAlgebra: sylvester
using LinearAlgebra: isposdef, ishermitian, issymmetric
using LinearAlgebra: Diagonal, diag, diagind, isdiag
using LinearAlgebra: UpperTriangular, LowerTriangular
using LinearAlgebra: BlasFloat, BlasReal, BlasComplex, BlasInt

export isisometry, isunitary

export qr_compact, qr_full, qr_null, lq_compact, lq_full, lq_null
export qr_compact!, qr_full!, qr_null!, lq_compact!, lq_full!, lq_null!
export svd_compact, svd_full, svd_vals, svd_trunc
export svd_compact!, svd_full!, svd_vals!, svd_trunc!
export eigh_full, eigh_vals, eigh_trunc
export eigh_full!, eigh_vals!, eigh_trunc!
export eig_full, eig_vals, eig_trunc
export eig_full!, eig_vals!, eig_trunc!
export gen_eig_full, gen_eig_vals
export gen_eig_full!, gen_eig_vals!
export schur_full, schur_vals
export schur_full!, schur_vals!
export left_polar, right_polar
export left_polar!, right_polar!
export left_orth, right_orth, left_null, right_null
export left_orth!, right_orth!, left_null!, right_null!

export LAPACK_HouseholderQR, LAPACK_HouseholderLQ, LAPACK_Simple, LAPACK_Expert, LAPACK_QRIteration, LAPACK_Bisection, LAPACK_MultipleRelativelyRobustRepresentations, LAPACK_DivideAndConquer, LAPACK_Jacobi
export LQViaTransposedQR
export DiagonalAlgorithm
export CUSOLVER_Simple, CUSOLVER_HouseholderQR, CUSOLVER_QRIteration, CUSOLVER_SVDPolar, CUSOLVER_Jacobi, CUSOLVER_Randomized, CUSOLVER_DivideAndConquer
export ROCSOLVER_HouseholderQR, ROCSOLVER_QRIteration, ROCSOLVER_Jacobi,
       ROCSOLVER_DivideAndConquer, ROCSOLVER_Bisection

export notrunc, truncrank, trunctol, truncerror, truncfilter

@static if VERSION >= v"1.11.0-DEV.469"
    eval(Expr(:public, :default_algorithm, :findtruncated, :findtruncated_svd,
              :select_algorithm))
    eval(Expr(:public, :TruncationByOrder, :TruncationByFilter, :TruncationByValue,
              :TruncationByError, :TruncationIntersection))
end

include("common/defaults.jl")
include("common/initialization.jl")
include("common/pullbacks.jl")
include("common/safemethods.jl")
include("common/view.jl")
include("common/regularinv.jl")
include("common/matrixproperties.jl")
include("common/gauge.jl")

include("yalapack.jl")
include("algorithms.jl")
include("interface/decompositions.jl")
include("interface/truncation.jl")
include("interface/qr.jl")
include("interface/lq.jl")
include("interface/svd.jl")
include("interface/eig.jl")
include("interface/eigh.jl")
include("interface/gen_eig.jl")
include("interface/schur.jl")
include("interface/polar.jl")
include("interface/orthnull.jl")

include("implementations/truncation.jl")
include("implementations/qr.jl")
include("implementations/lq.jl")
include("implementations/svd.jl")
include("implementations/eig.jl")
include("implementations/eigh.jl")
include("implementations/gen_eig.jl")
include("implementations/schur.jl")
include("implementations/polar.jl")
include("implementations/orthnull.jl")

include("pullbacks/qr.jl")
include("pullbacks/lq.jl")
include("pullbacks/eig.jl")
include("pullbacks/eigh.jl")
include("pullbacks/svd.jl")
include("pullbacks/polar.jl")

end
