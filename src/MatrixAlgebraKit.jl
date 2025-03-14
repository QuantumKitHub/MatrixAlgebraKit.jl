module MatrixAlgebraKit

using LinearAlgebra: LinearAlgebra
using LinearAlgebra: norm # TODO: eleminate if we use VectorInterface.jl?
using LinearAlgebra: mul!, rmul!, lmul!
using LinearAlgebra: sylvester
using LinearAlgebra: isposdef, ishermitian
using LinearAlgebra: Diagonal, diag, diagind
using LinearAlgebra: UpperTriangular, LowerTriangular
using LinearAlgebra: BlasFloat, BlasReal, BlasComplex, BlasInt, triu!, tril!, rdiv!, ldiv!

export qr_compact, qr_full, qr_null, lq_compact, lq_full, lq_null
export qr_compact!, qr_full!, qr_null!, lq_compact!, lq_full!, lq_null!
export svd_compact, svd_full, svd_vals, svd_trunc
export svd_compact!, svd_full!, svd_vals!, svd_trunc!
export eigh_full, eigh_vals, eigh_trunc
export eigh_full!, eigh_vals!, eigh_trunc!
export eig_full, eig_vals, eig_trunc
export eig_full!, eig_vals!, eig_trunc!
export schur_full, schur_vals
export schur_full!, schur_vals!
export left_polar, right_polar
export left_polar!, right_polar!
export left_orth, right_orth, left_null, right_null
export left_orth!, right_orth!, left_null!, right_null!

export LAPACK_HouseholderQR, LAPACK_HouseholderLQ,
       LAPACK_Simple, LAPACK_Expert,
       LAPACK_QRIteration, LAPACK_Bisection, LAPACK_MultipleRelativelyRobustRepresentations,
       LAPACK_DivideAndConquer, LAPACK_Jacobi
export truncrank, trunctol, truncabove, TruncationKeepSorted, TruncationKeepFiltered

include("common/defaults.jl")
include("common/initialization.jl")
include("common/pullbacks.jl")
include("common/safemethods.jl")
include("common/view.jl")
include("common/regularinv.jl")

include("yalapack.jl")
include("algorithms.jl")
include("interface/qr.jl")
include("interface/lq.jl")
include("interface/svd.jl")
include("interface/eig.jl")
include("interface/eigh.jl")
include("interface/schur.jl")
include("interface/polar.jl")
include("interface/orthnull.jl")

include("implementations/decompositions.jl")
include("implementations/truncation.jl")
include("implementations/qr.jl")
include("implementations/lq.jl")
include("implementations/svd.jl")
include("implementations/eig.jl")
include("implementations/eigh.jl")
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
