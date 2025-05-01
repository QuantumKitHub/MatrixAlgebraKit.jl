module MatrixAlgebraKitCUDAExt

using MatrixAlgebraKit
using MatrixAlgebraKit: @algdef, Algorithm, check_input
using MatrixAlgebraKit: one!, zero!, uppertriangular!, lowertriangular!
using MatrixAlgebraKit: diagview, sign_safe
using MatrixAlgebraKit: LQViaTransposedQR
using CUDA
using LinearAlgebra
using LinearAlgebra: BlasFloat

include("yacusolver.jl")
include("implementations/qr.jl")
include("implementations/svd.jl")

function MatrixAlgebraKit.default_qr_algorithm(A::CuMatrix{<:BlasFloat}; kwargs...)
    return CUSOLVER_HouseholderQR(; kwargs...)
end
function MatrixAlgebraKit.default_lq_algorithm(A::CuMatrix{<:BlasFloat}; kwargs...)
    qr_alg = CUSOLVER_HouseholderQR(; kwargs...)
    return LQViaTransposedQR(qr_alg)
end
function MatrixAlgebraKit.default_svd_algorithm(A::CuMatrix{<:BlasFloat}; kwargs...)
    return CUSOLVER_QRIteration(; kwargs...)
end

end