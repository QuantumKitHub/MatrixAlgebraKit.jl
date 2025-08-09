module MatrixAlgebraKitCUDAExt

using MatrixAlgebraKit
using MatrixAlgebraKit: @algdef, Algorithm, check_input
using MatrixAlgebraKit: one!, zero!, uppertriangular!, lowertriangular!
using MatrixAlgebraKit: diagview, sign_safe
using MatrixAlgebraKit: LQViaTransposedQR
using MatrixAlgebraKit: default_qr_algorithm, default_lq_algorithm, default_svd_algorithm 
using CUDA
using LinearAlgebra
using LinearAlgebra: BlasFloat

include("yacusolver.jl")
include("implementations/qr.jl")
include("implementations/svd.jl")

function MatrixAlgebraKit.default_qr_algorithm(::Type{T}; kwargs...) where {T<:StridedCuMatrix}
    return CUSOLVER_HouseholderQR(; kwargs...)
end
function MatrixAlgebraKit.default_lq_algorithm(::Type{T}; kwargs...) where {T<:StridedCuMatrix}
    qr_alg = CUSOLVER_HouseholderQR(; kwargs...)
    return LQViaTransposedQR(qr_alg)
end
function MatrixAlgebraKit.default_svd_algorithm(::Type{T}; kwargs...) where {T<:StridedCuMatrix}
    return CUSOLVER_QRIteration(; kwargs...)
end

end
