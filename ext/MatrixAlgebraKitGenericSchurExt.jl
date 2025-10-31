module MatrixAlgebraKitGenericSchurExt

using MatrixAlgebraKit
using MatrixAlgebraKit: check_input
using LinearAlgebra: Diagonal
using GenericSchur

function MatrixAlgebraKit.eig_full!(A::AbstractMatrix{T}, DV, alg::GLA_eig_Francis) where {T}
    check_input(eig_full!, A, DV, alg)
    D, V = DV
    D̃, Ṽ = GenericSchur.eigen!(A)
    copyto!(D, Diagonal(D̃))
    copyto!(V, Ṽ)
    return D, V
end

function MatrixAlgebraKit.eig_vals!(A::AbstractMatrix{T}, D, alg::GLA_eig_Francis) where {T}
    check_input(eig_vals!, A, D, alg)
    eigval = GenericSchur.eigvals!(A)
    copyto!(D, eigval)
    return D
end

end
