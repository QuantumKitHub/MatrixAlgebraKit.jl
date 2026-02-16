"""
    iszerotangent(x)

Return true if `x` is of a type that the different AD engines use to communicate
a (co)tangent that is identically zero. By overloading this method, and writing
pullback definitions in term of it, we will be able to hook into different AD
ecosystems
"""
function iszerotangent end

iszerotangent(::Any) = false
iszerotangent(::Nothing) = true

# Solve the Sylvester equation A*X + X*B + C = 0.
# When A === B (same Hermitian PD matrix, as in polar pullbacks), use an
# eigendecomposition-based solver to avoid LAPACK's trsyl! failing with
# LAPACKException(1) for close eigenvalues.
function _sylvester(A, B, C)
    if A === B
        return _sylvester_symm(A, C)
    end
    return LinearAlgebra.sylvester(A, B, C)
end

function _sylvester_symm(P, C)
    D, Q = LinearAlgebra.eigen(LinearAlgebra.Hermitian(P))
    Y = Q' * C * Q
    @inbounds for j in axes(Y, 2), i in axes(Y, 1)
        Y[i, j] = -Y[i, j] / (D[i] + D[j])
    end
    return Q * Y * Q'
end
