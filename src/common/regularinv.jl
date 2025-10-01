# Methods for regularizing inverses of linear problems.
# TODO: the current definitions probably need a more generic interface that
# can be more easily be extended or configured

"""
    inv_regularized(a::Number, tol = defaulttol(a))
    inv_regularized(A::Matrix, tol = defaulttol(A); isposdef = false, kwargs...)

Compute a smooth regularised inverse (L2 Tikhonov regularisation) of a number or square 
matrix a.

*   For numbers, this is given by `inv(hypot(a, tol))`.

*   For matrices, this is computed using the singular value decomposition and aplying
    `inv_regularized` to the singular values. If `isposdef = true`, the singular value
    decomposition is equivalent to the (Hermitian) eigenvalue decomposition of `A` and
    the latter is used instead.
"""
inv_regularized(a::Number, tol = defaulttol(a)) = inv(hypot(a, tol))
function inv_regularized(
        A::AbstractMatrix, tol = defaulttol(A); isposdef = isposdef(A), kwargs...
    )
    if isposdef
        D, V = eigh_full(A; kwargs...)
        Dinvsqrt = Diagonal(sqrt.(inv_regularized.(D.diag, tol)))
        VDinvsqrt = rmul!(V, Dinvsqrt)
        return VDinvsqrt * VDinvsqrt'
    else
        U, S, Vᴴ = svd_compact(A; kwargs...)
        Sinv = Diagonal(inv_regularized.(S.diag, tol))
        USinv = rmul!(U, Sinv)
        return Vᴴ' * USinv'
    end
end

function sylvester_regularized(
        A::AbstractArray, B::AbstractMatrix, tol = defaulttol(A);
        ishermitian = ishermitian(A) && ishermitian(B), kwargs...
    )
    if ishermitian
        D, U = eigh_full(A; kwargs...)
        UdCU = U' * B * U
        UdCU .*= inv_regularized.(D.diag .+ transpose(D.diag), tol)
        C = U * UdCU * U'
        return C
    else
        throw(ArgumentError("Nonhermitian regularised Sylvester equation not yet implemented"))
    end
end
