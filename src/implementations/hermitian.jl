# Inputs
# ------
function copy_input(::typeof(hermitianpart), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
copy_input(::typeof(antihermitianpart), A) = copy_input(hermitianpart, A)

function check_input(::typeof(hermitianpart!), A::AbstractMatrix, B::AbstractMatrix, ::AbstractAlgorithm)
    LinearAlgebra.checksquare(A)
    n = Base.require_one_based_indexing(A)
    B === A || @check_size(B, (n, n))
    return nothing
end
function check_input(::typeof(antihermitianpart!), A::AbstractMatrix, B::AbstractMatrix, ::AbstractAlgorithm)
    LinearAlgebra.checksquare(A)
    n = Base.require_one_based_indexing(A)
    B === A || @check_size(B, (n, n))
    return nothing
end

# Outputs
# -------
function initialize_output(::typeof(hermitianpart!), A::AbstractMatrix, ::NativeBlocked)
    return A
end
function initialize_output(::typeof(antihermitianpart!), A::AbstractMatrix, ::NativeBlocked)
    return A
end

# Implementation
# --------------
function hermitianpart!(A::AbstractMatrix, B, alg::NativeBlocked)
    check_input(hermitianpart!, A, B, alg)
    return hermitianpart_native!(A, B, Val(false); alg.kwargs...)
end
function antihermitianpart!(A::AbstractMatrix, B, alg::NativeBlocked)
    check_input(antihermitianpart!, A, B, alg)
    return hermitianpart_native!(A, B, Val(true); alg.kwargs...)
end

function hermitianpart_native!(A::AbstractMatrix, B::AbstractMatrix, anti::Val; blocksize = 32)
    n = size(A, 1)
    for j in 1:blocksize:n
        for i in 1:blocksize:(j - 1)
            jb = min(blocksize, n - j + 1)
            ib = blocksize
            _hermitianpart_offdiag!(
                view(A, i:(i + ib - 1), j:(j + jb - 1)),
                view(A, j:(j + jb - 1), i:(i + ib - 1)),
                view(B, i:(i + ib - 1), j:(j + jb - 1)),
                view(B, j:(j + jb - 1), i:(i + ib - 1)),
                anti
            )
        end
        jb = min(blocksize, n - j + 1)
        _hermitianpart_diag!(
            view(A, j:(j + jb - 1), j:(j + jb - 1)),
            view(B, j:(j + jb - 1), j:(j + jb - 1)),
            anti
        )
    end
    return B
end

function _hermitianpart_offdiag!(
        Au::AbstractMatrix, Al::AbstractMatrix, Bu::AbstractMatrix, Bl::AbstractMatrix, ::Val{anti}
    ) where {anti}

    m, n = size(Au) # == reverse(size(Au))
    return @inbounds for j in 1:n
        @simd for i in 1:m
            val = anti ? (Au[i, j] - adjoint(Al[j, i])) / 2 : (Au[i, j] + adjoint(Al[j, i])) / 2
            Bu[i, j] = val
            aval = adjoint(val)
            Bl[j, i] = anti ? -aval : aval
        end
    end
    return nothing
end
function _hermitianpart_diag!(A::AbstractMatrix, B::AbstractMatrix, ::Val{anti}) where {anti}
    n = size(A, 1)
    @inbounds for j in 1:n
        @simd for i in 1:(j - 1)
            val = anti ? (A[i, j] - adjoint(A[j, i])) / 2 : (A[i, j] + adjoint(A[j, i])) / 2
            B[i, j] = val
            aval = adjoint(val)
            B[j, i] = anti ? -aval : aval
        end
        B[j, j] = anti ? _imimag(A[j, j]) : real(A[j, j])
    end
    return nothing
end

_imimag(x::Real) = zero(x)
_imimag(x::Complex) = im * imag(x)
