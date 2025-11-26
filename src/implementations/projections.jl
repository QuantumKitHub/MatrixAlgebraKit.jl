# Inputs
# ------
function copy_input(::typeof(project_hermitian), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
copy_input(::typeof(project_antihermitian), A) = copy_input(project_hermitian, A)

copy_input(::typeof(project_isometric), A) = copy_input(left_polar, A)

function check_input(::typeof(project_hermitian!), A::AbstractMatrix, B::AbstractMatrix, ::AbstractAlgorithm)
    LinearAlgebra.checksquare(A)
    Base.require_one_based_indexing(A)
    n = size(A, 1)
    B === A || @check_size(B, (n, n))
    return nothing
end
function check_input(::typeof(project_antihermitian!), A::AbstractMatrix, B::AbstractMatrix, ::AbstractAlgorithm)
    LinearAlgebra.checksquare(A)
    Base.require_one_based_indexing(A)
    n = size(A, 1)
    B === A || @check_size(B, (n, n))
    return nothing
end

function check_input(::typeof(project_isometric!), A::AbstractMatrix, W::AbstractMatrix, ::AbstractAlgorithm)
    m, n = size(A)
    m >= n ||
        throw(ArgumentError("input matrix needs at least as many rows as columns"))
    @assert W isa AbstractMatrix
    @check_size(W, (m, n))
    @check_scalar(W, A)
    return nothing
end

# Outputs
# -------
function initialize_output(::typeof(project_hermitian!), A::AbstractMatrix, ::NativeBlocked)
    return A
end
function initialize_output(::typeof(project_antihermitian!), A::AbstractMatrix, ::NativeBlocked)
    return A
end

function initialize_output(::typeof(project_isometric!), A::AbstractMatrix, ::AbstractAlgorithm)
    return similar(A)
end

# Implementation
# --------------
function project_hermitian!(A::AbstractMatrix, Aₕ, alg::NativeBlocked)
    check_input(project_hermitian!, A, Aₕ, alg)
    return project_hermitian_native!(A, Aₕ, Val(false); alg.kwargs...)
end
function project_antihermitian!(A::AbstractMatrix, Aₐ, alg::NativeBlocked)
    check_input(project_antihermitian!, A, Aₐ, alg)
    return project_hermitian_native!(A, Aₐ, Val(true); alg.kwargs...)
end

function project_isometric!(A::AbstractMatrix, W, alg::AbstractAlgorithm)
    check_input(project_isometric!, A, W, alg)
    noP = similar(W, (0, 0))
    W, _ = left_polar!(A, (W, noP), alg)
    return W
end

function project_hermitian_native!(A::Diagonal, B::Diagonal, ::Val{anti}; kwargs...) where {anti}
    if anti
        diagview(A) .= _imimag.(diagview(B))
    else
        diagview(A) .= real.(diagview(B))
    end
    return A
end

function project_hermitian_native!(A::AbstractMatrix, B::AbstractMatrix, anti::Val; blocksize = 32)
    n = size(A, 1)
    for j in 1:blocksize:n
        for i in 1:blocksize:(j - 1)
            jb = min(blocksize, n - j + 1)
            ib = blocksize
            _project_hermitian_offdiag!(
                view(A, i:(i + ib - 1), j:(j + jb - 1)),
                view(A, j:(j + jb - 1), i:(i + ib - 1)),
                view(B, i:(i + ib - 1), j:(j + jb - 1)),
                view(B, j:(j + jb - 1), i:(i + ib - 1)),
                anti
            )
        end
        jb = min(blocksize, n - j + 1)
        _project_hermitian_diag!(
            view(A, j:(j + jb - 1), j:(j + jb - 1)),
            view(B, j:(j + jb - 1), j:(j + jb - 1)),
            anti
        )
    end
    return B
end

@inline function _project_hermitian(Aij::Number, Aji::Number, anti::Bool)
    return anti ? (Aij - Aji') / 2 : (Aij + Aji') / 2
end
function _project_hermitian_offdiag!(
        Au::AbstractMatrix, Al::AbstractMatrix, Bu::AbstractMatrix, Bl::AbstractMatrix, ::Val{anti}
    ) where {anti}
    m, n = size(Au) # == reverse(size(Au))
    return @inbounds for j in 1:n
        @simd for i in 1:m
            val = _project_hermitian(Au[i, j], Al[j, i], anti)
            Bu[i, j] = val
            aval = adjoint(val)
            Bl[j, i] = anti ? -aval : aval
        end
    end
    return nothing
end
function _project_hermitian_diag!(A::AbstractMatrix, B::AbstractMatrix, ::Val{anti}) where {anti}
    n = size(A, 1)
    @inbounds for j in 1:n
        @simd for i in 1:(j - 1)
            val = _project_hermitian(A[i, j], A[j, i], anti)
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
