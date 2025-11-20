# Inputs
# ------
function copy_input(::typeof(exponential), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end

copy_input(::typeof(exponential), A::Diagonal) = copy(A)

function check_input(::typeof(exponential!), A::AbstractMatrix, expA::AbstractMatrix, alg::AbstractAlgorithm)
    m, n = size(A)
    m == n || throw(DimensionMismatch("square input matrix expected. Got ($m,$n)"))
    @check_size(expA, (m, m))
    return @check_scalar(expA, A)
end

function check_input(::typeof(exponential!), A::AbstractMatrix, expA::AbstractMatrix, alg::MatrixFunctionViaEigh)
    if !ishermitian(A)
        throw(DomainError(A, "Hermitian matrix was expected. Use `project_hermitian` to project onto the nearest hermitian matrix)"))
    end
    m, n = size(A)
    m == n || throw(DimensionMismatch("square input matrix expected. Got ($m,$n)"))
    @check_size(expA, (m, m))
    return @check_scalar(expA, A)
end

function check_input(::typeof(exponential!), A::AbstractMatrix, expA::AbstractMatrix, ::DiagonalAlgorithm)
    m, n = size(A)
    @assert m == n && isdiag(A)
    @assert expA isa Diagonal
    @check_size(expA, (m, m))
    @check_scalar(expA, A)
    return nothing
end

# Outputs
# -------
function initialize_output(::typeof(exponential!), A::AbstractMatrix, ::AbstractAlgorithm)
    n = size(A, 1) # square check will happen later
    expA = similar(A, (n, n))
    return expA
end

function initialize_output(::typeof(exponential!), A::Diagonal, ::DiagonalAlgorithm)
    return similar(A)
end

# Implementation
# --------------
function exponential!(A::AbstractMatrix{T}, expA::AbstractMatrix{T}, alg::MatrixFunctionViaLA) where {T <: BlasFloat}
    check_input(exponential!, A, expA, alg)
    copyto!(expA, LinearAlgebra.exp(A))
    return expA
end

function exponential!(A::AbstractMatrix, expA::AbstractMatrix, alg::MatrixFunctionViaEigh)
    check_input(exponential!, A, expA, alg)
    D, V = eigh_full(A, alg.eigh_alg)

    diagview(D) .= exp.(diagview(D) ./ 2)
    rmul!(V, D)
    mul!(expA, V, adjoint(V))
    return expA
end

function exponential!(A::AbstractMatrix{T}, expA::AbstractMatrix{T}, alg::MatrixFunctionViaEig) where {T <: Real}
    check_input(exponential!, A, expA, alg)
    D, V = eig_full(A, alg.eig_alg)
    iV = inv(V)
    map!(exp, diagview(D), diagview(D))
    expA .= real.(rmul!(V, D) * iV)
    return expA
end

function exponential!(A::AbstractMatrix, expA::AbstractMatrix, alg::MatrixFunctionViaEig)
    check_input(exponential!, A, expA, alg)
    D, V = eig_full(A, alg.eig_alg)
    iV = inv(V)
    map!(exp, diagview(D), diagview(D))
    mul!(expA, rmul!(V, D), iV)
    return expA
end

# Diagonal logic
# --------------
function exponential!(A::Diagonal, expA, alg::DiagonalAlgorithm)
    check_input(exponential!, A, expA, alg)
    map!(exp, diagview(expA), diagview(A))
    return expA
end
