# Inputs
# ------
function copy_input(::typeof(exp), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end

copy_input(::typeof(exp), A::Diagonal) = copy(A)

function check_input(::typeof(exp!), A::AbstractMatrix, expA::AbstractMatrix, alg::AbstractAlgorithm)
    m, n = size(A)
    m == n || throw(DimensionMismatch("square input matrix expected"))
    @assert expA isa AbstractMatrix
    @check_size(expA, (m, m))
    @check_scalar(expA, A) 
end

# Outputs
# -------
function initialize_output(::typeof(exp!), A::AbstractMatrix, ::AbstractAlgorithm)
    n = size(A, 1) # square check will happen later
    expA = similar(A, (n, n))
    return expA
end

# Implementation
# --------------
function exp!(A::AbstractMatrix, expA::AbstractMatrix, alg::AbstractAlgorithm)
    copyto!(expA, LinearAlgebra.exp!(A))
    return A
end

function MatrixAlgebraKit.exp!(A::AbstractMatrix, expA::AbstractMatrix, alg::ExponentialViaEigh)
    D, V = eigh_full(A, alg.eigh_alg)
    return V * Diagonal(exp.(diagview(D))) * inv(V)
end

function MatrixAlgebraKit.exp!(A::AbstractMatrix, expA::AbstractMatrix, alg::ExponentialViaEig)
    D, V = eig_full(A, alg.eig_alg)
    return V * Diagonal(exp.(diagview(D))) * inv(V)
end

# Diagonal logic
# --------------
function exp!(A::Diagonal, expA, alg::DiagonalAlgorithm)
    check_input(exp!, A, expA, alg)
    copyto!(expA, Diagonal(LinearAlgebra.exp.(diagview(A))))
    return expA
end
