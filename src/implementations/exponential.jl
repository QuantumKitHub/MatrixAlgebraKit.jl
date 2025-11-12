# Inputs
# ------
function copy_input(::typeof(exponential), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end

copy_input(::typeof(exponential), A::Diagonal) = copy(A)

function check_input(::typeof(exponential!), A::AbstractMatrix, expA::AbstractMatrix, alg::AbstractAlgorithm)
    m, n = size(A)
    m == n || throw(DimensionMismatch("square input matrix expected"))
    @assert expA isa AbstractMatrix
    @check_size(expA, (m, m))
    return @check_scalar(expA, A)
end

# Outputs
# -------
function initialize_output(::typeof(exponential!), A::AbstractMatrix, ::AbstractAlgorithm)
    n = size(A, 1) # square check will happen later
    expA = similar(A, (n, n))
    return expA
end

# Implementation
# --------------
function exponential!(A::AbstractMatrix, expA::AbstractMatrix, alg::ExponentialViaLA)
    copyto!(expA, LinearAlgebra.exp(A))
    return expA
end

function exponential!(A::AbstractMatrix, expA::AbstractMatrix, alg::ExponentialViaEigh)
    D, V = eigh_full(A, alg.eigh_alg)
    copyto!(expA, V * Diagonal(exp.(diagview(D))) * inv(V))
    return expA
end

function exponential!(A::AbstractMatrix, expA::AbstractMatrix, alg::ExponentialViaEig)
    D, V = eig_full(A, alg.eig_alg)
    copyto!(expA, V * Diagonal(exp.(diagview(D))) * inv(V))
    return expA
end

# Diagonal logic
# --------------
function exponential!(A::Diagonal, expA, alg::DiagonalAlgorithm)
    check_input(exponential!, A, expA, alg)
    copyto!(expA, Diagonal(LinearAlgebra.exp.(diagview(A))))
    return expA
end
