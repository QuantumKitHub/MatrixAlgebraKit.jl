# Inputs
# ------
function copy_input(::typeof(exponential), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end

copy_input(::typeof(exponential), A::Diagonal) = copy(A)
copy_input(::typeof(exponential), (τ, A)::Tuple{Number, AbstractMatrix}) = (τ, copy!(similar(A, float(eltype(A))), A))
copy_input(::typeof(exponential), (τ, A)::Tuple{Number, Diagonal}) = τ, copy(A)

function check_input(::typeof(exponential!), A::AbstractMatrix, expA::AbstractMatrix, alg::AbstractAlgorithm)
    m, n = size(A)
    m == n || throw(DimensionMismatch("square input matrix expected. Got ($m,$n)"))
    @check_size(expA, (m, m))
    @check_scalar(expA, A)
    return nothing
end

function check_input(::typeof(exponential!), A::AbstractMatrix, expA::AbstractMatrix, alg::MatrixFunctionViaEigh)
    m, n = size(A)
    m == n || throw(DimensionMismatch("square input matrix expected. Got ($m,$n)"))
    @check_size(expA, (m, m))
    @check_scalar(expA, A)
    return nothing
end

function check_input(::typeof(exponential!), A::AbstractMatrix, expA::AbstractMatrix, ::DiagonalAlgorithm)
    m, n = size(A)
    @assert m == n && isdiag(A)
    @assert expA isa Diagonal
    @check_size(expA, (m, m))
    @check_scalar(expA, A)
    return nothing
end

function check_input(::typeof(exponential!), τ::Number, A::AbstractMatrix, expA::AbstractMatrix, alg::AbstractAlgorithm)
    m, n = size(A)
    m == n || throw(DimensionMismatch("square input matrix expected. Got ($m,$n)"))
    @check_size(expA, (m, m))
    (τ isa Real) ? @check_scalar(expA, A) : @check_scalar(expA, A, complex)
    return nothing
end

function check_input(::typeof(exponential!), τ::Number, A::AbstractMatrix, expA::AbstractMatrix, ::DiagonalAlgorithm)
    m, n = size(A)
    @assert m == n && isdiag(A)
    @assert expA isa Diagonal
    @check_size(expA, (m, m))
    (τ isa Real) ? @check_scalar(expA, A) : @check_scalar(expA, A, complex)
    return nothing
end

# Outputs
# -------
initialize_output(::typeof(exponential!), A::AbstractMatrix, ::AbstractAlgorithm) = A
initialize_output(::typeof(exponential!), (τ, A)::Tuple{T, AbstractMatrix}, ::AbstractAlgorithm) where {T <: Real} = A
initialize_output(::typeof(exponential!), (τ, A)::Tuple{Number, AbstractMatrix}, ::AbstractAlgorithm) = complex(A)

# Implementation
# --------------
function exponential!(A, expA, alg::MatrixFunctionViaLA)
    check_input(exponential!, A, expA, alg)
    A = LinearAlgebra.exp!(A)
    A === expA || copy!(expA, A)
    return expA
end

function exponential!(A, expA, alg::MatrixFunctionViaEigh)
    check_input(exponential!, A, expA, alg)
    D, V = eigh_full!(A, alg.eigh_alg)
    expD = map_diagonal!(x -> exp(x / 2), D, D)
    VexpD = rmul!(V, expD)
    return mul!(expA, VexpD, V')
end

function exponential!(A::AbstractMatrix, expA::AbstractMatrix, alg::MatrixFunctionViaEig)
    check_input(exponential!, A, expA, alg)
    D, V = eig_full!(A, alg.eig_alg)
    expD = map_diagonal!(exp, D, D)
    iV = inv(V)
    VexpD = rmul!(V, expD)
    if eltype(A) <: Real
        expA .= real.(VexpD * iV)
    else
        mul!(expA, VexpD, iV)
    end
    return expA
end

function exponential!((τ, A)::Tuple{Number, AbstractMatrix}, expA::AbstractMatrix, alg::MatrixFunctionViaLA)
    check_input(exponential!, τ, A, expA, alg)
    expA .= A .* τ
    return LinearAlgebra.exp!(expA)
end

function exponential!((τ, A)::Tuple{Number, AbstractMatrix}, expA::AbstractMatrix, alg::MatrixFunctionViaEigh)
    check_input(exponential!, τ, A, expA, alg)
    D, V = eigh_full!(A, alg.eigh_alg)
    expD = map_diagonal(x -> exp(x * τ), D)
    VexpD = V * expD
    if eltype(A) <: Real && eltype(τ) <: Real
        return expA .= real.(VexpD * V')
    else
        return mul!(expA, VexpD, V')
    end
end

function exponential!((τ, A)::Tuple{Number, AbstractMatrix}, expA, alg::MatrixFunctionViaEig)
    check_input(exponential!, τ, A, expA, alg)
    D, V = eig_full!(A, alg.eig_alg)
    expD = map_diagonal!(x -> exp(x * τ), D, D)
    iV = inv(V)
    VexpD = rmul!(V, expD)
    if eltype(A) <: Real && eltype(τ) <: Real
        expA .= real.(VexpD * iV)
        return expA
    else
        return mul!(expA, VexpD, iV)
    end
end

# Diagonal logic
# --------------
function exponential!(A, expA, alg::DiagonalAlgorithm)
    check_input(exponential!, A, expA, alg)
    return map_diagonal!(exp, expA, A)
end

function exponential!((τ, A)::Tuple{Number, AbstractMatrix}, expA, alg::DiagonalAlgorithm)
    check_input(exponential!, τ, A, expA, alg)
    return map_diagonal!(x -> exp(x * τ), expA, A)
end
