# Inputs
# ------
function copy_input(::typeof(exponential), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end

copy_input(::typeof(exponential), A::Diagonal) = copy(A)

function copy_input(::typeof(exponentiali), τ::Number, A::AbstractMatrix)
    return τ, copy!(similar(A, complex(eltype(A))), A)
end

copy_input(::typeof(exponentiali), τ::Number, A::Diagonal) = τ, copy(A)

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

function check_input(::typeof(exponentiali!), A::AbstractMatrix, expA::AbstractMatrix, alg::AbstractAlgorithm)
    m, n = size(A)
    m == n || throw(DimensionMismatch("square input matrix expected. Got ($m,$n)"))
    return @check_size(expA, (m, m))
end

function check_input(::typeof(exponentiali!), A::AbstractMatrix, expA::AbstractMatrix, alg::MatrixFunctionViaEigh)
    if !ishermitian(A)
        throw(DomainError(A, "Hermitian matrix was expected. Use `project_hermitian` to project onto the nearest hermitian matrix)"))
    end
    m, n = size(A)
    m == n || throw(DimensionMismatch("square input matrix expected. Got ($m,$n)"))
    return @check_size(expA, (m, m))
end

function check_input(::typeof(exponentiali!), A::AbstractMatrix, expA::AbstractMatrix, ::DiagonalAlgorithm)
    m, n = size(A)
    @assert m == n && isdiag(A)
    @assert expA isa Diagonal
    return @check_size(expA, (m, m))
end

# Outputs
# -------
initialize_output(::typeof(exponential!), A::AbstractMatrix, ::AbstractAlgorithm) = A
initialize_output(::typeof(exponentiali!), τ::Number, A::AbstractMatrix, ::AbstractAlgorithm) =
    complex(A)

# Implementation
# --------------
function exponential!(A, expA, alg::MatrixFunctionViaLA)
    check_input(exponential!, A, expA, alg)
    return LinearAlgebra.exp!(A)
end

function exponential!(A::AbstractMatrix, expA::AbstractMatrix, alg::MatrixFunctionViaEigh)
    check_input(exponential!, A, expA, alg)
    D, V = eigh_full!(A, alg.eigh_alg)
    diagview(D) .= exp.(diagview(D) ./ 2)
    VexpD = rmul!(V, D)
    return mul!(expA, VexpD, V')
end

function exponential!(A::AbstractMatrix, expA::AbstractMatrix, alg::MatrixFunctionViaEig)
    check_input(exponential!, A, expA, alg)
    D, V = eig_full!(A, alg.eig_alg)
    diagview(D) .= exp.(diagview(D))
    iV = inv(V)
    VexpD = rmul!(V, D)
    if eltype(A) <: Real
        expA .= real.(VexpD * iV)
    else
        mul!(expA, VexpD, iV)
    end
    return expA
end

function exponentiali!(τ::Number, A::AbstractMatrix, expA::AbstractMatrix, alg::MatrixFunctionViaLA)
    check_input(exponentiali!, A, expA, alg)
    copyto!(expA, LinearAlgebra.exp(im * τ * A))
    return expA
end

function exponentiali!(τ::Number, A::AbstractMatrix, expA::AbstractMatrix, alg::MatrixFunctionViaEigh)
    check_input(exponentiali!, A, expA, alg)
    D, V = eigh_full!(A, alg.eigh_alg)
    expD = diagonal(exp.(diagview(D) .* (im * τ)))
    if eltype(A) <: Real
        VexpD = V * expD
        return expA .= real.(VexpD * V')
    else
        VexpD = rmul!(V, expD)
        return mul!(expA, VexpD, V')
    end
end

function exponentiali!(τ::Number, A::AbstractMatrix, expA::AbstractMatrix, alg::MatrixFunctionViaEig)
    check_input(exponentiali!, A, expA, alg)
    D, V = eig_full!(A, alg.eig_alg)
    diagview(D) .= exp.(diagview(D) .* (im * τ))
    iV = inv(V)
    VexpD = rmul!(V, D)
    return mul!(expA, VexpD, iV)
end

# Diagonal logic
# --------------
function exponential!(A::Diagonal, expA, alg::DiagonalAlgorithm)
    check_input(exponential!, A, expA, alg)
    diagview(expA) .= exp.(diagview(A))
    return expA
end

function exponentiali!(τ::Number, A::Diagonal, expA, alg::DiagonalAlgorithm)
    check_input(exponentiali!, A, expA, alg)
    diagview(expA) .= exp.(diagview(A) .* (im * τ))
    return expA
end
