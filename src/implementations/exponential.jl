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
function initialize_output(::typeof(exponential!), A::AbstractMatrix, ::AbstractAlgorithm)
    n = size(A, 1) # square check will happen later
    expA = similar(A, (n, n))
    return expA
end

initialize_output(::typeof(exponential!), A::Diagonal, ::DiagonalAlgorithm) = A

function initialize_output(::typeof(exponentiali!), τ::Number, A::AbstractMatrix, ::AbstractAlgorithm)
    n = size(A, 1) # square check will happen later
    expA = similar(complex(A), (n, n))
    return expA
end

initialize_output(::typeof(exponentiali!), τ::Number, A::Diagonal, ::DiagonalAlgorithm) = complex(A)

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

function exponentiali!(τ::Number, A::AbstractMatrix{T1}, expA::AbstractMatrix{T2}, alg::MatrixFunctionViaLA) where {T1 <: BlasFloat, T2 <: BlasFloat}
    check_input(exponentiali!, A, expA, alg)
    copyto!(expA, LinearAlgebra.exp(im * τ * A))
    return expA
end

function exponentiali!(τ::Number, A::AbstractMatrix, expA::AbstractMatrix, alg::MatrixFunctionViaEigh)
    check_input(exponentiali!, A, expA, alg)
    Dreal, Vreal = eigh_full(A, alg.eigh_alg)

    Dcomplex = complex(Dreal)
    Vcomplex = complex(Vreal)

    iV = copy(adjoint(Vcomplex))

    diagview(Dcomplex) .= exp.(diagview(Dcomplex) .* (im * τ))
    rmul!(Vcomplex, Dcomplex)
    mul!(expA, Vcomplex, iV)
    return expA
end

function exponentiali!(τ::T, A::AbstractMatrix{T}, expA::AbstractMatrix{T}, alg::MatrixFunctionViaEig) where {T <: Real}
    check_input(exponentiali!, A, expA, alg)
    D, V = eig_full(A, alg.eig_alg)
    iV = inv(V)
    map!(exp, diagview(D), diagview(D) .* (im * τ))
    expA .= real.(rmul!(V, D) * iV)
    return expA
end

function exponentiali!(τ::Number, A::AbstractMatrix, expA::AbstractMatrix, alg::MatrixFunctionViaEig)
    check_input(exponentiali!, A, expA, alg)
    D, V = eig_full(A, alg.eig_alg)
    iV = inv(V)
    map!(exp, diagview(D), diagview(D) .* (im * τ))
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

function exponentiali!(τ::Number, A::Diagonal, expA, alg::DiagonalAlgorithm)
    check_input(exponentiali!, A, expA, alg)
    map!(exp, diagview(expA), diagview(A) .* (im * τ))
    return expA
end
