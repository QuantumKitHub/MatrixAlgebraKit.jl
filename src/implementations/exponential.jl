# Inputs
# ------
function copy_input(::typeof(exponential), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end

copy_input(::typeof(exponential), A::Diagonal) = copy(A)
copy_input(::typeof(exponential), (τ, A)::Tuple{Number, AbstractMatrix}) = (τ, copy!(similar(A, float(eltype(A))), A))
copy_input(::typeof(exponential), (τ, A)::Tuple{Number, Diagonal}) = τ, copy(A)

function check_input(::typeof(exponential!), A::AbstractMatrix, expA::AbstractMatrix, alg::AbstractAlgorithm)
    m = LinearAlgebra.checksquare(A)
    @check_size(expA, (m, m))
    @check_scalar(expA, A)
    return nothing
end

function check_input(::typeof(exponential!), A::AbstractMatrix, expA::AbstractMatrix, ::DiagonalAlgorithm)
    m = LinearAlgebra.checksquare(A)
    @assert isdiag(A)
    @assert expA isa Diagonal
    @check_size(expA, (m, m))
    @check_scalar(expA, A)
    return nothing
end

function check_input(::typeof(exponential!), (τ, A)::Tuple{Number, AbstractMatrix}, expA::AbstractMatrix, alg::AbstractAlgorithm)
    m = LinearAlgebra.checksquare(A)
    @check_size(expA, (m, m))
    @check_scalar(expA, A, (τ isa Real) ? identity : complex)
    return nothing
end

function check_input(::typeof(exponential!), (τ, A)::Tuple{Number, AbstractMatrix}, expA::AbstractMatrix, ::DiagonalAlgorithm)
    m = LinearAlgebra.checksquare(A)
    @assert isdiag(A)
    @assert expA isa Diagonal
    @check_size(expA, (m, m))
    @check_scalar(expA, A, (τ isa Real) ? identity : complex)
    return nothing
end

# Outputs
# -------
initialize_output(::typeof(exponential!), A::AbstractMatrix, ::AbstractAlgorithm) = A
initialize_output(::typeof(exponential!), (τ, A)::Tuple{T, AbstractMatrix}, ::AbstractAlgorithm) where {T <: Real} = A
initialize_output(::typeof(exponential!), (τ, A)::Tuple{Number, AbstractMatrix}, ::AbstractAlgorithm) = complex(A)

# Implementation
# --------------
function exponential!(A::AbstractMatrix, expA::AbstractMatrix, alg::MatrixFunctionViaLA)
    check_input(exponential!, A, expA, alg)
    A = LinearAlgebra.exp!(A)
    A === expA || copy!(expA, A)
    return expA
end

exponential!(A, expA, alg::MatrixFunctionViaEigh) = exponential!((one(eltype(A)), A), expA, alg)
exponential!(A::AbstractMatrix, expA::AbstractMatrix, alg::MatrixFunctionViaEig) = exponential!((one(eltype(A)), A), expA, alg)

function exponential!((τ, A)::Tuple{Number, AbstractMatrix}, expA::AbstractMatrix, alg::AbstractAlgorithm)
    expA .= A .* τ
    return exponential!(expA, expA, alg)
end

function exponential!((τ, A)::Tuple{Number, AbstractMatrix}, expA::AbstractMatrix, alg::MatrixFunctionViaEigh)
    check_input(exponential!, (τ, A), expA, alg)
    D, V = eigh_full!(A, alg.eigh_alg)
    if eltype(A) <: Real
        if eltype(τ) <: Real
            VexpD = rmul!(V, exponential!((τ / 2, D), D))
        else
            VexpD = V * exponential((τ / 2, D))
        end
        return mul!(expA, VexpD, transpose(VexpD))
    else
        if eltype(τ) <: Real
            VexpD = V * exponential!((τ, D), D)
        else
            VexpD = V * exponential((τ, D))
        end
        return mul!(expA, VexpD, V')
    end
end

function exponential!((τ, A)::Tuple{Number, AbstractMatrix}, expA::AbstractMatrix, alg::MatrixFunctionViaEig)
    check_input(exponential!, (τ, A), expA, alg)
    D, V = eig_full!(A, alg.eig_alg)
    if eltype(A) <: Real && eltype(τ) <: Real
        VexpD = V * exponential!((τ, D), D)
        expAc = rdiv!(VexpD, LinearAlgebra.lu!(V))
        return expA .= real.(expAc)
    else
        expA .= V .* transpose(diagview(exponential!((τ, D), D)))
        return rdiv!(expA, LinearAlgebra.lu!(V))
    end
end

# Diagonal logic
# --------------
function exponential!(A::AbstractMatrix, expA::AbstractMatrix, alg::DiagonalAlgorithm)
    check_input(exponential!, A, expA, alg)
    return map_diagonal!(exp, expA, A)
end

function exponential!((τ, A)::Tuple{Number, AbstractMatrix}, expA::AbstractMatrix, alg::DiagonalAlgorithm)
    check_input(exponential!, (τ, A), expA, alg)
    return map_diagonal!(x -> exp(x * τ), expA, A)
end
