# Inputs
# ------
function copy_input(::typeof(squareroot), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
copy_input(::typeof(squareroot), A::Diagonal) = Diagonal(float.(diagview(A)))

function check_input(::typeof(squareroot!), A::AbstractMatrix, sqrtA, alg::AbstractAlgorithm)
    m = LinearAlgebra.checksquare(A)
    @check_size(sqrtA, (m, m))
    @check_scalar(sqrtA, A)
    return nothing
end

function check_input(::typeof(squareroot!), A::AbstractMatrix, sqrtA, ::DiagonalAlgorithm)
    m = LinearAlgebra.checksquare(A)
    @assert isdiag(A)
    @assert sqrtA isa Diagonal
    @check_size(sqrtA, (m, m))
    @check_scalar(sqrtA, A)
    return nothing
end

# Algorithm selection
# -------------------
squareroot!(A::AbstractMatrix, alg::DefaultAlgorithm) = squareroot!(A, select_algorithm(squareroot!, A, nothing; alg.kwargs...))
squareroot!(A::AbstractMatrix, out, alg::DefaultAlgorithm) = squareroot!(A, out, select_algorithm(squareroot!, A, nothing; alg.kwargs...))

# Outputs
# -------
initialize_output(::typeof(squareroot!), A::AbstractMatrix, ::AbstractAlgorithm) = A

# Implementation
# --------------
function squareroot!(A::AbstractMatrix, sqrtA, alg::MatrixFunctionViaLA)
    check_input(squareroot!, A, sqrtA, alg)
    isempty(alg.kwargs) || throw(ArgumentError("`MatrixFunctionViaLA` does not accept keyword arguments for `squareroot`"))
    # `LinearAlgebra.sqrt` of a real matrix is real whenever the principal square root is,
    # so a complex result with a real output signals a genuine domain violation
    sqrtAc = LinearAlgebra.sqrt(A)
    if eltype(sqrtAc) <: Complex && !(eltype(sqrtA) <: Complex)
        throw(_realness_domainerror(squareroot!))
    end
    copy!(sqrtA, sqrtAc)
    return sqrtA
end

function squareroot!(A::AbstractMatrix, sqrtA, alg::MatrixFunctionViaEigh)
    check_input(squareroot!, A, sqrtA, alg)
    D, V = eigh_full!(A, alg.eigh_alg)
    diag_alg = DiagonalAlgorithm(; domain_atol = alg.domain_atol)
    # `sqrt(A) = (V * D^(1/4)) * (V * D^(1/4))'` is hermitian by construction
    sqrtD = squareroot!(D, D, diag_alg)
    Vs = rmul!(V, squareroot!(sqrtD, sqrtD, diag_alg))
    return _mul_herm!(sqrtA, Vs)
end

function squareroot!(A::AbstractMatrix, sqrtA, alg::MatrixFunctionViaEig)
    check_input(squareroot!, A, sqrtA, alg)
    D, V = eig_full!(A, alg.eig_alg)
    diag_alg = DiagonalAlgorithm(; domain_atol = alg.domain_atol)
    if eltype(A) <: Real
        atol = something(alg.domain_atol, default_domain_atol(diagview(D)))
        _clamp_domain_eigenvalues!(diagview(D), atol)
        VsqrtD = V * squareroot!(D, D, diag_alg)
        sqrtAc = rdiv!(VsqrtD, LinearAlgebra.lu!(V))
        return sqrtA .= real.(sqrtAc)
    else
        sqrtA .= V .* transpose(diagview(squareroot!(D, D, diag_alg)))
        return rdiv!(sqrtA, LinearAlgebra.lu!(V))
    end
end

# Diagonal logic
# --------------
function squareroot!(A::AbstractMatrix, sqrtA, alg::DiagonalAlgorithm)
    check_input(squareroot!, A, sqrtA, alg)
    λ = diagview(sqrtA)
    copyto!(λ, diagview(A))
    if eltype(λ) <: Real
        atol = something(get(alg.kwargs, :domain_atol, nothing), default_domain_atol(λ))
        _clamp_domain_eigenvalues!(λ, atol)
    end
    λ .= sqrt.(λ)
    return sqrtA
end
