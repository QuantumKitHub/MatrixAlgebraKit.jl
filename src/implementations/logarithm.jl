# Inputs
# ------
function copy_input(::typeof(logarithm), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
copy_input(::typeof(logarithm), A::Diagonal) = Diagonal(float.(diagview(A)))

function check_input(::typeof(logarithm!), A::AbstractMatrix, logA, alg::AbstractAlgorithm)
    m = LinearAlgebra.checksquare(A)
    @check_size(logA, (m, m))
    @check_scalar(logA, A)
    return nothing
end

function check_input(::typeof(logarithm!), A::AbstractMatrix, logA, ::DiagonalAlgorithm)
    m = LinearAlgebra.checksquare(A)
    @assert isdiag(A)
    @assert logA isa Diagonal
    @check_size(logA, (m, m))
    @check_scalar(logA, A)
    return nothing
end

# Algorithm selection
# -------------------
logarithm!(A::AbstractMatrix, alg::DefaultAlgorithm) = logarithm!(A, select_algorithm(logarithm!, A, nothing; alg.kwargs...))
logarithm!(A::AbstractMatrix, out, alg::DefaultAlgorithm) = logarithm!(A, out, select_algorithm(logarithm!, A, nothing; alg.kwargs...))

# Outputs
# -------
initialize_output(::typeof(logarithm!), A::AbstractMatrix, ::AbstractAlgorithm) = A

# Implementation
# --------------
function logarithm!(A::AbstractMatrix, logA, alg::MatrixFunctionViaLA)
    check_input(logarithm!, A, logA, alg)
    isempty(alg.kwargs) || throw(ArgumentError("`MatrixFunctionViaLA` does not accept keyword arguments for `logarithm`"))
    result = LinearAlgebra.log(A)
    _copy_result!(logarithm!, logA, result)
    return logA
end

function logarithm!(A::AbstractMatrix, logA, alg::MatrixFunctionViaEigh)
    check_input(logarithm!, A, logA, alg)
    D, V = eigh_full!(A, alg.eigh_alg)
    λ = diagview(D)
    atol = something(alg.domain_atol, default_domain_atol(λ))
    _check_nonzero_eigenvalues(λ, atol)
    _clamp_domain_eigenvalues!(λ, atol)
    λ .= log.(λ)
    VD = V * D
    mul!(logA, VD, V')
    return project_hermitian!(logA)
end

function logarithm!(A::AbstractMatrix, logA, alg::MatrixFunctionViaEig)
    check_input(logarithm!, A, logA, alg)
    D, V = eig_full!(A, alg.eig_alg)
    λ = diagview(D)
    atol = something(alg.domain_atol, default_domain_atol(λ))
    _check_nonzero_eigenvalues(λ, atol)
    if eltype(A) <: Real
        _clamp_domain_eigenvalues!(λ, atol)
        λ .= log.(λ)
        VD = V * D
        logAc = rdiv!(VD, LinearAlgebra.lu!(V))
        return logA .= real.(logAc)
    else
        λ .= log.(λ)
        logA .= V .* transpose(λ)
        return rdiv!(logA, LinearAlgebra.lu!(V))
    end
end

# Diagonal logic
# --------------
function logarithm!(A::AbstractMatrix, logA, alg::DiagonalAlgorithm)
    check_input(logarithm!, A, logA, alg)
    λ = diagview(logA)
    copyto!(λ, diagview(A))
    atol = something(get(alg.kwargs, :domain_atol, nothing), default_domain_atol(λ))
    _check_nonzero_eigenvalues(λ, atol)
    if eltype(λ) <: Real
        _clamp_domain_eigenvalues!(λ, atol)
    end
    λ .= log.(λ)
    return logA
end
