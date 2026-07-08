# Inputs
# ------
function copy_input(::typeof(power), A::AbstractMatrix, p::Real)
    return copy!(similar(A, float(eltype(A))), A), p
end
copy_input(::typeof(power), A::Diagonal, p::Real) = Diagonal(float.(diagview(A))), p

function check_input(::typeof(power!), A::AbstractMatrix, p::Real, powA, alg::AbstractAlgorithm)
    m = LinearAlgebra.checksquare(A)
    @check_size(powA, (m, m))
    @check_scalar(powA, A)
    return nothing
end

function check_input(::typeof(power!), A::AbstractMatrix, p::Real, powA, ::DiagonalAlgorithm)
    m = LinearAlgebra.checksquare(A)
    @assert isdiag(A)
    @assert powA isa Diagonal
    @check_size(powA, (m, m))
    @check_scalar(powA, A)
    return nothing
end

# Algorithm selection
# -------------------
power!(A::AbstractMatrix, p::Real, alg::DefaultAlgorithm) = power!(A, p, select_algorithm(power!, (A, p), nothing; alg.kwargs...))
power!(A::AbstractMatrix, p::Real, out, alg::DefaultAlgorithm) = power!(A, p, out, select_algorithm(power!, (A, p), nothing; alg.kwargs...))

# Outputs
# -------
initialize_output(::typeof(power!), A::AbstractMatrix, p::Real, ::AbstractAlgorithm) = A

# Implementation
# --------------
function power!(A::AbstractMatrix, p::Real, powA, alg::MatrixFunctionViaLA)
    check_input(power!, A, p, powA, alg)
    isempty(alg.kwargs) || throw(ArgumentError("`MatrixFunctionViaLA` does not accept keyword arguments for `power`"))
    result = A^p
    _copy_result!(power!, powA, result)
    return powA
end

function power!(A::AbstractMatrix, p::Real, powA, alg::MatrixFunctionViaEigh)
    check_input(power!, A, p, powA, alg)
    D, V = eigh_full!(A, alg.eigh_alg)
    λ = diagview(D)
    if isinteger(p)
        p < 0 && any(iszero, λ) && throw(LinearAlgebra.SingularException(0))
        λ .= λ .^ p
        VD = V * D
        mul!(powA, VD, V')
    else
        atol = something(alg.domain_atol, default_domain_atol(λ))
        p < 0 && _check_nonzero_eigenvalues(λ, atol)
        _clamp_domain_eigenvalues!(λ, atol)
        # `A^p = (V * D^(p/2)) * (V * D^(p/2))'` is hermitian by construction
        λ .= λ .^ (p / 2)
        Vs = rmul!(V, D)
        mul!(powA, Vs, Vs')
    end
    return project_hermitian!(powA)
end

function power!(A::AbstractMatrix, p::Real, powA, alg::MatrixFunctionViaEig)
    check_input(power!, A, p, powA, alg)
    D, V = eig_full!(A, alg.eig_alg)
    λ = diagview(D)
    if isinteger(p)
        p < 0 && any(iszero, λ) && throw(LinearAlgebra.SingularException(0))
    else
        atol = something(alg.domain_atol, default_domain_atol(λ))
        p < 0 && _check_nonzero_eigenvalues(λ, atol)
        eltype(A) <: Real && _clamp_domain_eigenvalues!(λ, atol)
    end
    if eltype(A) <: Real
        λ .= λ .^ p
        VD = V * D
        powAc = rdiv!(VD, LinearAlgebra.lu!(V))
        return powA .= real.(powAc)
    else
        λ .= λ .^ p
        powA .= V .* transpose(λ)
        return rdiv!(powA, LinearAlgebra.lu!(V))
    end
end

# Diagonal logic
# --------------
function power!(A::AbstractMatrix, p::Real, powA, alg::DiagonalAlgorithm)
    check_input(power!, A, p, powA, alg)
    λ = diagview(powA)
    copyto!(λ, diagview(A))
    if isinteger(p)
        p < 0 && any(iszero, λ) && throw(LinearAlgebra.SingularException(0))
    else
        atol = something(get(alg.kwargs, :domain_atol, nothing), default_domain_atol(λ))
        p < 0 && _check_nonzero_eigenvalues(λ, atol)
        eltype(λ) <: Real && _clamp_domain_eigenvalues!(λ, atol)
    end
    λ .= λ .^ p
    return powA
end
