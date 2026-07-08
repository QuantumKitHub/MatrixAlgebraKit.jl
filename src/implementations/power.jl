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
    powAc = A^p
    if eltype(powAc) <: Complex && !(eltype(powA) <: Complex)
        # `LinearAlgebra` computes fractional powers of real matrices in complex
        # arithmetic and only casts back to real when the result is exactly real,
        # so rounding-level imaginary components do not signal a domain violation.
        # The tolerance is based on the working precision, which may be lower than
        # the result eltype suggests (e.g. `Float32` input promotes to `ComplexF64`).
        atol = defaulttol(powA) * norm(powAc, Inf)
        all(x -> abs(imag(x)) <= atol, powAc) || throw(_realness_domainerror(power!))
        powA .= real.(powAc)
        return powA
    end
    copy!(powA, powAc)
    return powA
end

function power!(A::AbstractMatrix, p::Real, powA, alg::MatrixFunctionViaEigh)
    check_input(power!, A, p, powA, alg)
    D, V = eigh_full!(A, alg.eigh_alg)
    diag_alg = DiagonalAlgorithm(; domain_atol = alg.domain_atol)
    if isinteger(p)
        VD = V * power!(D, p, D, diag_alg)
        mul!(powA, VD, V')
        return project_hermitian!(powA)
    else
        # `A^p = (V * D^(p/2)) * (V * D^(p/2))'` is hermitian by construction
        Vs = rmul!(V, power!(D, p / 2, D, diag_alg))
        return _mul_herm!(powA, Vs)
    end
end

function power!(A::AbstractMatrix, p::Real, powA, alg::MatrixFunctionViaEig)
    check_input(power!, A, p, powA, alg)
    D, V = eig_full!(A, alg.eig_alg)
    diag_alg = DiagonalAlgorithm(; domain_atol = alg.domain_atol)
    if eltype(A) <: Real
        isinteger(p) || _clamp_domain_eigenvalues!(D, alg.domain_atol)
        VpD = V * power!(D, p, D, diag_alg)
        powAc = rdiv!(VpD, LinearAlgebra.lu!(V))
        return powA .= real.(powAc)
    else
        powA .= V .* transpose(diagview(power!(D, p, D, diag_alg)))
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
