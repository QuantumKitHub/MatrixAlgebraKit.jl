# Inputs
# ------
copy_input(::typeof(power), A::AbstractMatrix, p::Real) = _matrixfunction_copy_input(A), p

function check_input(::typeof(power!), A::AbstractMatrix, p::Real, powA, alg::AbstractAlgorithm)
    return _matrixfunction_check_input(A, powA, alg)
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
    iszero(p) && return one!(powA)
    isone(p) && ((powA === A || copy!(powA, A)); return powA)
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
    else
        copy!(powA, powAc)
    end
    return powA
end

function power!(A::AbstractMatrix, p::Real, powA, alg::MatrixFunctionViaEigh)
    check_input(power!, A, p, powA, alg)
    iszero(p) && return one!(powA)
    isone(p) && ((powA === A || copy!(powA, A)); return powA)
    D, V = eigh_full!(A, alg.eigh_alg)
    diag_alg = DiagonalAlgorithm(; domain_atol = alg.domain_atol)
    isinteger(p) && return _apply_eigh!(powA, V, power!(D, p, D, diag_alg))
    # `A^p = (V * D^(p/2)) * (V * D^(p/2))'` is hermitian by construction
    return _mul_herm!(powA, rmul!(V, power!(D, p / 2, D, diag_alg)))
end

function power!(A::AbstractMatrix, p::Real, powA, alg::MatrixFunctionViaEig)
    check_input(power!, A, p, powA, alg)
    iszero(p) && return one!(powA)
    isone(p) && ((powA === A || copy!(powA, A)); return powA)
    D, V = eig_full!(A, alg.eig_alg)
    # only a fractional power of a real matrix needs the spectrum off the negative real axis
    eltype(A) <: Real && !isinteger(p) && _clamp_domain_eigenvalues!(D, alg.domain_atol)
    diag_alg = DiagonalAlgorithm(; domain_atol = alg.domain_atol)
    return _apply_eig!(powA, V, power!(D, p, D, diag_alg))
end

# Diagonal logic
# --------------
function power!(A::AbstractMatrix, p::Real, powA, alg::DiagonalAlgorithm)
    check_input(power!, A, p, powA, alg)
    iszero(p) && return one!(powA)
    isone(p) && ((powA === A || copy!(powA, A)); return powA)
    λ = diagview(powA)
    copy!(λ, diagview(A))
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
