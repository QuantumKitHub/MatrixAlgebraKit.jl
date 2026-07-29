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
    domain_atol = _la_domain_atol(alg, power!)
    iszero(p) && return one!(powA)
    isone(p) && ((powA === A || copy!(powA, A)); return powA)
    powAc = A^p
    if eltype(powAc) <: Complex && !(eltype(powA) <: Complex)
        _la_project_real!(powA, powAc, domain_atol, power!)
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
    diag_alg = DiagonalAlgorithm(; domain_atol = _resolve_domain_atol(diagview(D), alg))
    if isinteger(p)
        _apply_eigh!(powA, V, power!(D, p, D, diag_alg))
    else
        # `A^p = (V * D^(p/2)) * (V * D^(p/2))'` is hermitian by construction
        _mul_herm!(powA, rmul!(V, power!(D, p / 2, D, diag_alg)))
    end
    return powA
end

function power!(A::AbstractMatrix, p::Real, powA, alg::MatrixFunctionViaEig)
    check_input(power!, A, p, powA, alg)
    iszero(p) && return one!(powA)
    isone(p) && ((powA === A || copy!(powA, A)); return powA)
    D, V = eig_full!(A, alg.eig_alg)
    λ = diagview(D)
    atol = _resolve_domain_atol(λ, alg)
    # a negative exponent excludes the origin from the domain, whether or not it is an integer
    p < 0 && _check_nonzero_eigenvalues(λ, atol)
    # only a fractional power of a real matrix needs the spectrum off the negative real axis
    if eltype(A) <: Real && !isinteger(p)
        p < 0 ? _check_domain_eigenvalues(λ, atol, false) : _clamp_domain_eigenvalues!(λ, atol)
    end
    diag_alg = DiagonalAlgorithm(; domain_atol = atol)
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
    # a nonnegative integer exponent is defined for every square matrix and needs no tolerance
    if p < 0 || !isinteger(p)
        atol = _resolve_domain_atol(λ, alg)
        # a negative exponent excludes the origin from the domain, whether or not it is an integer
        p < 0 && _check_nonzero_eigenvalues(λ, atol)
        if eltype(λ) <: Real && !isinteger(p)
            p < 0 ? _check_domain_eigenvalues(λ, atol, false) : _clamp_domain_eigenvalues!(λ, atol)
        end
    end
    λ .= λ .^ p
    return powA
end
