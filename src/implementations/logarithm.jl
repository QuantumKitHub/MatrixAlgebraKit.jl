# Inputs
# ------
copy_input(::typeof(logarithm), A::AbstractMatrix) = _matrixfunction_copy_input(A)

function check_input(::typeof(logarithm!), A::AbstractMatrix, logA, alg::AbstractAlgorithm)
    return _matrixfunction_check_input(A, logA, alg)
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
    domain_atol = _la_domain_atol(alg, logarithm!)
    # `LinearAlgebra.log` of a real matrix is real whenever the principal logarithm is. Note that a
    # (numerically) zero eigenvalue goes undetected here, as documented in the manual.
    logAc = LinearAlgebra.log(A)
    if eltype(logAc) <: Complex && !(eltype(logA) <: Complex)
        _la_project_real!(logA, logAc, domain_atol, logarithm!)
    else
        copy!(logA, logAc)
    end
    return logA
end

function logarithm!(A::AbstractMatrix, logA, alg::MatrixFunctionViaEigh)
    check_input(logarithm!, A, logA, alg)
    D, V = eigh_full!(A, alg.eigh_alg)
    diag_alg = DiagonalAlgorithm(; domain_atol = _resolve_domain_atol(diagview(D), alg))
    return _apply_eigh!(logA, V, logarithm!(D, D, diag_alg))
end

function logarithm!(A::AbstractMatrix, logA, alg::MatrixFunctionViaEig)
    check_input(logarithm!, A, logA, alg)
    D, V = eig_full!(A, alg.eig_alg)
    λ = diagview(D)
    atol = _resolve_domain_atol(λ, alg)
    _check_nonzero_eigenvalues(λ, atol)
    # a real result requires the spectrum to stay off the negative real axis
    eltype(A) <: Real && _check_domain_eigenvalues(λ, atol, false)
    diag_alg = DiagonalAlgorithm(; domain_atol = atol)
    return _apply_eig!(logA, V, logarithm!(D, D, diag_alg))
end

# Diagonal logic
# --------------
function logarithm!(A::AbstractMatrix, logA, alg::DiagonalAlgorithm)
    check_input(logarithm!, A, logA, alg)
    λ = diagview(logA)
    copy!(λ, diagview(A))
    atol = _resolve_domain_atol(λ, alg)
    # `log(0)` does not exist, so the origin is excluded from the domain and nothing is clamped
    _check_nonzero_eigenvalues(λ, atol)
    eltype(λ) <: Real && _check_domain_eigenvalues(λ, atol, false)
    λ .= log.(λ)
    return logA
end
