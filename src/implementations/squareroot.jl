# Inputs
# ------
copy_input(::typeof(squareroot), A::AbstractMatrix) = _matrixfunction_copy_input(A)

function check_input(::typeof(squareroot!), A::AbstractMatrix, sqrtA, alg::AbstractAlgorithm)
    return _matrixfunction_check_input(A, sqrtA, alg)
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
    domain_atol = _la_domain_atol(alg, squareroot!)
    # `LinearAlgebra.sqrt` of a real matrix is real whenever the principal square root is
    sqrtAc = LinearAlgebra.sqrt(A)
    if eltype(sqrtAc) <: Complex && !(eltype(sqrtA) <: Complex)
        _la_project_real!(sqrtA, sqrtAc, domain_atol, squareroot!)
    else
        copy!(sqrtA, sqrtAc)
    end
    return sqrtA
end

function squareroot!(A::AbstractMatrix, sqrtA, alg::MatrixFunctionViaEigh)
    check_input(squareroot!, A, sqrtA, alg)
    D, V = eigh_full!(A, alg.eigh_alg)
    diag_alg = DiagonalAlgorithm(; domain_atol = _resolve_domain_atol(diagview(D), alg))
    # `sqrt(A) = (V * D^(1/4)) * (V * D^(1/4))'` is hermitian by construction
    Vs = rmul!(V, power!(D, 1 // 4, D, diag_alg))
    return _mul_herm!(sqrtA, Vs)
end

function squareroot!(A::AbstractMatrix, sqrtA, alg::MatrixFunctionViaEig)
    check_input(squareroot!, A, sqrtA, alg)
    D, V = eig_full!(A, alg.eig_alg)
    λ = diagview(D)
    atol = _resolve_domain_atol(λ, alg)
    # a real result requires the spectrum to stay off the negative real axis
    eltype(A) <: Real && _clamp_domain_eigenvalues!(λ, atol)
    diag_alg = DiagonalAlgorithm(; domain_atol = atol)
    return _apply_eig!(sqrtA, V, squareroot!(D, D, diag_alg))
end

# Diagonal logic
# --------------
function squareroot!(A::AbstractMatrix, sqrtA, alg::DiagonalAlgorithm)
    check_input(squareroot!, A, sqrtA, alg)
    λ = diagview(sqrtA)
    copy!(λ, diagview(A))
    # `sqrt(0) = 0`, so the domain includes its boundary
    eltype(λ) <: Real && _clamp_domain_eigenvalues!(λ, _resolve_domain_atol(λ, alg))
    λ .= sqrt.(λ)
    return sqrtA
end
