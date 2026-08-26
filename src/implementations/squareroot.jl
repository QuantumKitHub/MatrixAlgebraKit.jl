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
    _check_la_kwargs(alg, squareroot!)
    # `LinearAlgebra.sqrt` of a real matrix is real whenever the principal square root is
    sqrtAc = LinearAlgebra.sqrt(A)
    if eltype(sqrtAc) <: Complex && !(eltype(sqrtA) <: Complex)
        all(isfinite, sqrtAc) || throw_nonfinite_result(squareroot!)
        throw_complex_result(squareroot!)
    end
    copy!(sqrtA, sqrtAc)
    return sqrtA
end

function squareroot!(A::AbstractMatrix, sqrtA, alg::MatrixFunctionViaEigh)
    check_input(squareroot!, A, sqrtA, alg)
    D, V = eigh_full!(A, select_algorithm(eigh_full!, A, _eigh_alg(alg)))
    λ = diagview(D)
    _clamp_domain_eigenvalues!(λ, _resolve_domain_atol(λ, alg))
    λ .= sqrt.(λ)
    return _apply_eigh!(sqrtA, V, D)
end

function squareroot!(A::AbstractMatrix, sqrtA, alg::MatrixFunctionViaEig)
    check_input(squareroot!, A, sqrtA, alg)
    D, V = eig_full!(A, select_algorithm(eig_full!, A, _eig_alg(alg)))
    λ = diagview(D)
    atol = _resolve_domain_atol(λ, alg)
    # a real result requires the spectrum to stay off the negative real axis; whether an eigenvalue
    # sits *on* that axis keeps the algorithm default however `domain_atol` was set
    eltype(A) <: Real && _clamp_domain_eigenvalues!(λ, atol, _axis_atol(λ))
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
