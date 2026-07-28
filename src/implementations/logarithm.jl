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
    isempty(alg.kwargs) || throw(ArgumentError("`MatrixFunctionViaLA` does not accept keyword arguments for `logarithm`"))
    # `LinearAlgebra.log` of a real matrix is real whenever the principal logarithm is,
    # so a complex result with a real output signals a genuine domain violation
    logAc = LinearAlgebra.log(A)
    if eltype(logAc) <: Complex && !(eltype(logA) <: Complex)
        throw(_realness_domainerror(logarithm!))
    end
    copy!(logA, logAc)
    return logA
end

function logarithm!(A::AbstractMatrix, logA, alg::MatrixFunctionViaEigh)
    check_input(logarithm!, A, logA, alg)
    D, V = eigh_full!(A, alg.eigh_alg)
    diag_alg = DiagonalAlgorithm(; domain_atol = alg.domain_atol)
    return _apply_eigh!(logA, V, logarithm!(D, D, diag_alg))
end

function logarithm!(A::AbstractMatrix, logA, alg::MatrixFunctionViaEig)
    check_input(logarithm!, A, logA, alg)
    D, V = eig_full!(A, alg.eig_alg)
    # a real result requires the spectrum to stay off the negative real axis
    eltype(A) <: Real && _clamp_domain_eigenvalues!(D, alg.domain_atol)
    diag_alg = DiagonalAlgorithm(; domain_atol = alg.domain_atol)
    return _apply_eig!(logA, V, logarithm!(D, D, diag_alg))
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
