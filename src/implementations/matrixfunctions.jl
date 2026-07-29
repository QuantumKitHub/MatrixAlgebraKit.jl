# Shared input and output handling
# --------------------------------
# `squareroot`, `logarithm` and `power` all map a square matrix to a matrix of the same size and
# scalar type, so they agree on `copy_input`, `check_input` and `initialize_output`; only their
# kernels differ. Each implementation file forwards to the helpers below, which keeps the
# per-function definitions to the one line that names the function.

_matrixfunction_copy_input(A::AbstractMatrix) = copy!(similar(A, float(eltype(A))), A)
_matrixfunction_copy_input(A::Diagonal) = map_diagonal(float, A)

function _matrixfunction_check_input(A::AbstractMatrix, out, ::AbstractAlgorithm)
    m = LinearAlgebra.checksquare(A)
    @check_size(out, (m, m))
    @check_scalar(out, A)
    return nothing
end

function _matrixfunction_check_input(A::AbstractMatrix, out, ::DiagonalAlgorithm)
    m = LinearAlgebra.checksquare(A)
    @assert isdiag(A)
    @assert out isa Diagonal
    @check_size(out, (m, m))
    @check_scalar(out, A)
    return nothing
end

# Shared reconstruction from an eigenvalue decomposition
# ------------------------------------------------------
# Both take the already-transformed eigenvalues `fD = f(D)` and rebuild `f(A)`.

# `f(A) = V f(D) V⁻¹` for a general `A`. A real matrix has a complex decomposition but a real
# `f(A)` whenever `f(D)` closes under conjugation, so the imaginary part is dropped after the
# solve; the callers are responsible for rejecting the inputs where it would not be.
function _apply_eig!(fA, V, fD)
    if eltype(fA) <: Real
        VfD = V * fD
        fAc = rdiv!(VfD, LinearAlgebra.lu!(V))
        return fA .= real.(fAc)
    else
        fA .= V .* transpose(diagview(fD))
        return rdiv!(fA, LinearAlgebra.lu!(V))
    end
end

# `f(A) = V f(D) V'` for a hermitian `A`. The product is hermitian only up to roundoff, so it is
# projected afterwards; where `f` admits a square root, prefer building `f(A)` as a symmetric
# product via `_mul_herm!`, which is hermitian by construction.
_apply_eigh!(fA, V, fD) = project_hermitian!(mul!(fA, V * fD, V'))

# Shared helpers for matrix functions with a restricted domain
# -------------------------------------------------------------

# The throwing branches live in `@noinline` helpers so that the reductions and broadcasts
# below stay free of error-path code, which keeps them GPU friendly.
@noinline function throw_negative_eigenvalue(λmin, atol, what)
    return throw(
        DomainError(
            λmin,
            "The matrix has $what beyond `domain_atol = $atol` and the result of this matrix function is complex. " *
                "Pass a complex matrix to obtain the principal value, or increase `domain_atol` if the eigenvalue is a rounding artifact."
        )
    )
end

@noinline function throw_zero_eigenvalue(amin, atol)
    return throw(
        DomainError(
            amin,
            "The matrix has a (numerically) zero eigenvalue within `domain_atol = $atol`, for which this matrix function is not defined."
        )
    )
end

_clamp_domain_eigenvalues!(D::Diagonal, atol::Real) =
    _clamp_domain_eigenvalues!(diagview(D), atol)

# Clamp real eigenvalues that are negative within `atol` (rounding artifacts) to zero,
# and throw a `DomainError` for eigenvalues that are genuinely negative, since then the
# result cannot be expressed with the same (real) scalar type.
function _clamp_domain_eigenvalues!(λ::AbstractVector{<:Real}, atol::Real)
    λmin = minimum(λ; init = zero(eltype(λ)))
    atol = atol < 0 ? default_domain_atol(λ) : oftype(λmin, atol)
    λmin < -atol && throw_negative_eigenvalue(λmin, atol, "a negative real eigenvalue")
    λ .= max.(λ, zero(eltype(λ)))
    return λ
end

# Complex eigenvalues of a real matrix: only eigenvalues (numerically) on the negative
# real axis obstruct a real result; complex-conjugate pairs do not.
function _clamp_domain_eigenvalues!(λ::AbstractVector{<:Complex}, atol::Real)
    onaxis = x -> abs(imag(x)) <= atol && real(x) < 0
    λmin = mapreduce(x -> onaxis(x) ? real(x) : zero(real(x)), min, λ; init = zero(real(eltype(λ))))
    atol = atol < 0 ? default_domain_atol(λ) : oftype(λmin, atol)
    λmin < -atol && throw_negative_eigenvalue(λmin, atol, "an eigenvalue on the negative real axis")
    λ .= ifelse.(onaxis.(λ), zero(eltype(λ)), λ)
    return λ
end

# Reject (numerically) zero eigenvalues for functions that are undefined there,
# e.g. `logarithm` and `power` with a negative fractional power.
function _check_nonzero_eigenvalues(λ, atol::Real)
    amin = minimum(abs, λ; init = typemax(real(eltype(λ))))
    amin <= atol && throw_zero_eigenvalue(amin, atol)
    return λ
end

# For `MatrixFunctionViaLA`, domain violations surface as a complex result from
# `LinearAlgebra` while the output should remain real.
function _realness_domainerror(f)
    return DomainError(
        f,
        "The result of this matrix function applied to the given real matrix is complex (eigenvalues on the negative real axis). " *
            "Pass a complex matrix to obtain the principal value, or use `MatrixFunctionViaEigh`/`MatrixFunctionViaEig` with a suitable " *
            "`domain_atol` if the offending eigenvalues are rounding artifacts."
    )
end
