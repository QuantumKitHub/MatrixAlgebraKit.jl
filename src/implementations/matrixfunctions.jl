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

# Clamp real eigenvalues that are negative within `atol` (rounding artifacts) to zero,
# and throw a `DomainError` for eigenvalues that are genuinely negative, since then the
# result cannot be expressed with the same (real) scalar type.
function _clamp_domain_eigenvalues!(λ::AbstractVector{<:Real}, atol::Real)
    λmin = minimum(λ; init = zero(eltype(λ)))
    λmin < -atol && throw_negative_eigenvalue(λmin, atol, "a negative real eigenvalue")
    λ .= max.(λ, zero(eltype(λ)))
    return λ
end

# Convenience method for the eigenvalues of a decomposition, deriving the default
# tolerance from the eigenvalues themselves when `domain_atol` is `nothing`.
function _clamp_domain_eigenvalues!(D::Diagonal, domain_atol::Union{Nothing, Real})
    λ = diagview(D)
    atol = something(domain_atol, default_domain_atol(λ))
    return _clamp_domain_eigenvalues!(λ, atol)
end

# Complex eigenvalues of a real matrix: only eigenvalues (numerically) on the negative
# real axis obstruct a real result; complex-conjugate pairs do not.
function _clamp_domain_eigenvalues!(λ::AbstractVector{<:Complex}, atol::Real)
    onaxis = x -> abs(imag(x)) <= atol && real(x) < 0
    λmin = mapreduce(x -> onaxis(x) ? real(x) : zero(real(x)), min, λ; init = zero(real(eltype(λ))))
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
