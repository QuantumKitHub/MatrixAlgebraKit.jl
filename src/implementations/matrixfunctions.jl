# Shared helpers for matrix functions with a restricted domain
# -------------------------------------------------------------

# Clamp real eigenvalues that are negative within `atol` (rounding artifacts) to zero,
# and throw a `DomainError` for eigenvalues that are genuinely negative, since then the
# result cannot be expressed with the same (real) scalar type.
function _clamp_domain_eigenvalues!(λ::AbstractVector{<:Real}, atol::Real)
    for i in eachindex(λ)
        x = λ[i]
        if x < -atol
            throw(
                DomainError(
                    x,
                    "The matrix has a negative real eigenvalue beyond `domain_atol = $atol` and the result of this matrix function is complex. " *
                        "Pass a complex matrix to obtain the principal value, or increase `domain_atol` if the eigenvalue is a rounding artifact."
                )
            )
        elseif x < 0
            λ[i] = zero(x)
        end
    end
    return λ
end

# Complex eigenvalues of a real matrix: only eigenvalues (numerically) on the negative
# real axis obstruct a real result; complex-conjugate pairs do not.
function _clamp_domain_eigenvalues!(λ::AbstractVector{<:Complex}, atol::Real)
    for i in eachindex(λ)
        x = λ[i]
        if abs(imag(x)) <= atol && real(x) < 0
            if real(x) < -atol
                throw(
                    DomainError(
                        x,
                        "The matrix has an eigenvalue on the negative real axis beyond `domain_atol = $atol` and the result of this matrix function is complex. " *
                            "Pass a complex matrix to obtain the principal value, or increase `domain_atol` if the eigenvalue is a rounding artifact."
                    )
                )
            else
                λ[i] = zero(x)
            end
        end
    end
    return λ
end

# Reject (numerically) zero eigenvalues for functions that are undefined there,
# e.g. `logarithm` and `power` with a negative fractional power.
function _check_nonzero_eigenvalues(λ, atol::Real)
    for x in λ
        if abs(x) <= atol
            throw(
                DomainError(
                    x,
                    "The matrix has a (numerically) zero eigenvalue within `domain_atol = $atol`, for which this matrix function is not defined."
                )
            )
        end
    end
    return λ
end

# For `MatrixFunctionViaLA`, domain violations surface as a complex result from
# `LinearAlgebra`; detect this after the fact to preserve type-stable output.
# `LinearAlgebra` sometimes computes a real result in complex arithmetic (e.g. fractional
# powers via a complex Schur decomposition), so rounding-level imaginary components do
# not signal a domain violation and are simply discarded.
function _copy_result!(f, out::AbstractMatrix, result::AbstractMatrix)
    if eltype(result) <: Complex && !(eltype(out) <: Complex)
        # base the tolerance on the working precision, which is the lower of the two:
        # `LinearAlgebra` may internally promote, e.g. `Float32` fractional powers are
        # computed with `Float64`-eltype results but `Float32`-level accuracy
        atol = max(defaulttol(out), defaulttol(result)) * norm(result, Inf)
        for x in result
            if abs(imag(x)) > atol
                throw(
                    DomainError(
                        f,
                        "The result of this matrix function applied to the given real matrix is complex (eigenvalues on the negative real axis). " *
                            "Pass a complex matrix to obtain the principal value, or use `MatrixFunctionViaEigh`/`MatrixFunctionViaEig` with a suitable " *
                            "`domain_atol` if the offending eigenvalues are rounding artifacts."
                    )
                )
            end
        end
        out .= real.(result)
    else
        copy!(out, result)
    end
    return out
end
