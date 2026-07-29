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
# `_clamp_domain_eigenvalues!` is for the functions whose domain includes its boundary and
# `_check_domain_eigenvalues` for the ones that exclude it; see the manual section on domain
# considerations for what that means for `domain_atol`.

# Each algorithm obtains its eigenvalues with a different accuracy, so each brings its own default.
default_domain_atol(λ, ::DiagonalAlgorithm) = _roundoff_domain_atol(λ)
default_domain_atol(λ, ::MatrixFunctionViaEigh) = _roundoff_domain_atol(λ)
default_domain_atol(λ, ::MatrixFunctionViaEig) = _conditioning_domain_atol(λ)

# A negative tolerance denotes the runtime default, which keeps the algorithm types concrete.
_domain_atol(alg::Union{MatrixFunctionViaEig, MatrixFunctionViaEigh}) = alg.domain_atol
_domain_atol(alg::DiagonalAlgorithm) = get(alg.kwargs, :domain_atol, -1.0)

# Callers resolve the default before delegating to an inner `DiagonalAlgorithm`, so that the
# tolerance follows the algorithm that computed the eigenvalues.
function _resolve_domain_atol(λ, alg)
    atol = _domain_atol(alg)
    R = real(float(eltype(λ)))
    return atol < 0 ? convert(R, default_domain_atol(λ, alg)) : convert(R, atol)
end

# The throwing branches live in `@noinline` helpers so that the reductions and broadcasts
# below stay free of error-path code, which keeps them GPU friendly.
@noinline function throw_negative_eigenvalue(λmin, atol, what, clampable)
    advice = if clampable
        "Pass a complex matrix to obtain the principal value, or increase `domain_atol` if the eigenvalue is a rounding artifact."
    else
        "Pass a complex matrix to obtain the principal value; increasing `domain_atol` cannot help, as this matrix function is " *
            "undefined on the domain boundary itself."
    end
    return throw(
        DomainError(
            λmin,
            "The matrix has $what beyond `domain_atol = $atol` and the result of this matrix function is complex. " * advice
        )
    )
end

@noinline function throw_zero_eigenvalue(amin, atol)
    return throw(
        DomainError(
            amin,
            "The matrix has a (numerically) zero eigenvalue within `domain_atol = $atol`, for which this matrix function is not defined. " *
                "Decrease `domain_atol` if the eigenvalue is genuine and should not be treated as zero."
        )
    )
end

# Real eigenvalues that are negative beyond `atol` cannot be expressed with the same (real) scalar
# type. `clampable` only selects the advice in the error message.
function _check_domain_eigenvalues(λ::AbstractVector{<:Real}, atol::Real, clampable::Bool = true)
    λmin = minimum(λ; init = zero(eltype(λ)))
    λmin < -atol && throw_negative_eigenvalue(λmin, atol, "a negative real eigenvalue", clampable)
    return λ
end

# Complex eigenvalues of a real matrix: only eigenvalues (numerically) on the negative
# real axis obstruct a real result; complex-conjugate pairs do not.
_onaxis(x, atol) = abs(imag(x)) <= atol && real(x) < 0

function _check_domain_eigenvalues(λ::AbstractVector{<:Complex}, atol::Real, clampable::Bool = true)
    λmin = mapreduce(x -> _onaxis(x, atol) ? real(x) : zero(real(x)), min, λ; init = zero(real(eltype(λ))))
    λmin < -atol && throw_negative_eigenvalue(λmin, atol, "an eigenvalue on the negative real axis", clampable)
    return λ
end

# Move the eigenvalues that violate the domain within `atol` onto the boundary.
function _clamp_domain_eigenvalues!(λ::AbstractVector{<:Real}, atol::Real)
    _check_domain_eigenvalues(λ, atol)
    λ .= max.(λ, zero(eltype(λ)))
    return λ
end

function _clamp_domain_eigenvalues!(λ::AbstractVector{<:Complex}, atol::Real)
    _check_domain_eigenvalues(λ, atol)
    λ .= ifelse.(_onaxis.(λ, atol), zero(eltype(λ)), λ)
    return λ
end

# Reject (numerically) zero eigenvalues for functions that are undefined there,
# e.g. `logarithm` and `power` with a negative exponent.
function _check_nonzero_eigenvalues(λ, atol::Real)
    amin = minimum(abs, λ; init = typemax(real(eltype(λ))))
    amin <= atol && throw_zero_eigenvalue(amin, atol)
    return λ
end

# Shared helpers for `MatrixFunctionViaLA`
# ---------------------------------------
# `LinearAlgebra` never exposes the spectrum, so the domain check happens in result space and
# `domain_atol` bounds the imaginary part of `f(A)` instead; see the manual for the consequences.

@noinline function throw_la_kwargs(f, ks)
    return throw(
        ArgumentError("`MatrixFunctionViaLA` only accepts the `domain_atol` keyword argument for `$f`, got $ks")
    )
end

# `MatrixFunctionViaLA` accepts generic keywords, so the kernels validate the ones they support.
function _la_domain_atol(alg::MatrixFunctionViaLA, f)
    ks = keys(alg.kwargs)
    (isempty(ks) || ks == (:domain_atol,)) || throw_la_kwargs(f, ks)
    return get(alg.kwargs, :domain_atol, -1.0)
end

@noinline function throw_complex_result(f, atol, imagmax)
    return throw(
        DomainError(
            f,
            "The result of this matrix function applied to the given real matrix is complex (eigenvalues on the negative real axis): " *
                "its imaginary part reaches $imagmax, beyond `domain_atol = $atol`. Pass a complex matrix to obtain the principal " *
                "value, or increase `domain_atol` if the imaginary part is a rounding artifact."
        )
    )
end

@noinline function throw_nonfinite_result(f)
    return throw(
        DomainError(
            f,
            "The result of this matrix function is not finite, which signals a (numerically) singular input for which it is undefined. " *
                "Use `MatrixFunctionViaEig`/`MatrixFunctionViaEigh` to have the spectrum itself checked against `domain_atol`."
        )
    )
end

# Project a complex `LinearAlgebra` result onto the real output. A rounding-level imaginary part is
# not a domain violation: `LinearAlgebra` casts back to real only when the imaginary part vanishes
# identically, so `schurpow` yields a complex matrix even for an in-domain real one.
function _la_project_real!(fA, fAc, domain_atol::Real, f)
    all(isfinite, fAc) || throw_nonfinite_result(f)
    R = real(eltype(fA))
    # the working precision is that of the output: `LinearAlgebra` computes in complex arithmetic
    # throughout, so e.g. a `Float32` input promotes all the way to `ComplexF64`
    atol = domain_atol < 0 ? defaulttol(fA) * convert(R, norm(fAc, Inf)) : convert(R, domain_atol)
    imagmax = convert(R, maximum(abs ∘ imag, fAc; init = zero(real(eltype(fAc)))))
    imagmax <= atol || throw_complex_result(f, atol, imagmax)
    fA .= real.(fAc)
    return fA
end
