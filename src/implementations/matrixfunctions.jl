# Inputs
# ------
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

# Reconstruction from an eigenvalue decomposition
# -----------------------------------------------
# both take the already-transformed eigenvalues `fD = f(D)` and rebuild `f(A)`

# `f(A) = V f(D) V⁻¹`. A real matrix has a complex decomposition but a real `f(A)` whenever `f(D)`
# closes under conjugation, so the imaginary part is dropped after the solve; the callers are
# responsible for rejecting the inputs where it would not be.
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

# `f(A) = V f(D) V'` for a hermitian `A`; the product is hermitian only up to roundoff
_apply_eigh!(fA, V, fD) = project_hermitian!(mul!(fA, V * fD, V'))

# Domain handling
# ---------------
# `atol` is the slack on the domain boundary, i.e. how far outside the domain an eigenvalue may
# stray before it counts as a genuine violation rather than a rounding artifact. `axis_atol` is the
# accuracy with which the algorithm resolves whether an eigenvalue lies *on* the negative real axis
# at all, which is a property of the eigensolver rather than a user choice.

# an unset keyword denotes the runtime default
_domain_atol(alg) = get(alg.kwargs, :domain_atol, nothing)

# callers resolve the default before delegating to an inner `DiagonalAlgorithm`, so that the
# tolerance follows the algorithm that computed the eigenvalues
function _resolve_domain_atol(λ, alg)
    R = real(float(eltype(λ)))
    atol = _domain_atol(alg)
    return convert(R, isnothing(atol) ? default_domain_atol(λ) : atol)
end

_axis_atol(λ) = convert(real(float(eltype(λ))), default_domain_atol(λ))

# the wrapped decomposition algorithm is optional as well
_eig_alg(alg) = get(alg.kwargs, :eig_alg, nothing)
_eigh_alg(alg) = get(alg.kwargs, :eigh_alg, nothing)
_schur_alg(alg) = get(alg.kwargs, :schur_alg, nothing)

# `0` denotes the algorithm default, `1` the unblocked algorithm, following `qr_householder!`
_blocksize(alg) = get(alg.kwargs, :blocksize, 0)

# the throwing branches live in `@noinline` helpers so that the reductions and broadcasts
# below stay free of error-path code, which keeps them GPU friendly
@noinline function throw_negative_eigenvalue(λmin, atol, what)
    return throw(
        DomainError(
            λmin,
            "The matrix has $what beyond `domain_atol = $atol` and the result of this matrix function is complex. " *
                "Pass a complex matrix to obtain the principal value, or increase `domain_atol` if the eigenvalue is a rounding artifact."
        )
    )
end

# real eigenvalues are on the real axis by construction, so `axis_atol` is unused here
function _check_domain_eigenvalues(λ::AbstractVector{<:Real}, atol::Real, axis_atol::Real = atol)
    λmin = minimum(λ; init = zero(eltype(λ)))
    λmin < -atol && throw_negative_eigenvalue(λmin, atol, "a negative real eigenvalue")
    return λ
end

# for the complex eigenvalues of a real matrix, only the ones (numerically) on the negative real
# axis obstruct a real result; complex-conjugate pairs do not
_onaxis(x, axis_atol) = abs(imag(x)) <= axis_atol && real(x) < 0

function _check_domain_eigenvalues(λ::AbstractVector{<:Complex}, atol::Real, axis_atol::Real = atol)
    λmin = mapreduce(x -> _onaxis(x, axis_atol) ? real(x) : zero(real(x)), min, λ; init = zero(real(eltype(λ))))
    λmin < -atol && throw_negative_eigenvalue(λmin, atol, "an eigenvalue on the negative real axis")
    return λ
end

# move the eigenvalues that violate the domain within `atol` onto the boundary
function _clamp_domain_eigenvalues!(λ::AbstractVector{<:Real}, atol::Real, axis_atol::Real = atol)
    _check_domain_eigenvalues(λ, atol, axis_atol)
    λ .= max.(λ, zero(eltype(λ)))
    return λ
end

function _clamp_domain_eigenvalues!(λ::AbstractVector{<:Complex}, atol::Real, axis_atol::Real = atol)
    _check_domain_eigenvalues(λ, atol, axis_atol)
    λ .= ifelse.(_onaxis.(λ, axis_atol), zero(eltype(λ)), λ)
    return λ
end

# Domain handling for `MatrixFunctionViaLA`
# -----------------------------------------
# `LinearAlgebra` never exposes the spectrum, so there is nothing to compare against a tolerance:
# a complex result for a real input is a domain violation, full stop

@noinline function throw_la_kwargs(f, ks)
    return throw(
        ArgumentError(
            "`MatrixFunctionViaLA` accepts no keyword arguments for `$f`, got $ks. In particular " *
                "`domain_atol` is not supported, as `LinearAlgebra` does not expose the spectrum; " *
                "use `MatrixFunctionViaSchur`, `MatrixFunctionViaEig` or `MatrixFunctionViaEigh` instead."
        )
    )
end

# `MatrixFunctionViaLA` accepts generic keywords, so the kernels reject the ones they cannot honor
function _check_la_kwargs(alg::MatrixFunctionViaLA, f)
    ks = keys(alg.kwargs)
    isempty(ks) || throw_la_kwargs(f, ks)
    return nothing
end

@noinline function throw_complex_result(f)
    return throw(
        DomainError(
            f,
            "The result of this matrix function applied to the given real matrix is complex (eigenvalues on the negative real axis). " *
                "Pass a complex matrix to obtain the principal value, or use `MatrixFunctionViaEig`/`MatrixFunctionViaEigh` to have " *
                "the spectrum itself checked against `domain_atol`."
        )
    )
end

@noinline function throw_nonfinite_result(f, advice)
    return throw(
        DomainError(
            f,
            "The result of this matrix function is not finite, which signals a (numerically) singular input for which it is undefined. " *
                advice
        )
    )
end

const _NONFINITE_LA_ADVICE = "Use `MatrixFunctionViaEig`/`MatrixFunctionViaEigh` to have the spectrum itself checked against `domain_atol`."

# the Schur form does expose the spectrum, so a singular result there is a property of the input
# rather than of the tolerance: it means two of the eigenvalue square roots cancel, as a repeated
# zero eigenvalue does, leaving one of the equations for the off-diagonal entries unsolvable
const _NONFINITE_SCHUR_ADVICE = "Two of the eigenvalue square roots sum to zero, which leaves the result undefined whatever the tolerance."
