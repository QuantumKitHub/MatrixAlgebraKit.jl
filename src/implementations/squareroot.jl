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
        all(isfinite, sqrtAc) || throw_nonfinite_result(squareroot!, _NONFINITE_LA_ADVICE)
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

# Schur logic
# -----------
function squareroot!(A::AbstractMatrix, sqrtA, alg::MatrixFunctionViaSchur)
    check_input(squareroot!, A, sqrtA, alg)
    T, Z, vals = schur_full!(A, select_algorithm(schur_full!, A, _schur_alg(alg)))
    # the (quasi-)diagonal of `T` is the spectrum, and it is the block structure rather than a
    # tolerance that decides whether an eigenvalue lies on the negative real axis
    eltype(T) <: Real && _clamp_domain_quasitriu!(T, _resolve_domain_atol(vals, alg))
    R = _squareroot_quasitriu!(zero!(similar(T)), T, _squareroot_blocksize(T, alg))
    all(isfinite, R) || throw_nonfinite_result(squareroot!, _NONFINITE_SCHUR_ADVICE)
    ZR = mul!(T, Z, R) # `T` shares its storage with `A` and is no longer needed
    has_equal_storage(sqrtA, T) || return mul!(sqrtA, ZR, Z')
    return copy!(sqrtA, mul!(R, ZR, Z'))
end

# How deep it pays to recurse follows whether the multiplications reach a level-3 BLAS: measured
# over `n = 500` to `2000` a BLAS float is fastest at `2`-`4` and loses up to `1.7x` at `64`, while
# `BigFloat` prefers the entrywise algorithm outright and loses `2.2x` at `1`.
function _squareroot_blocksize(T, alg)
    blocksize = _blocksize(alg)
    blocksize > 0 && return blocksize
    return eltype(T) <: BlasFloat ? 4 : 64
end

# A real result requires the real eigenvalues to be nonnegative, and those are exactly the `1x1`
# blocks: a `2x2` block has `T[i, i+1] * T[i+1, i] < 0`, hence a conjugate pair off the negative
# real axis, which always has a real square root. Hence no `axis_atol` here.
function _clamp_domain_quasitriu!(T::AbstractMatrix{<:Real}, atol::Real)
    n = size(T, 1)
    tmin = zero(eltype(T))
    @inbounds begin
        i = 1
        while i <= n
            p = _quasitriu_blocksize(T, i, n)
            p == 1 && (tmin = min(tmin, T[i, i]))
            i += p
        end
        tmin < -atol && throw_negative_eigenvalue(tmin, atol, "a negative real eigenvalue")
        i = 1
        while i <= n
            p = _quasitriu_blocksize(T, i, n)
            p == 1 && T[i, i] < 0 && (T[i, i] = zero(eltype(T)))
            i += p
        end
    end
    return T
end

# `R = [R11 R12; 0 R22]` turns `R^2 = T` into two smaller square roots and the Sylvester equation
# `R11 * R12 + R12 * R22 = T12`, whose corrections are matrix multiplications
function _squareroot_quasitriu!(R, T, blocksize::Int, ws = _quasitriu_workspace(R))
    n = size(T, 1)
    s = n > blocksize ? _quasitriu_split(T, n) : 0
    iszero(s) && return _squareroot_quasitriu_point!(R, T, ws)
    r1, r2 = 1:s, (s + 1):n
    R11, R22 = view(R, r1, r1), view(R, r2, r2)
    _squareroot_quasitriu!(R11, view(T, r1, r1), blocksize, ws)
    _squareroot_quasitriu!(R22, view(T, r2, r2), blocksize, ws)
    R12 = copy!(view(R, r1, r2), view(T, r1, r2))
    _quasitriu_sylvester!(R11, R22, R12, ws, blocksize)
    return R
end

# Björck-Hammarling: the diagonal blocks are scalar or `2x2` square roots, the off-diagonal blocks
# solve `R_ii * R_ij + R_ij * R_jj = T_ij - sum(R_ik * R_kj for i < k < j)`, by block column so that
# the sum only involves blocks that are already known
function _squareroot_quasitriu_point!(R, T, ws)
    n = size(T, 1)
    @inbounds begin
        i = 1
        while i <= n
            p = _quasitriu_blocksize(T, i, n)
            if p == 1
                R[i, i] = _squareroot_diag(T[i, i])
            else
                I = i:(i + 1)
                _squareroot_2x2!(view(R, I, I), view(T, I, I))
            end
            i += p
        end
        j = 1
        while j <= n
            q = _quasitriu_blocksize(T, j, n)
            J = j:(j + q - 1)
            i = j - 1
            while i >= 1
                p = _quasitriu_blocksize_end(T, i)
                I = (i - p + 1):i
                Rij = copy!(view(R, I, J), view(T, I, J))
                K = (i + 1):(j - 1)
                isempty(K) || mul!(Rij, view(R, I, K), view(R, K, J), -1, 1)
                _quasitriu_sylvester_block!(view(R, I, I), view(R, J, J), Rij, ws)
                i -= p
            end
            j += q
        end
    end
    return R
end

# `sqrt` is discontinuous across the negative real axis, where the sign of a computed zero imaginary
# part is noise; pinning the branch keeps two copies of a negative eigenvalue from receiving
# opposite roots, which would leave the off-diagonal equations singular.
_squareroot_diag(t::Real) = sqrt(t)
function _squareroot_diag(t::Complex)
    imt = imag(t)
    return sqrt(iszero(imt) ? complex(real(t), abs(imt)) : t)
end

# Real square root of a `2x2` block with eigenvalues `θ ± im * μ`, following Higham (2008), Alg. 6.5
# and eqs. (6.8)-(6.9); the standardized form produced by LAPACK `?gees` and `GenericSchur.gschur!`
# has equal diagonal entries.
Base.@propagate_inbounds function _squareroot_2x2!(R, T)
    θ, b, c = T[1, 1], T[1, 2], T[2, 1]
    μ = sqrt(abs(b)) * sqrt(abs(c))
    # the real part of `sqrt(θ + im * μ)`, in the form that avoids cancellation for `θ < 0`
    t = sqrt((abs(θ) + hypot(θ, μ)) / 2)
    α = θ >= zero(θ) ? t : μ / (2 * t)
    R[1, 1] = α
    R[2, 2] = α
    R[1, 2] = b / (2 * α)
    R[2, 1] = c / (2 * α)
    return R
end
