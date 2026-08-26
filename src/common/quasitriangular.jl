# Methods for upper quasi-triangular matrices, i.e. the real Schur form: block upper triangular with
# diagonal blocks of size 1 for a real eigenvalue and 2 for a complex-conjugate pair. A complex
# Schur form is the special case where every block has size 1.

# `T[i+1, i]` is the only subdiagonal entry that can be nonzero, so it decides the size of the block
# starting at `i`, while `T[i, i-1]` decides the size of the one ending there
@inline _quasitriu_blocksize(T, i, n) = @inbounds ((i < n && !iszero(T[i + 1, i])) ? 2 : 1)
@inline _quasitriu_blocksize_end(T, i) = @inbounds ((i > 1 && !iszero(T[i, i - 1])) ? 2 : 1)

# bisection point that does not cut a 2x2 block, or `0` for a matrix that is a single block
function _quasitriu_split(T, n)
    n <= 1 && return 0
    s = n ÷ 2
    @inbounds iszero(T[s + 1, s]) || (s += 1)
    return s < n ? s : 0
end

# scratch space for the 4x4 system of the 2x2-by-2x2 Sylvester solve, allocated once per call
_quasitriu_workspace(A) = (similar(A, (4, 4)), similar(A, (4,)))

# Sylvester equations
# -------------------

# `A * X + X * B = C` for upper quasi-triangular `A` and `B`, in place in `C`; note the opposite sign
# convention to `LinearAlgebra.sylvester`. A singular system surfaces as a non-finite result.
function _quasitriu_sylvester!(A, B, C, ws, blocksize::Int)
    m, n = size(C)
    # split the larger dimension; an indivisible one is a single 2x2 block, and then so is the other
    sa = (m >= n && m > blocksize) ? _quasitriu_split(A, m) : 0
    sb = (iszero(sa) && n > blocksize) ? _quasitriu_split(B, n) : 0
    if !iszero(sa)
        # `A = [A11 A12; 0 A22]` and `X = [X1; X2]`, so `X2` is determined on its own
        r1, r2 = 1:sa, (sa + 1):m
        A11, A12, A22 = view(A, r1, r1), view(A, r1, r2), view(A, r2, r2)
        C1, C2 = view(C, r1, :), view(C, r2, :)
        _quasitriu_sylvester!(A22, B, C2, ws, blocksize)
        mul!(C1, A12, C2, -1, 1)
        _quasitriu_sylvester!(A11, B, C1, ws, blocksize)
    elseif !iszero(sb)
        # `B = [B11 B12; 0 B22]` and `X = [X1 X2]`, so `X1` is determined on its own
        c1, c2 = 1:sb, (sb + 1):n
        B11, B12, B22 = view(B, c1, c1), view(B, c1, c2), view(B, c2, c2)
        C1, C2 = view(C, :, c1), view(C, :, c2)
        _quasitriu_sylvester!(A, B11, C1, ws, blocksize)
        mul!(C2, C1, B12, -1, 1)
        _quasitriu_sylvester!(A, B22, C2, ws, blocksize)
    else
        _quasitriu_sylvester_point!(A, B, C, ws)
    end
    return C
end

# Bartels-Stewart: block columns of `B` from the left and block rows of `A` from the bottom, so that
# the corrections only involve blocks that are already solved
function _quasitriu_sylvester_point!(A, B, C, ws)
    m, n = size(C)
    j = 1
    @inbounds while j <= n
        q = _quasitriu_blocksize(B, j, n)
        J = j:(j + q - 1)
        i = m
        while i >= 1
            p = _quasitriu_blocksize_end(A, i)
            I = (i - p + 1):i
            Cij = view(C, I, J)
            i < m && mul!(Cij, view(A, I, (i + 1):m), view(C, (i + 1):m, J), -1, 1)
            j > 1 && mul!(Cij, view(C, I, 1:(j - 1)), view(B, 1:(j - 1), J), -1, 1)
            _quasitriu_sylvester_block!(view(A, I, I), view(B, J, J), Cij, ws)
            i -= p
        end
        j += q
    end
    return C
end

# `A * X + X * B = C` for blocks of size 1 or 2, in place in `C`
Base.@propagate_inbounds function _quasitriu_sylvester_block!(A, B, C, ws)
    p, q = size(C)
    if p == 1 && q == 1
        d = A[1, 1] + B[1, 1]
        # a vanishing right hand side is consistent with a singular equation, and selects the root
        # that vanishes along with it
        C[1, 1] = (iszero(d) && iszero(C[1, 1])) ? d : C[1, 1] / d
    elseif p == 2 && q == 1
        b = B[1, 1]
        _solve_adjugate2!(A[1, 1] + b, A[1, 2], A[2, 1], A[2, 2] + b, C)
    elseif p == 1 && q == 2
        # `a * x + x * B = c` transposes into a system for the row vector `x`
        a = A[1, 1]
        _solve_adjugate2!(B[1, 1] + a, B[2, 1], B[1, 2], B[2, 2] + a, C)
    else
        _quasitriu_sylvester_2x2!(A, B, C, ws)
    end
    return C
end

# the 2x2-by-2x2 case is the 4x4 system `(I ⊗ A + Bᵀ ⊗ I) * vec(X) = vec(C)`
function _quasitriu_sylvester_2x2!(A, B, C, (M, v))
    z = zero(eltype(M))
    @inbounds for j in 1:2, i in 1:2
        r = i + 2 * (j - 1)
        for l in 1:2, k in 1:2
            M[r, k + 2 * (l - 1)] = (j == l ? A[i, k] : z) + (i == k ? B[l, j] : z)
        end
        v[r] = C[i, j]
    end
    _solve_gauss4!(M, v)
    @inbounds for j in 1:2, i in 1:2
        C[i, j] = v[i + 2 * (j - 1)]
    end
    return C
end

# Small dense solvers
# -------------------

# `[m11 m12; m21 m22] * x = c` in place in `c`, through the adjugate
Base.@propagate_inbounds function _solve_adjugate2!(m11, m12, m21, m22, c)
    d = m11 * m22 - m12 * m21
    c1, c2 = c[1], c[2]
    c[1] = (m22 * c1 - m12 * c2) / d
    c[2] = (m11 * c2 - m21 * c1) / d
    return c
end

# 4x4 Gaussian elimination with partial pivoting, in place in `M` and `v`
function _solve_gauss4!(M, v)
    n = 4
    @inbounds for k in 1:n
        p, amax = k, abs(M[k, k])
        for i in (k + 1):n
            a = abs(M[i, k])
            a > amax && ((p, amax) = (i, a))
        end
        if p != k
            for j in k:n
                M[k, j], M[p, j] = M[p, j], M[k, j]
            end
            v[k], v[p] = v[p], v[k]
        end
        piv = M[k, k]
        for i in (k + 1):n
            f = M[i, k] / piv
            iszero(f) && continue
            for j in (k + 1):n
                M[i, j] -= f * M[k, j]
            end
            v[i] -= f * v[k]
        end
    end
    @inbounds for k in n:-1:1
        s = v[k]
        for j in (k + 1):n
            s -= M[k, j] * v[j]
        end
        v[k] = s / M[k, k]
    end
    return v
end
