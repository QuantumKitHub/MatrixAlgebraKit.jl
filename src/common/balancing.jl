"""
    balance!(A::AbstractMatrix; radix=2, maxiter=100) -> A, scale

Balance the square matrix `A` in place through a diagonal similarity `A ← D⁻¹ A D` that reduces its norm.
Each sweep computes, for every index at once, the power-of-`radix` factor that best equalizes the
off-diagonal row and column norms (a simultaneous, Osborne-style variant of the Parlett–Reinsch scaling),
and applies all factors together; sweeps repeat until no index changes or `maxiter` is reached.

The radix is chosen such that the transformation is exact even in floating point arithmetic, which for most
scalar types is just ``2``. The row and column norms are simultaneously adapted to avoid scalar indexing
and lots of kernel calls on GPUs.

The returned `scale` holds the diagonal of `D`, so that a matrix function of the original
input can be recovered from a matrix function `f` of the balanced matrix through
`D f(D⁻¹AD) D⁻¹`, i.e. `expA[i, j] = scale[i] * f(B)[i, j] / scale[j]`.
"""
function balance!(A::AbstractMatrix{T}; radix::Integer = 2, maxiter::Integer = 100) where {T}
    n = LinearAlgebra.checksquare(A)
    R = real(T)
    β = convert(R, radix)
    logβ = log(β)
    scale = fill!(similar(A, R, n), one(R))

    colnorm = similar(A, R, n)
    rownorm = similar(A, R, n)
    f = similar(A, R, n)
    colsum = reshape(colnorm, 1, n)
    rowsum = reshape(rownorm, n, 1)
    d = abs.(diagview(A))
    fᵀ = transpose(f)

    for _ in 1:maxiter
        fill!(colsum, zero(R))
        Base.mapreducedim!(abs, +, colsum, A)
        fill!(rowsum, zero(R))
        Base.mapreducedim!(abs, +, rowsum, A)
        colnorm .-= d
        rownorm .-= d
        f .= _balance_factor.(colnorm, rownorm, β, logβ)
        all(isone, f) && break
        # apply Aᵢⱼ ← Aᵢⱼ fⱼ / fᵢ (i.e. column j scaled by fⱼ, row i by 1/fᵢ) and accumulate.
        A .= A .* fᵀ ./ f
        scale .*= f
    end

    return A, scale
end

# Nearest power-of-`radix` factor `f` that equalizes the scaled off-diagonal norms
# `colnorm·f` and `rownorm/f`, kept only when it reduces their sum (avoids oscillation and
# leaves degenerate rows/columns untouched).
@inline function _balance_factor(colnorm::R, rownorm::R, β::R, logβ::R) where {R}
    (colnorm > 0 && rownorm > 0) || return one(R)
    f = β^round(log(rownorm / colnorm) / (2 * logβ))
    return (colnorm * f + rownorm / f) < convert(R, 19 // 20) * (colnorm + rownorm) ? f : one(R)
end
