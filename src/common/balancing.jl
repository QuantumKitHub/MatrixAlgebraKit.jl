"""
    balance!(A::AbstractMatrix; radix=2) -> A, scale

Balance the square matrix `A` in place through a diagonal similarity `A ← D⁻¹ A D` that
reduces its norm, using the scaling phase of the Parlett–Reinsch algorithm (as in LAPACK's
`gebal`). The returned `scale` holds the diagonal of `D`, so that a matrix function of the
original input can be recovered from a matrix function `f` of the balanced matrix through
`D f(D⁻¹AD) D⁻¹`, i.e. `expA[i, j] = scale[i] * f(B)[i, j] / scale[j]`.
"""
function balance!(A::AbstractMatrix{T}; radix::Integer = 2) where {T}
    n = LinearAlgebra.checksquare(A)
    R = real(T)
    scale = ones(R, n)
    β = convert(R, radix)
    β² = β * β

    converged = false
    while !converged
        converged = true
        for i in 1:n
            colnorm = zero(R)
            rownorm = zero(R)
            for j in 1:n
                j == i && continue
                colnorm += abs(A[j, i])
                rownorm += abs(A[i, j])
            end
            (iszero(colnorm) || iszero(rownorm)) && continue

            factor = one(R)
            total = colnorm + rownorm
            threshold = rownorm / β
            while colnorm < threshold
                factor *= β
                colnorm *= β²
            end
            threshold = rownorm * β
            while colnorm >= threshold
                factor /= β
                colnorm /= β²
            end

            (colnorm + rownorm) < convert(R, 0.95) * factor * total || continue
            converged = false
            scale[i] *= factor
            invfactor = inv(factor)
            for j in 1:n
                A[i, j] *= invfactor
            end
            for j in 1:n
                A[j, i] *= factor
            end
        end
    end

    return A, scale
end
