# safe and regularised replacements of common functionality

# Sign
"""
    sign_safe(s::Number)

Compute the sign of a number `s`, but return `+1` if `s` is zero so that the result is
always a number with modulus 1, i.e. an element of the unitary group U(1).
"""
sign_safe(s::Real) = ifelse(s < zero(s), -one(s), +one(s))
sign_safe(s::Complex) = ifelse(iszero(s), one(s), s / abs(s))

# Inverse

"""
    function inv_safe(a::Number, tol=defaulttol(a))

Compute the inverse of a number `a`, but return zero if `a` is smaller than `tol`.
"""
inv_safe(a::Number, tol=defaulttol(a)) = abs(a) < tol ? zero(a) : inv(a)
