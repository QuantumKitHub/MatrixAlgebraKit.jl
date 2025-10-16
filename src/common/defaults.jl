# TODO: make Defaults module? Replace `eltype` with `VectorInterface.scalartype`?

"""
    defaulttol(x)

Default tolerance or precision for a given object, e.g. to decide when it can
be considerd to be zero or ignored in some other way, or how accurate some
quantity needs to be computed.
"""
defaulttol(x::Any) = eps(real(float(one(eltype(x)))))^(2 / 3)

"""
    default_pullback_gaugetol(a)

Default tolerance for deciding to warn if incoming adjoints of a pullback rule
has components that are not gauge-invariant.
"""
function default_pullback_gaugetol(a)
    n = norm(a, Inf)
    return eps(eltype(n))^(3 / 4) * max(n, one(n))
end

"""
    default_hermitian_tol(A)

Default tolerance for deciding to warn if the provided `A` is not hermitian.
"""
function default_hermitian_tol(A)
    n = norm(A, Inf)
    return eps(eltype(n))^(3 / 4) * max(n, one(n))
end
