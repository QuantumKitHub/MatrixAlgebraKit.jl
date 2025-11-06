# TODO: make Defaults module? Replace `eltype` with `VectorInterface.scalartype`?

"""
    defaulttol(x)

Default tolerance or precision for a given object, e.g. to decide when it can
be considerd to be zero or ignored in some other way, or how accurate some
quantity needs to be computed.
"""
defaulttol(x::Any) = eps(real(float(one(eltype(x)))))^(2 / 3)

"""
    default_pullback_gauge_atol(ΔA...)

Default tolerance for deciding to warn if incoming adjoints of a pullback rule
has components that are not gauge-invariant.
"""
default_pullback_gauge_atol(A) = iszerotangent(A) ? 0 : eps(norm(A, Inf))^(3 / 4)
function default_pullback_gauge_atol(A, As...)
    As′ = filter(!iszerotangent, (A, As...))
    return isempty(As′) ? 0 : eps(norm(As′, Inf))^(3 / 4)
end

"""
    default_pullback_degeneracy_atol(A)

Default tolerance for deciding when values should be considered as degenerate.
"""
default_pullback_degeneracy_atol(A) = eps(norm(A, Inf))^(3 / 4)

"""
    default_pullback_rank_atol(A)

Default tolerance for deciding what values should be considered equal to 0.
"""
default_pullback_rank_atol(A) = eps(norm(A, Inf))^(3 / 4)

"""
    default_hermitian_tol(A)

Default tolerance for deciding to warn if the provided `A` is not hermitian.
"""
default_hermitian_tol(A) = eps(norm(A, Inf))^(3 / 4)
