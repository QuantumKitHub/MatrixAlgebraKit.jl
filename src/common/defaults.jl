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
default_pullback_rank_atol(A::Diagonal) = default_pullback_rank_atol(diagview(A))

"""
    default_pullback_complement_atol(D, ΔD, ΔV)
    default_pullback_complement_atol(S, ΔU, ΔS, ΔVᴴ)

Default tolerance for the series that sums the contribution from outside the kept vectors in
the truncated pullbacks. The terms of that series have the units of `ΔA`, so the tolerance is
set by the cotangents in those units: `ΔD` and `ΔV ./ transpose(D)`, or `ΔS`, `ΔU ./ transpose(S)`
and `ΔVᴴ ./ S`.
"""
function default_pullback_complement_atol(D, ΔD, ΔV)
    d = diagview(D)
    x = zero(real(float(eltype(d))))
    iszerotangent(ΔD) || (x = max(x, norm(ΔD, Inf)))
    iszerotangent(ΔV) || (x = max(x, norm(ΔV ./ transpose(d), Inf)))
    return eps(typeof(x))^(3 / 4) * x
end
function default_pullback_complement_atol(S, ΔU, ΔS, ΔVᴴ)
    s = diagview(S)
    x = zero(real(float(eltype(s))))
    iszerotangent(ΔS) || (x = max(x, norm(ΔS, Inf)))
    iszerotangent(ΔU) || (x = max(x, norm(ΔU ./ transpose(s), Inf)))
    iszerotangent(ΔVᴴ) || (x = max(x, norm(ΔVᴴ ./ s, Inf)))
    return eps(typeof(x))^(3 / 4) * x
end

"""
    default_hermitian_tol(A)

Default tolerance for deciding to warn if the provided `A` is not hermitian.
"""
default_hermitian_tol(A) = eps(norm(A, Inf))^(3 / 4)


const DEFAULT_FIXGAUGE = Ref(true)

@doc """
    default_fixgauge() -> current_value
    default_fixgauge(new_value::Bool) -> previous_value

Global toggle for enabling or disabling the default behavior of gauge fixing the output of the eigen- and singular value decompositions.
""" default_fixgauge

default_fixgauge() = DEFAULT_FIXGAUGE[]
function default_fixgauge(new_value::Bool)
    previous_value = DEFAULT_FIXGAUGE[]
    DEFAULT_FIXGAUGE[] = new_value
    return previous_value
end

const _fixgauge_docs = "The `fixgauge` keyword can be used to toggle whether or not to fix the gauge of the output, see also [`default_fixgauge`](@ref) for a global toggle and [`gaugefix!`](@ref) for implementation details."
