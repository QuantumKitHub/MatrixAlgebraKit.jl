# Eigh functions
# --------------
# TODO: kwargs for sorting eigenvalues?

docs_eigh_note = """
    Note that [`eigh_full`](@ref) and its variants assume that the input matrix is hermitian,
    or thus symmetric if the input is real. The resulting algorithms exploit this structure,
    and return eigenvalues that are always real, and eigenvectors that are orthogonal and have
    the same `eltype` as the input matrix. If the input matrix does not have this structure,
    the generic eigenvalue decomposition provided by [`eig_full`](@ref) and its variants
    should be used instead.
"""

"""
    eigh_full(A; kwargs...) -> D, V, ϵ
    eigh_full(A, alg::AbstractAlgorithm) -> D, V, ϵ
    eigh_full!(A, [DV]; kwargs...) -> D, V, ϵ
    eigh_full!(A, [DV], alg::AbstractAlgorithm) -> D, V, ϵ

Compute the full eigenvalue decomposition of the symmetric or hermitian matrix `A`,
such that `A * V = V * D`, where the unitary matrix `V` contains the orthogonal eigenvectors
and the real diagonal matrix `D` contains the associated eigenvalues.

The function also returns `ϵ`, the truncation error defined as the 2-norm of the 
discarded eigenvalues.

!!! note
    The bang method `eigh_full!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `DV` as output.

!!! note
$(docs_eigh_note)

See also [`eigh_vals(!)`](@ref eigh_vals) and [`eigh_trunc(!)`](@ref eigh_trunc).
"""
@functiondef eigh_full

"""
    eigh_trunc(A; [trunc], kwargs...) -> D, V
    eigh_trunc(A, alg::AbstractAlgorithm) -> D, V
    eigh_trunc!(A, [DV]; [trunc], kwargs...) -> D, V
    eigh_trunc!(A, [DV], alg::AbstractAlgorithm) -> D, V

Compute a partial or truncated eigenvalue decomposition of the symmetric or hermitian matrix
`A`, such that `A * V ≈ V * D`, where the isometric matrix `V` contains a subset of the
orthogonal eigenvectors and the real diagonal matrix `D` contains the associated eigenvalues,
selected according to a truncation strategy.

## Keyword arguments
The behavior of this function is controlled by the following keyword arguments:

- `trunc`: Specifies the truncation strategy. This can be:
  - A `NamedTuple` with fields `atol`, `rtol`, and/or `maxrank`, which will be converted to
    a [`TruncationStrategy`](@ref). For details on available truncation strategies, see
    [Truncations](@ref).
  - A `TruncationStrategy` object directly (e.g., `truncrank(10)`, `trunctol(atol=1e-6)`, or
    combinations using `&`).
  - `nothing` (default), which keeps all eigenvalues.

- Other keyword arguments are passed to the algorithm selection procedure. If no explicit
  `alg` is provided, these keywords are used to select and configure the algorithm through
  [`MatrixAlgebraKit.select_algorithm`](@ref). The remaining keywords after algorithm
  selection are passed to the algorithm constructor. See [`MatrixAlgebraKit.default_algorithm`](@ref)
  for the default algorithm selection behavior.

When `alg` is a [`TruncatedAlgorithm`](@ref), the `trunc` keyword cannot be specified as the
truncation strategy is already embedded in the algorithm.

!!! note
    The bang method `eigh_trunc!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `DV` as output.

!!! note
$(docs_eigh_note)

See also [`eigh_full(!)`](@ref eigh_full), [`eigh_vals(!)`](@ref eigh_vals), and
[Truncations](@ref) for more information on truncation strategies.
"""
@functiondef eigh_trunc

"""
    eigh_vals(A; kwargs...) -> D
    eigh_vals(A, alg::AbstractAlgorithm) -> D
    eigh_vals!(A, [D]; kwargs...) -> D
    eigh_vals!(A, [D], alg::AbstractAlgorithm) -> D

Compute the list of (real) eigenvalues of the symmetric or hermitian matrix `A`.

!!! note
    The bang method `eigh_vals!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `DV` as output.

!!! note
$(docs_eigh_note)

See also [`eigh_full(!)`](@ref eigh_full) and [`eigh_trunc(!)`](@ref eigh_trunc).
"""
@functiondef eigh_vals

# Algorithm selection
# -------------------
default_eigh_algorithm(A; kwargs...) = default_eigh_algorithm(typeof(A); kwargs...)
function default_eigh_algorithm(T::Type; kwargs...)
    throw(MethodError(default_eigh_algorithm, (T,)))
end
function default_eigh_algorithm(::Type{T}; kwargs...) where {T <: YALAPACK.BlasMat}
    return LAPACK_MultipleRelativelyRobustRepresentations(; kwargs...)
end
function default_eigh_algorithm(::Type{T}; kwargs...) where {T <: Diagonal}
    return DiagonalAlgorithm(; kwargs...)
end

for f in (:eigh_full!, :eigh_vals!)
    @eval function default_algorithm(::typeof($f), ::Type{A}; kwargs...) where {A}
        return default_eigh_algorithm(A; kwargs...)
    end
end

function select_algorithm(::typeof(eigh_trunc!), A, alg; trunc = nothing, kwargs...)
    if alg isa TruncatedAlgorithm
        isnothing(trunc) ||
            throw(ArgumentError("`trunc` can't be specified when `alg` is a `TruncatedAlgorithm`"))
        return alg
    else
        alg_eig = select_algorithm(eigh_full!, A, alg; kwargs...)
        return TruncatedAlgorithm(alg_eig, select_truncation(trunc))
    end
end
