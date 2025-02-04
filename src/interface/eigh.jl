# Eigh functions
# --------------

# TODO: kwargs for sorting eigenvalues?

docs_eigh_note = """
Note that [`eigh_full`](@ref) and its variants assume additional structure on the input,
and therefore will retain the `eltype` of the input for the eigenvalues and eigenvectors.
For generic eigenvalue decompositions, see [`eig_full`](@ref).
"""

# TODO: do we need "full"?
"""
    eigh_full(A; kwargs...) -> D, V
    eigh_full(A, alg::AbstractAlgorithm) -> D, V
    eigh_full!(A, [DV]; kwargs...) -> D, V
    eigh_full!(A, [DV], alg::AbstractAlgorithm) -> D, V

Compute the full eigenvalue decomposition of the symmetric or hermitian matrix `A`,
such that `A * V = V * D`, where the unitary matrix `V` contains the orthogonal eigenvectors
and the real diagonal matrix `D` contains the associated eigenvalues.

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
    eigh_trunc(A; kwargs...) -> D, V
    eigh_trunc(A, alg::AbstractAlgorithm) -> D, V
    eigh_trunc!(A, [DV]; kwargs...) -> D, V
    eigh_trunc!(A, [DV], alg::AbstractAlgorithm) -> D, V

Compute a partial or truncated eigenvalue decomposition of the symmetric or hermitian matrix
`A`, such that `A * V ≈ V * D`, where the isometric matrix `V` contains a subset of the
orthogonal eigenvectors and the real diagonal matrix `D` contains the associated eigenvalues,
selected according to a truncation strategy. 

!!! note
    The bang method `eigh_trunc!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `DV` as output.

!!! note
    $(docs_eigh_note)

See also [`eigh_full(!)`](@ref eigh_full) and [`eigh_vals(!)`](@ref eigh_vals).
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
for f in (:eigh_full, :eigh_vals)
    f! = Symbol(f, :!)
    @eval begin
        function select_algorithm(::typeof($f), A; kwargs...)
            return select_algorithm($f!, A; kwargs...)
        end
        function select_algorithm(::typeof($f!), A; alg=nothing, kwargs...)
            if alg isa AbstractAlgorithm
                return alg
            elseif alg isa Symbol
                return Algorithm{alg}(; kwargs...)
            else
                isnothing(alg) || throw(ArgumentError("Unknown alg $alg"))
                return default_eig_algorithm(A; kwargs...)
            end
        end
    end
end

function select_algorithm(::typeof(eigh_trunc), A; kwargs...)
    return select_algorithm(eigh_trunc!, A; kwargs...)
end
function select_algorithm(::typeof(eigh_trunc!), A; alg=nothing, trunc=nothing, kwargs...)
    alg_eigh = select_algorithm(eigh_full!, A; alg, kwargs...)
    alg_trunc = trunc isa TruncationStrategy ? trunc :
                trunc isa NamedTuple ? TruncationStrategy(; trunc...) :
                isnothing(trunc) ? NoTruncation() :
                throw(ArgumentError("Unknown truncation strategy: $trunc"))
    return TruncatedAlgorithm(alg_eigh, alg_trunc)
end

# Default to LAPACK 
function default_eigh_algorithm(A::StridedMatrix{T}; kwargs...) where {T<:BlasFloat}
    return LAPACK_MultipleRelativelyRobustRepresentations(; kwargs...)
end
