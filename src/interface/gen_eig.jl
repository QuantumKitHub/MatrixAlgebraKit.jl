# Gen Eig API
# -------
function gen_eig!(A::AbstractMatrix, B::AbstractMatrix, args...; kwargs...)
    return gen_eig_full!(A, B, args...; kwargs...)
end
function gen_eig(A::AbstractMatrix, B::AbstractMatrix, args...; kwargs...)
    return gen_eig_full(A, B, args...; kwargs...)
end

# Gen Eig functions
# -------------

# TODO: kwargs for sorting eigenvalues?

docs_gen_eig_note = """
Note that [`gen_eig_full`](@ref) and its variants do not assume additional structure on the inputs,
and therefore will always return complex eigenvalues and eigenvectors. For the real
generalized eigenvalue decomposition is not yet supported.
"""

# TODO: do we need "full"?
"""
    gen_eig_full(A, B; kwargs...) -> W, V
    gen_eig_full(A, B, alg::AbstractAlgorithm) -> W, V
    gen_eig_full!(A, B, [WV]; kwargs...) -> W, V
    gen_eig_full!(A, B, [WV], alg::AbstractAlgorithm) -> W, V

Compute the full generalized eigenvalue decomposition of the square matrices `A` and `B`,
such that `A * V = B * V * W`, where the invertible matrix `V` contains the generalized eigenvectors
and the diagonal matrix `W` contains the associated generalized eigenvalues.

!!! note
    The bang method `gen_eig_full!` optionally accepts the output structure and
    possibly destroys the input matrices `A` and `B`.
    Always use the return value of the function as it may not always be
    possible to use the provided `WV` as output.

!!! note
    $(docs_gen_eig_note)

See also [`gen_eig_vals(!)`](@ref eig_vals).
"""
@functiondef n_args=2 gen_eig_full

"""
    gen_eig_vals(A, B; kwargs...) -> W
    gen_eig_vals(A, B, alg::AbstractAlgorithm) -> W
    gen_eig_vals!(A, B, [W]; kwargs...) -> W 
    gen_eig_vals!(A, B, [W], alg::AbstractAlgorithm) -> W

Compute the list of generalized eigenvalues of `A` and `B`.

!!! note
    The bang method `gen_eig_vals!` optionally accepts the output structure and
    possibly destroys the input matrices `A` and `B`. Always use the return
    value of the function as it may not always be possible to use the
    provided `W` as output.

!!! note
    $(docs_gen_eig_note)

See also [`gen_eig_full(!)`](@ref gen_eig_full).
"""
@functiondef n_args=2 gen_eig_vals

# Algorithm selection
# -------------------
default_gen_eig_algorithm(A, B; kwargs...) = default_gen_eig_algorithm(typeof(A), typeof(B); kwargs...)
default_gen_eig_algorithm(::Type{TA}, ::Type{TB}; kwargs...) where {TA, TB} = throw(MethodError(default_gen_eig_algorithm, (TA,TB)))
function default_gen_eig_algorithm(::Type{TA}, ::Type{TB}; kwargs...) where {TA<:YALAPACK.BlasMat,TB<:YALAPACK.BlasMat}
    return LAPACK_Simple(; kwargs...)
end

for f in (:gen_eig_full!, :gen_eig_vals!)
    @eval function default_algorithm(::typeof($f), ::Tuple{A, B}; kwargs...) where {A, B}
        return default_gen_eig_algorithm(A, B; kwargs...)
    end
end
