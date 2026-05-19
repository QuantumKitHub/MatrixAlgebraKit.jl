# Gaussian sketching
# ------------------
"""
    GaussianSketching(howmany; numiter, rng)

Sketching strategy using a Gaussian random matrix with optional power iterations to improve
accuracy on slowly-decaying spectra.

## Fields
- `howmany::Int`: number of singular values to compute.
- `numiter::Int`: number of power iterations (`numiter ≥ 1`; the first counts as the initial
  sketch).
- `rng::AbstractRNG`: random number generator used to draw the Gaussian sketch matrix.
"""
struct GaussianSketching{RNG <: Random.AbstractRNG} <: SketchingStrategy
    howmany::Int
    numiter::Int
    rng::RNG
end

function GaussianSketching(howmany::Integer; numiter::Integer = 2, rng::Random.AbstractRNG = Random.default_rng())
    howmany ≥ 0 || throw(ArgumentError("howmany must be non-negative"))
    numiter ≥ 1 || throw(ArgumentError("numiter must be at least 1 ($numiter)"))
    return GaussianSketching{typeof(rng)}(howmany, numiter, rng)
end

# Entry points
# ------------
"""
    left_sketch(A; howmany, kwargs...) -> Q, B
    left_sketch(A, alg::AbstractAlgorithm) -> Q, B
    left_sketch!(A, [QB]; howmany, kwargs...) -> Q, B
    left_sketch!(A, [QB], alg::AbstractAlgorithm) -> Q, B

Compute an isometric matrix `Q` (orthonormal columns) of size m×k, whose column span approximates the range of `A` of size m×n.
Also create the core factor `B = Q' * A`.
Here `k = howmany` is the sketch dimension.

The keyword arguments construct a [`GaussianSketching`](@ref) strategy unless an explicit `alg::SketchingStrategy` is supplied.
`howmany` is required.

!!! note
    The bang method `left_sketch!` optionally accepts the output matrices `Q, B` and possibly destroys the input matrix `A`.
    Always use the return value of the function as it may not always be possible to use the provided `Q, B` as output.

See also [`right_sketch(!)`](@ref right_sketch) and [`SketchedAlgorithm`](@ref).
"""
@functiondef left_sketch

"""
    right_sketch(A; howmany, kwargs...) -> B, Pᴴ
    right_sketch(A, alg::AbstractAlgorithm) -> B, Pᴴ
    right_sketch!(A, [BPᴴ]; howmany, kwargs...) -> B, Pᴴ
    right_sketch!(A, [BPᴴ], alg::AbstractAlgorithm) -> B, Pᴴ

Compute a right-isometric matrix `Pᴴ` (orthonormal rows) of size k×n, whose row span approximates the range of `A` of size m×n.
Also create the core factor `B = A * Pᴴ'`
Here `k = howmany` is the sketch dimension.

The keyword arguments construct a [`GaussianSketching`](@ref) strategy unless an explicit `alg::SketchingStrategy` is supplied.
`howmany` is required.

!!! note
    The bang method `right_sketch!` optionally accepts the output matrices `BPᴴ` and possibly destroys the input matrix `A`.
    Always use the return value of the function as it may not always be possible to use the provided `BPᴴ` as output.

See also [`left_sketch(!)`](@ref left_sketch) and [`SketchedAlgorithm`](@ref).
"""
@functiondef right_sketch

# Algorithm selection
# -------------------
default_sketch_algorithm(A; kwargs...) = default_sketch_algorithm(typeof(A); kwargs...)
default_sketch_algorithm(T::Type; kwargs...) = throw(MethodError(default_sketch_algorithm, (T,)))
function default_sketch_algorithm(::Type{T}; howmany, kwargs...) where {T <: AbstractMatrix}
    return GaussianSketching(howmany; kwargs...)
end
function default_sketch_algorithm(::Type{<:Base.ReshapedArray{T, N, A}}; kwargs...) where {T, N, A}
    return default_sketch_algorithm(A; kwargs...)
end
function default_sketch_algorithm(::Type{<:SubArray{T, N, A}}; kwargs...) where {T, N, A}
    return default_sketch_algorithm(A; kwargs...)
end

for f in (:left_sketch!, :right_sketch!)
    @eval function default_algorithm(::typeof($f), ::Type{A}; kwargs...) where {A}
        return default_sketch_algorithm(A; kwargs...)
    end
end
