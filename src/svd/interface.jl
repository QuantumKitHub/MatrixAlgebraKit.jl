using LinearAlgebra:
                     LinearAlgebra, Factorization, Algorithm, default_svd_alg, Adjoint,
                     Transpose
using BlockArrays: AbstractBlockMatrix, BlockedArray, BlockedMatrix, BlockedVector
using BlockArrays: BlockLayout

# Singular Value Decomposition:
# need new type to deal with U and V having possible different types
# this is basically a copy of the LinearAlgebra implementation, but has more
# in-place support

# TODO: decide if we want to supply a fallback to LinearAlgebra methods

"""
    SVD

Factorization type of the singular value decomposition (SVD) of an operator `A`.
This is the return type of [`svd(_)`](@ref), the corresponding factorization function.

If `F::SVD` is the factorization object, `U`, `S`, `V` and `Vt` can be obtained
via `F.U`, `F.S`, `F.V` and `F.Vt`, such that `A = U * S * Vt`.
The singular values in `S` are sorted in descending order.

Iterating the decomposition produces the components `U`, `S`, and `Vt`.

# Examples
```jldoctest
julia> A = [1. 0. 0. 0. 2.; 0. 0. 3. 0. 0.; 0. 0. 0. 0. 0.; 0. 2. 0. 0. 0.]
4×5 Matrix{Float64}:
 1.0  0.0  0.0  0.0  2.0
 0.0  0.0  3.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  2.0  0.0  0.0  0.0

julia> F = svd(A)
SVD{Matrix{Float64}, Vector{Float64}, Matrix{Float64}}
U factor:
4×4 Matrix{Float64}:
 0.0  1.0   0.0  0.0
 1.0  0.0   0.0  0.0
 0.0  0.0   0.0  1.0
 0.0  0.0  -1.0  0.0
singular values:
4-element Vector{Float64}:
 3.0
 2.23606797749979
 2.0
 0.0
Vt factor:
4×5 Matrix{Float64}:
 -0.0        0.0  1.0  -0.0  0.0
  0.447214   0.0  0.0   0.0  0.894427
  0.0       -1.0  0.0   0.0  0.0
  0.0        0.0  0.0   1.0  0.0

julia> F.U * F.S * F.Vt
4×5 Matrix{Float64}:
 1.0  0.0  0.0  0.0  2.0
 0.0  0.0  3.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  2.0  0.0  0.0  0.0

julia> u, s, v = F; # destructuring via iteration

julia> u == F.U && s == F.S && v == F.V
true
```
"""
struct SVD{TU,TS,TV}
    U::TU
    S::TS
    Vt::TV
end
function SVD(U::AbstractArray{T}, S::AbstractVector{Tr}, Vt::AbstractArray{T}) where {T,Tr}
    return SVD{typeof(U),typeof(S),typeof(Vt)}(U, S, Vt)
end

# iteration for destructuring into components
Base.iterate(S::SVD) = (S.U, Val(:S))
Base.iterate(S::SVD, ::Val{:S}) = (S.S, Val(:V))
Base.iterate(S::SVD, ::Val{:Vt}) = (S.Vt, Val(:done))
Base.iterate(::SVD, ::Val{:done}) = nothing

# TODO: decide if we want to use _U, _S, _Vt for internal fields
function Base.getproperty(F::SVD, d::Symbol)
    if d === :V
        return getfield(F, :Vt)'
    elseif d === :S
        return instantiate_S(F)
    else
        return getfield(F, d)
    end
end

function instantiate_S(F::SVD{<:Any,<:AbstractVector,<:Any})
    return LinearAlgebra.Diagonal(getfield(F, :S))
end

function Base.propertynames(F::SVD, private::Bool=false)
    return private ? (:V, fieldnames(typeof(F))...) : (:U, :S, :V, :Vt)
end

Base.size(A::SVD) = (size(A.U, 1:(ndims(A.U) - 1))..., size(A.Vt, 2:ndims(A.Vt))...)

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::SVD)
    summary(io, F)
    println(io)
    println(io, "U factor:")
    show(io, mime, F.U)
    println(io, "\nsingular values:")
    show(io, mime, F.S)
    println(io, "\nVt factor:")
    return show(io, mime, F.Vt)
end

Base.adjoint(usv::SVD) = SVD(adjoint(usv.Vt), usv.S, adjoint(usv.U))
Base.transpose(usv::SVD) = SVD(transpose(usv.Vt), usv.S, transpose(usv.U))

# Conversion
Base.AbstractMatrix(F::SVD) = F.U * F.S * F.Vt
Base.AbstractArray(F::SVD) = AbstractMatrix(F)
Base.Matrix(F::SVD) = Array(AbstractArray(F))
Base.Array(F::SVD) = Matrix(F)
SVD(usv::SVD) = usv
SVD(usv::LinearAlgebra.SVD) = SVD(usv.U, usv.S, usv.Vt)

# functions default to LinearAlgebra
# ----------------------------------
"""
    svd!(A; full::Bool = false, alg::Algorithm = default_svd_alg(A)) -> SVD

`svd!` is the same as [`svd`](@ref), but saves space by
overwriting the input `A`, instead of creating a copy. See documentation of [`svd`](@ref) for details.
"""
function svd!(A; kwargs...)
    alg = default_svd_alg(A; kwargs...)
    F = SVD(A, alg)
    return svd!(F, A, alg)
end

"""
    svd(A; full::Bool = false, alg::Algorithm = default_svd_alg(A)) -> SVD

Compute the singular value decomposition (SVD) of `A` and return an `SVD` object.

`U`, `S`, `V` and `Vt` can be obtained from the factorization `F` with `F.U`,
`F.S`, `F.V` and `F.Vt`, such that `A = U * S * Vt`.
The algorithm produces `Vt` and hence `Vt` is more efficient to extract than `V`.
The singular values in `S` are typically sorted in descending order.

Iterating the decomposition produces the components `U`, `S`, and `V`.

If `full = false` (default), a "thin" SVD is returned. For an ``M
\\times N`` matrix `A`, in the full factorization `U` is ``M \\times M``
and `V` is ``N \\times N``, while in the thin factorization `U` is ``M
\\times K`` and `V` is ``N \\times K``, where ``K = \\min(M,N)`` is the
number of singular values.

`alg` specifies which algorithm and LAPACK method to use for SVD, which by default
can be one of the following:
- `alg = DivideAndConquer()` (default): Calls LAPACK `gesdd`.
- `alg = QRIteration()`: Calls LAPACK `gesvd` (typically slower but more accurate/stable).

# Examples
```jldoctest
julia> A = rand(4,3);

julia> F = svd(A); # Store the Factorization Object

julia> A ≈ F.U * F.S * F.Vt
true

julia> U, S, Vt = F; # destructuring via iteration

julia> A ≈ U * S * Vt
true

julia> Uonly, = svd(A); # Store U only

julia> Uonly == U
true
```
"""
svd(A; kwargs...) = svd!(svdcopy_oftype(A); kwargs...)

svdcopy_oftype(A, S=svdtype(eltype(A))) = LinearAlgebra.copy_mutable(A, S)
svdtype(::Type{T}) = promote_type(Float32, typeof(zero(T) / sqrt(abs2(one(T)))))

default_svd_alg(::DenseMatrix; full=false) = DivideAndConquer(full)