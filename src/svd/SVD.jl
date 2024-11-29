"""
    SVD <: Factorization

Matrix factorization type of the singular value decomposition (SVD) of a matrix `A`.
This is the return type of [`svd(A, ...)`](@ref), the corresponding matrix factorization function.

If `F::SVD` is the factorization object, `U`, `S`, `V` and `Vt` can be obtained
via `F.U`, `F.S`, `F.V` and `F.Vt`, such that `A = U * S * Vt`.
The singular values in `S` are sorted in descending order.

Iterating the decomposition produces the components `U`, `S`, and `V`.

# Examples
```jldoctest
julia> A = [1. 0. 0. 0. 2.; 0. 0. 3. 0. 0.; 0. 0. 0. 0. 0.; 0. 2. 0. 0. 0.]
4×5 Matrix{Float64}:
 1.0  0.0  0.0  0.0  2.0
 0.0  0.0  3.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  2.0  0.0  0.0  0.0

julia> F = svd(A)
SVD{Matrix{Float64},Vector{Float64},Matrix{Float64}}
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

julia> u, s, vt = F; # destructuring via iteration

julia> u == F.U && s == F.S && v == F.V
true
```
"""
struct SVD{TU,TS,TV}
    U::TU
    S::TS
    Vt::TV
end

# iteration for destructuring into components
Base.iterate(S::SVD) = (S.U, Val(:S))
Base.iterate(S::SVD, ::Val{:S}) = (S.S, Val(:Vt))
Base.iterate(S::SVD, ::Val{:Vt}) = (S.V, Val(:done))
Base.iterate(::SVD, ::Val{:done}) = nothing

function Base.propertynames(F::SVD, private::Bool=false)
    return private ? (:V, fieldnames(typeof(F))...) : (:U, :S, :V, :Vt)
end

function Base.getproperty(F::SVD, d::Symbol)
    if d === :V
        return getfield(F, :Vt)'
    elseif d === :S
        return instantiate_S(F)
    else
        return getfield(F, d)
    end
end

# hook to make S a matrix type if necessary.
# Typically Diagonal but don't want to enforce that
function instantiate_S(F::SVD{TU,TS,TV}) where {TU,TS<:AbstractVector,TV}
    return Diagonal(getfield(F, :S))
end

Base.size(A::SVD, dim::Integer) = dim == 1 ? size(A.U, dim) : size(A.Vt, dim)
Base.size(A::SVD) = (size(A, 1), size(A, 2))

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
Base.AbstractMatrix(F::SVD) = *(F...)
Base.AbstractArray(F::SVD) = AbstractMatrix(F)
Base.Matrix(F::SVD) = Array(AbstractArray(F))
Base.Array(F::SVD) = Matrix(F)
SVD(usv::SVD) = usv
# TODO: move to linearalgebra extension?
# SVD(usv::LinearAlgebra.SVD) = SVD(usv.U, usv.S, usv.Vt)
