# Based on the design of GPUArrays.jl

"""
    TestSuite

Suite of tests that may be used for all packages inheriting from MatrixAlgebraKit.

"""
module TestSuite

using Test
using MatrixAlgebraKit
using MatrixAlgebraKit: diagview
using LinearAlgebra: Diagonal, norm, istriu, istril, I
using Random, StableRNGs
using Mooncake
using AMDGPU, CUDA
using Enzyme, EnzymeTestUtils

const tests = Dict()

macro testsuite(name, ex)
    safe_name = lowercase(replace(replace(name, " " => "_"), "/" => "_"))
    fn = Symbol("test_", safe_name)
    return quote
        $(esc(fn))(AT; eltypes = supported_eltypes(AT, $(esc(fn)))) = $(esc(ex))(AT, eltypes)
        @assert !haskey(tests, $name) "testsuite already exists"
        tests[$name] = $fn
    end
end

testargs_summary(args...) = string(args)

const rng = StableRNG(123)
seed_rng!(seed) = Random.seed!(rng, seed)

instantiate_matrix(::Type{T}, size) where {T <: Number} = randn(rng, T, size)
instantiate_matrix(::Type{AT}, size) where {AT <: Array} = randn(rng, eltype(AT), size)
instantiate_matrix(::Type{AT}, size) where {AT <: CuArray} = CuArray(randn(rng, eltype(AT), size))
instantiate_matrix(::Type{AT}, size) where {AT <: ROCArray} = ROCArray(randn(rng, eltype(AT), size))
instantiate_matrix(::Type{AT}, size) where {AT <: Diagonal} = Diagonal(randn(rng, eltype(AT), size))
instantiate_matrix(::Type{AT}, size) where {T, AT <: Diagonal{T, <:CuVector}} = Diagonal(CuArray(randn(rng, eltype(AT), size)))
instantiate_matrix(::Type{AT}, size) where {T, AT <: Diagonal{T, <:ROCVector}} = Diagonal(ROCArray(randn(rng, eltype(AT), size)))

precision(::Type{T}) where {T <: Number} = sqrt(eps(real(T)))
precision(::Type{T}) where {T} = precision(eltype(T))

function has_positive_diagonal(A)
    T = eltype(A)
    return if T <: Real
        all(≥(zero(T)), diagview(A))
    else
        all(≥(zero(real(T))), real(diagview(A))) &&
            all(≈(zero(real(T))), imag(diagview(A)))
    end
end
isleftnull(N, A; atol::Real = 0, rtol::Real = precision(eltype(A))) =
    isapprox(norm(A' * N), 0; atol = max(atol, norm(A) * rtol))

isrightnull(Nᴴ, A; atol::Real = 0, rtol::Real = precision(eltype(A))) =
    isapprox(norm(A * Nᴴ'), 0; atol = max(atol, norm(A) * rtol))

is_positive(::MatrixAlgebraKit.AbstractAlgorithm) = false
is_pivoted(::MatrixAlgebraKit.AbstractAlgorithm) = false
is_positive(alg::MatrixAlgebraKit.LAPACK_HouseholderQR) = alg.positive
is_pivoted(alg::MatrixAlgebraKit.LAPACK_HouseholderQR) = alg.pivoted
is_positive(alg::MatrixAlgebraKit.LAPACK_HouseholderLQ) = alg.positive
is_pivoted(alg::MatrixAlgebraKit.LAPACK_HouseholderLQ) = alg.pivoted
is_positive(alg::MatrixAlgebraKit.CUSOLVER_HouseholderQR) = alg.positive
is_positive(alg::MatrixAlgebraKit.ROCSOLVER_HouseholderQR) = alg.positive
is_positive(alg::MatrixAlgebraKit.LQViaTransposedQR) = is_positive(alg.qr_alg)
is_pivoted(alg::MatrixAlgebraKit.LQViaTransposedQR) = is_pivoted(alg.qr_alg)

isleftcomplete(V, N) = V * V' + N * N' ≈ I
isleftcomplete(V::AnyCuMatrix, N::AnyCuMatrix) = isleftcomplete(collect(V), collect(N))
isleftcomplete(V::AnyROCMatrix, N::AnyROCMatrix) = isleftcomplete(collect(V), collect(N))
isrightcomplete(Vᴴ, Nᴴ) = Vᴴ' * Vᴴ + Nᴴ' * Nᴴ ≈ I
isrightcomplete(V::AnyCuMatrix, N::AnyCuMatrix) = isrightcomplete(collect(V), collect(N))
isrightcomplete(V::AnyROCMatrix, N::AnyROCMatrix) = isrightcomplete(collect(V), collect(N))

instantiate_unitary(T, A, sz) = qr_compact(randn!(similar(A, eltype(T), sz, sz)))[1]
# AMDGPU can't generate ComplexF32 random numbers
function instantiate_unitary(T, A::ROCMatrix{<:Complex}, sz)
    sqA = randn!(similar(A, real(eltype(T)), sz, sz)) .+ im .* randn!(similar(A, real(eltype(T)), sz, sz))
    return qr_compact(sqA)[1]
end
instantiate_unitary(::Type{<:Diagonal}, A, sz) = Diagonal(fill!(similar(parent(A), eltype(A), sz), one(eltype(A))))

function instantiate_rank_deficient_matrix(T, sz; trunc = trunctol(rtol = 0.5))
    A = instantiate_matrix(T, sz)
    V, C = left_orth!(A; trunc = trunctol(rtol = 0.5))
    return mul!(A, V, C)
end

include("ad_utils.jl")

include("projections.jl")

# Decompositions
# --------------
include("decompositions/qr.jl")
include("decompositions/lq.jl")
include("decompositions/polar.jl")
include("decompositions/schur.jl")
include("decompositions/eig.jl")
include("decompositions/eigh.jl")
include("decompositions/orthnull.jl")
include("decompositions/svd.jl")

# Mooncake
# --------
include("mooncake/mooncake.jl")
include("mooncake/qr.jl")
include("mooncake/lq.jl")
include("mooncake/eig.jl")
include("mooncake/eigh.jl")
include("mooncake/svd.jl")
include("mooncake/polar.jl")
include("mooncake/orthnull.jl")

include("chainrules.jl")

# Enzyme
# ------
include("enzyme/eig.jl")
include("enzyme/eigh.jl")
include("enzyme/qr.jl")
include("enzyme/lq.jl")
include("enzyme/svd.jl")
include("enzyme/polar.jl")
include("enzyme/orthnull.jl")

end
