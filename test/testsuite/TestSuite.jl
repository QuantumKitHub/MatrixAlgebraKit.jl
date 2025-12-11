# Based on the design of GPUArrays.jl

"""
    TestSuite

Suite of tests that may be used for all packages inheriting from MatrixAlgebraKit.

"""
module TestSuite

using Test
using MatrixAlgebraKit
using MatrixAlgebraKit: diagview
using LinearAlgebra: Diagonal, norm, istriu, istril
using Random, StableRNGs
using AMDGPU, CUDA

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

# TODO: actually make this a test
macro testinferred(ex)
    return esc(:(@inferred $ex))
end

include("ad_utils.jl")

include("qr.jl")
include("lq.jl")
include("polar.jl")
include("orthnull.jl")
include("projections.jl")
include("eigh.jl")
include("eig.jl")
include("schur.jl")
include("svd.jl")
include("mooncake.jl")
include("chainrules.jl")

end
