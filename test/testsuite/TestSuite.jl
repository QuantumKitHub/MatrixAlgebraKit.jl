# Based on the design of GPUArrays.jl

"""
    TestSuite

Suite of tests that may be used for all packages inheriting from MatrixAlgebraKit.

"""
module TestSuite

using Test, TestExtras
using MatrixAlgebraKit
using MatrixAlgebraKit: diagview
using LinearAlgebra: norm, istriu

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

instantiate_matrix(::Type{T}, size) where {T <: Number} = randn(T, size)
instantiate_matrix(::Type{AT}, size) where {AT <: Array} = randn(eltype(AT), size)

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

# TODO: actually make this a test
macro testinferred(ex)
    return esc(:(@inferred $ex))
end

include("qr.jl")

end
