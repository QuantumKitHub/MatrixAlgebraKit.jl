using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using MatrixAlgebraKit: diagview
using LinearAlgebra

BLASFloats = (Float32, Float64, ComplexF32, ComplexF64)
@testset "exp! for T = $T" for T in BLASFloats
    rng = StableRNG(123)
    m = 2

    A = randn(rng, T, m, m)
    A = (A + A') / 2

    D, V = @constinferred eigh_full(A)

    expA = @constinferred exp(A)
    Dexp, Vexp = @constinferred eigh_full(expA)

    println("A = ", A)
    println("exp(A) = ", expA)

    println("LHS = ", diagview(Dexp))
    println("RHS = ", LinearAlgebra.exp.(diagview(D)))
    @assert diagview(Dexp) ≈ LinearAlgebra.exp.(diagview(D))

end