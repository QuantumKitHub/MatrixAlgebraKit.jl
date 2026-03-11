# batched_mul
# -----------

@doc """
    batched_mul!(Cs, As, Bs, α, β; kwargs...) -> Cs
    batched_mul!(Cs, As, Bs, α, β, alg::AbstractAlgorithm) -> Cs

Compute the batched matrix multiplication `Cs[k] = α * As[k] * Bs[k] + β * Cs[k]`
for all batch indices `k`, where `As`, `Bs`, `Cs` are `AbstractVector`s of matrices.

The out-of-place form allocates `Cs` and computes `As * Bs` (i.e. `α = true`, `β = false`).
""" batched_mul!

function batched_mul!(Cs, As, Bs, α::Number, β::Number; alg = nothing, kwargs...)
    alg = select_algorithm(batched_mul!, (Cs, As, Bs), alg; kwargs...)
    return batched_mul!(Cs, As, Bs, α, β, alg)
end

# strided_batched_mul
# -------------------

@doc """
    strided_batched_mul!(C, A, B, α, β; kwargs...) -> C
    strided_batched_mul!(C, A, B, α, β, alg::AbstractAlgorithm) -> C

Compute the batched matrix multiplication `C[:, :, k] = α * A[:, :, k] * B[:, :, k] + β * C[:, :, k]`
for all batch indices `k`, where `A` is an `m×p×batch` array, `B` is a `p×n×batch` array, and
`C` is an `m×n×batch` array.

The out-of-place form allocates `C` and computes `A * B` (i.e. `α = true`, `β = false`).
""" strided_batched_mul!

function strided_batched_mul!(C, A, B, α, β; alg = nothing, kwargs...)
    alg = select_algorithm(strided_batched_mul!, (C, A, B), alg; kwargs...)
    return strided_batched_mul!(C, A, B, α, β, alg)
end

# Algorithm selection
# -------------------

"""
    GEMM(; driver::Driver = DefaultDriver, kwargs...)

Algorithm type for (batched) GEneral Matrix Multiplication.
"""
@algdef GEMM

"""
    MatrixAlgebraKit.default_batched_mul_algorithm(Cs, As, Bs; kwargs...)
    MatrixAlgebraKit.default_batched_mul_algorithm(::Type{TC}, ::Type{TA}, ::Type{TB}; kwargs...)

Select the default algorithm for [`batched_mul!`](@ref) given input arrays `As` and `Bs`.
"""
default_batched_mul_algorithm(Cs, As, Bs; kwargs...) =
    default_batched_mul_algorithm(typeof(Cs), typeof(As), typeof(Bs); kwargs...)
default_batched_mul_algorithm(::Type, ::Type, ::Type; kwargs...) = GEMM(; kwargs...)

for f in (:batched_mul!, :strided_batched_mul!)
    @eval function default_algorithm(::typeof($f), ::Tuple{C, A, B}; kwargs...) where {C, A, B}
        return default_batched_mul_algorithm(C, A, B; kwargs...)
    end
end
