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

# grouped_batched_mul
# -------------------

@doc """
    grouped_batched_mul!(Css, Ass, Bss, alphas, betas; kwargs...) -> Css
    grouped_batched_mul!(Css, Ass, Bss, alphas, betas, alg::AbstractAlgorithm) -> Css

Compute a grouped batched matrix multiplication:
`Css[g][k] = alphas[g] * Ass[g][k] * Bss[g][k] + betas[g] * Css[g][k]`
for all group indices `g` and batch indices `k` within each group.

Each of `Css`, `Ass`, `Bss` is a vector of groups, where each group is a vector of matrices.
All matrices within a group must have compatible and uniform dimensions.
Different groups may have different dimensions, transpose characters, and scaling factors.
""" grouped_batched_mul!

function grouped_batched_mul!(Css, Ass, Bss, alphas::AbstractVector, betas::AbstractVector; alg = nothing, kwargs...)
    alg = select_algorithm(grouped_batched_mul!, (Css, Ass, Bss), alg; kwargs...)
    return grouped_batched_mul!(Css, Ass, Bss, alphas, betas, alg)
end

# Algorithm types
# ---------------

"""
    GEMM(; kwargs...)

Algorithm for batched matrix multiplication via BLAS. Uses `cblas_?gemm_batch_strided`
(via YABLAS) for strided 3D arrays, `cblas_?gemm_batch` (via YABLAS) for vectors of
matrices, or CUBLAS on GPU. Requires MKL; errors if the underlying BLAS does not support
the required function. Note that standard OpenBLAS supports `cblas_?gemm_batch` but not
`cblas_?gemm_batch_strided`.
"""
@algdef GEMM

"""
    LoopGEMM(; kwargs...)

Algorithm for batched matrix multiplication via a sequential loop over `mul!`.
Always works regardless of the underlying BLAS implementation.
"""
@algdef LoopGEMM

# Algorithm selection
# -------------------

"""
    MatrixAlgebraKit.default_batched_mul_algorithm(Cs, As, Bs; kwargs...)
    MatrixAlgebraKit.default_batched_mul_algorithm(::Type{TC}, ::Type{TA}, ::Type{TB}; kwargs...)

Select the default algorithm for [`batched_mul!`](@ref) and [`strided_batched_mul!`](@ref).

The default is `LoopGEMM()` for CPU arrays and `GEMM()` for GPU arrays. Extensions may
override the default for their specific array types (e.g. the MKL extension selects
`GEMM()` for both strided 3D arrays and vectors of `BlasFloat` matrices, and the CUDA
extension selects `GEMM()` for `CuArray`s).
"""
default_batched_mul_algorithm(Cs, As, Bs; kwargs...) =
    default_batched_mul_algorithm(typeof(Cs), typeof(As), typeof(Bs); kwargs...)
default_batched_mul_algorithm(::Type, ::Type, ::Type; kwargs...) = LoopGEMM(; kwargs...)

for f in (:batched_mul!, :strided_batched_mul!, :grouped_batched_mul!)
    @eval function default_algorithm(::typeof($f), ::Tuple{C, A, B}; kwargs...) where {C, A, B}
        return default_batched_mul_algorithm(C, A, B; kwargs...)
    end
end
