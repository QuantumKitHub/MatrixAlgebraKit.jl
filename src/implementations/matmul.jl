check_matmul_dims(::Type{Bool}, C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix) =
    size(C, 1) == size(A, 1) && size(C, 2) == size(B, 2) && size(A, 2) == size(B, 1)
check_matmul_dims(C, A, B) = check_matmul_dims(Bool, C, A, B) ||
    throw(DimensionMismatch(lazy"incompatible matrix multiplication dimensions: $(size(C)) ← $(size(A)) * $(size(B))"))

# batched_mul
# -----------

function check_input(::typeof(batched_mul!), Cs, As, Bs, ::AbstractAlgorithm)
    length(As) == length(Bs) == length(Cs) ||
        throw(DimensionMismatch(lazy"Length of Cs ($(length(Cs))), As ($(length(As))) and Bs ($(length(Bs))) must match"))
    foreach(check_matmul_dims, Cs, As, Bs)
    return nothing
end

function batched_mul!(Cs, As, Bs, alpha::Number, beta::Number, alg::LoopGEMM)
    check_input(batched_mul!, Cs, As, Bs, alg)
    @inbounds for k in eachindex(Cs, As, Bs)
        mul!(Cs[k], As[k], Bs[k], alpha, beta)
    end
    return Cs
end

function batched_mul!(
        Cs::AbstractVector{<:AbstractMatrix{T}},
        As::AbstractVector{<:AbstractMatrix{T}},
        Bs::AbstractVector{<:AbstractMatrix{T}},
        alpha::Number, beta::Number, alg::GEMM
    ) where {T <: BlasFloat}
    check_input(batched_mul!, Cs, As, Bs, alg)
    transA = YABLAS._trans_char(first(As))
    transB = YABLAS._trans_char(first(Bs))
    YABLAS.gemm_batched!(transA, transB, T(alpha), As, Bs, T(beta), Cs)
    return Cs
end

# grouped_batched_mul
# -------------------

function check_input(::typeof(grouped_batched_mul!), Css, Ass, Bss, alphas, betas, ::AbstractAlgorithm)
    ngroups = length(Ass)
    length(Bss) == ngroups && length(Css) == ngroups ||
        throw(DimensionMismatch(lazy"lengths of Css ($(length(Css))), Ass ($ngroups), and Bss ($(length(Bss))) must match"))
    length(alphas) == ngroups ||
        throw(DimensionMismatch(lazy"length of alphas ($(length(alphas))) must match number of groups ($ngroups)"))
    length(betas) == ngroups ||
        throw(DimensionMismatch(lazy"length of betas ($(length(betas))) must match number of groups ($ngroups)"))
    for g in 1:ngroups
        ng = length(Ass[g])
        length(Bss[g]) == ng && length(Css[g]) == ng ||
            throw(DimensionMismatch(lazy"group $g: lengths of Css[$g] ($(length(Css[g]))), Ass[$g] ($ng), and Bss[$g] ($(length(Bss[g]))) must match"))
        foreach(check_matmul_dims, Css[g], Ass[g], Bss[g])
    end
    return nothing
end

function grouped_batched_mul!(Css, Ass, Bss, alphas::AbstractVector, betas::AbstractVector, alg::LoopGEMM)
    check_input(grouped_batched_mul!, Css, Ass, Bss, alphas, betas, alg)
    @inbounds for g in eachindex(Css, Ass, Bss, alphas, betas)
        for k in eachindex(Css[g], Ass[g], Bss[g])
            mul!(Css[g][k], Ass[g][k], Bss[g][k], alphas[g], betas[g])
        end
    end
    return Css
end

function grouped_batched_mul!(
        Css::AbstractVector{<:AbstractVector{<:AbstractMatrix{T}}},
        Ass::AbstractVector{<:AbstractVector{<:AbstractMatrix{T}}},
        Bss::AbstractVector{<:AbstractVector{<:AbstractMatrix{T}}},
        alphas::AbstractVector, betas::AbstractVector, alg::GEMM
    ) where {T <: BlasFloat}
    check_input(grouped_batched_mul!, Css, Ass, Bss, alphas, betas, alg)
    transAs = map(Ag -> YABLAS._trans_char(first(Ag)), Ass)
    transBs = map(Bg -> YABLAS._trans_char(first(Bg)), Bss)
    YABLAS.gemm_grouped_batched!(transAs, transBs, T.(alphas), Ass, Bss, T.(betas), Css)
    return Css
end

# strided_batched_mul
# -------------------

function check_input(
        ::typeof(strided_batched_mul!),
        C::AbstractArray{<:Any, 3}, A::AbstractArray{<:Any, 3}, B::AbstractArray{<:Any, 3},
        ::AbstractAlgorithm
    )
    m, p1, batch_a = size(A)
    p2, n, batch_b = size(B)
    mc, nc, batch_c = size(C)
    p1 == p2 || throw(DimensionMismatch(lazy"Inner dimensions of A ($p1) and B ($p2) must match"))
    batch_a == batch_b || throw(DimensionMismatch(lazy"Batch sizes of A ($batch_a) and B ($batch_b) must match"))
    mc == m || throw(DimensionMismatch(lazy"Output rows ($mc) must match A rows ($m)"))
    nc == n || throw(DimensionMismatch(lazy"Output cols ($nc) must match B cols ($n)"))
    batch_c == batch_a || throw(DimensionMismatch(lazy"Output batch ($batch_c) must match input batch ($batch_a)"))
    return nothing
end

function strided_batched_mul!(
        C::AbstractArray{<:Any, 3}, A::AbstractArray{<:Any, 3}, B::AbstractArray{<:Any, 3},
        alpha::Number, beta::Number, alg::LoopGEMM
    )
    check_input(strided_batched_mul!, C, A, B, alg)
    for k in axes(A, 3)
        mul!(view(C, :, :, k), view(A, :, :, k), view(B, :, :, k), alpha, beta)
    end
    return C
end

function strided_batched_mul!(
        C::AbstractArray{T, 3}, A::AbstractArray{T, 3}, B::AbstractArray{T, 3},
        alpha::Number, beta::Number, alg::GEMM
    ) where {T <: BlasFloat}
    check_input(strided_batched_mul!, C, A, B, alg)
    YABLAS.gemm_strided_batched!('N', 'N', T(alpha), A, B, T(beta), C)
    return C
end
