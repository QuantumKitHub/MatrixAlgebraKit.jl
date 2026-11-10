"""
    lq_pushforward!(
        ΔA, A, LQ, ΔLQ;
        rank_atol::Real = default_pullback_rank_atol(LQ[1])
    )

Compute the pushforward `ΔLQ` of the LQ decomposition `LQ` of `lq_compact(A;
positive = true)` or `lq_full(A; positive = true)` given the tangent `ΔA` of `A`.

If the original matrix `A` is rank-deficient (rank `r < min(size(A)...)`), only the first `r`
rows of `Q` and the first `r` columns of `L` are differentiable, and the tangents of the
remaining rows of `Q` and columns of `L` are set to zero. Similarly, for `lq_full` the extra
rows of `Q` are only determined up to a unitary rotation, and only their gauge-invariant
tangent component along the first `r` rows of `Q` is computed.

The tangent `ΔQ` may be the same array as `ΔA` (as happens for an in-place `lq_compact!`
with `Q === A`), in which case `ΔA` is overwritten.

See also [`qr_pushforward!`](@ref).
"""
function lq_pushforward!(
        ΔA, A, LQ, ΔLQ;
        rank_atol::Real = default_pullback_rank_atol(LQ[1]), kwargs...
    )
    # TODO: Support L-only `right_orth!` with an empty Q output; this rule currently requires Q.
    L, Q = LQ
    isempty(Q) && !isempty(L) &&
        throw(ArgumentError("LQ differentiation with an omitted Q output is not supported; compute both L and Q."))
    ΔL, ΔQ = ΔLQ
    m = size(L, 1)
    n = size(Q, 2)
    minmn = min(m, n)
    p = lq_rank(L; rank_atol)
    (m, n) == size(ΔA) || throw(DimensionMismatch("size of ΔA ($(size(ΔA))) does not match size of L*Q ($m, $n)"))

    if Base.mightalias(ΔA, ΔL) || (ΔQ !== ΔA && Base.mightalias(ΔA, ΔQ))
        ΔA = copy(ΔA)
    end

    Q₁ = view(Q, 1:p, :)
    # Julia 1.13 `ldiv!` checks `istriu` on the parent, which
    # falls back to scalar indexing for a view of a GPU array
    L₁₁ = LowerTriangular(L[1:p, 1:p])
    L₂₁ = view(L, (p + 1):m, 1:p)

    ΔQ₁ = view(ΔQ, 1:p, :)
    ΔL₁₁ = view(ΔL, 1:p, 1:p)
    ΔL₂₁ = view(ΔL, (p + 1):m, 1:p)

    zero!(ΔL)
    mul!(ΔL₂₁, view(ΔA, (p + 1):m, :), Q₁')

    ΔQ === ΔA || copyto!(ΔQ₁, view(ΔA, 1:p, :))
    ldiv!(L₁₁, ΔQ₁)

    ΔQ₁Q₁ᴴ = ΔQ₁ * Q₁'
    M = ΔQ₁Q₁ᴴ + ΔQ₁Q₁ᴴ'
    diagview(M) ./= 2
    view(M, uppertriangularind(M)) .= zero(eltype(M))
    mul!(ΔL₁₁, L₁₁, M)
    mul!(ΔQ₁, M, Q₁, -1, 1)
    mul!(ΔL₂₁, L₂₁, mul!(ΔQ₁Q₁ᴴ, ΔQ₁, Q₁'), -1, 1)

    zero!(view(ΔQ, (p + 1):size(ΔQ, 1), :))
    if p == minmn && size(Q, 1) > minmn
        Q₃ = view(Q, (minmn + 1):size(Q, 1), :)
        ΔQ₃ = view(ΔQ, (minmn + 1):size(Q, 1), :)
        mul!(ΔQ₃, Q₃ * ΔQ₁', Q₁, -1, 0)
    end
    return ΔL, ΔQ
end

function lq_pushforward!(
        ΔA::Diagonal, A, LQ::Tuple{Diagonal, Diagonal}, ΔLQ::Tuple{Diagonal, Diagonal};
        rank_atol::Real = default_pullback_rank_atol(LQ[1]), kwargs...
    )
    L, Q = LQ
    isempty(Q) && !isempty(L) &&
        throw(ArgumentError("LQ differentiation with an omitted Q output is not supported; compute both L and Q."))
    ΔL, ΔQ = ΔLQ
    size(ΔA) == size(Q) || throw(DimensionMismatch("size of ΔA ($(size(ΔA))) does not match size of L*Q $(size(Q))"))
    p = lq_rank(L; rank_atol)
    _diagonal_qr_lq_pushforward!(diagview(ΔA), diagview(Q), diagview(L), diagview(ΔQ), diagview(ΔL), p)
    return ΔL, ΔQ
end


"""
    lq_null_pushforward!(ΔA, A, Nᴴ, ΔNᴴ; kwargs...)

Compute the pushforward `ΔNᴴ` of the left nullspace basis `Nᴴ` of `lq_null(A)`
given the tangent `ΔA` of `A`.

See also [`lq_pushforward!`](@ref).
"""
function lq_null_pushforward!(ΔA, A, Nᴴ, ΔNᴴ; kwargs...)
    if size(Nᴴ, 1) == 0
        zero!(ΔNᴴ)
        return ΔNᴴ
    end
    L, Q = lq_compact(A; positive = true)
    ΔQNᴴ = ldiv!(LowerTriangular(L), ΔA * Nᴴ')
    return mul!(ΔNᴴ, ΔQNᴴ', Q, -1, 0)
end
