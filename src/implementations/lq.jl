# Inputs
# ------
function copy_input(::typeof(lq_full), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
function copy_input(::typeof(lq_compact), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
function copy_input(::typeof(lq_null), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end

function check_input(::typeof(lq_full!), A::AbstractMatrix, LQ)
    m, n = size(A)
    L, Q = LQ
    @assert L isa AbstractMatrix && Q isa AbstractMatrix
    isempty(L) || @check_size(L, (m, n))
    @check_scalar(L, A)
    @check_size(Q, (n, n))
    @check_scalar(Q, A)
    return nothing
end
function check_input(::typeof(lq_compact!), A::AbstractMatrix, LQ)
    m, n = size(A)
    minmn = min(m, n)
    L, Q = LQ
    @assert L isa AbstractMatrix && Q isa AbstractMatrix
    isempty(L) || @check_size(L, (m, minmn))
    @check_scalar(L, A)
    @check_size(Q, (minmn, n))
    @check_scalar(Q, A)
    return nothing
end
function check_input(::typeof(lq_null!), A::AbstractMatrix, Nᴴ)
    m, n = size(A)
    minmn = min(m, n)
    @assert Nᴴ isa AbstractMatrix
    @check_size(Nᴴ, (n - minmn, n))
    @check_scalar(Nᴴ, A)
    return nothing
end

# Outputs
# -------
function initialize_output(::typeof(lq_full!), A::AbstractMatrix, ::LAPACK_HouseholderLQ)
    m, n = size(A)
    L = similar(A, (m, n))
    Q = similar(A, (n, n))
    return (L, Q)
end
function initialize_output(::typeof(lq_compact!), A::AbstractMatrix, ::LAPACK_HouseholderLQ)
    m, n = size(A)
    minmn = min(m, n)
    L = similar(A, (m, minmn))
    Q = similar(A, (minmn, n))
    return (L, Q)
end
function initialize_output(::typeof(lq_null!), A::AbstractMatrix, ::LAPACK_HouseholderLQ)
    m, n = size(A)
    minmn = min(m, n)
    Nᴴ = similar(A, (n - minmn, n))
    return Nᴴ
end

# Implementation
# --------------
# actual implementation
function lq_full!(A::AbstractMatrix, LQ, alg::LAPACK_HouseholderLQ)
    check_input(lq_full!, A, LQ)
    L, Q = LQ
    _lapack_lq!(A, L, Q; alg.kwargs...)
    return L, Q
end
function lq_compact!(A::AbstractMatrix, LQ, alg::LAPACK_HouseholderLQ)
    check_input(lq_compact!, A, LQ)
    L, Q = LQ
    _lapack_lq!(A, L, Q; alg.kwargs...)
    return L, Q
end
function lq_null!(A::AbstractMatrix, Nᴴ, alg::LAPACK_HouseholderLQ)
    check_input(lq_null!, A, Nᴴ)
    _lapack_lq_null!(A, Nᴴ; alg.kwargs...)
    return Nᴴ
end

function _lapack_lq!(A::AbstractMatrix, L::AbstractMatrix, Q::AbstractMatrix;
                     positive=false,
                     pivoted=false,
                     blocksize=((pivoted || A === Q) ? 1 : YALAPACK.default_qr_blocksize(A)))
    m, n = size(A)
    minmn = min(m, n)
    computeL = length(L) > 0
    inplaceQ = Q === A

    if pivoted
        throw(ArgumentError("LAPACK does not provide an implementation for a pivoted LQ decomposition"))
    end
    if inplaceQ && (computeL || positive || blocksize > 1 || n < m)
        throw(ArgumentError("inplace Q only supported if matrix is wide (`m <= n`), L is not required, and using the unblocked algorithm (`blocksize=1`) with `positive=false`"))
    end

    if blocksize > 1
        mb = min(minmn, blocksize)
        if computeL # first use L as space for T
            A, T = YALAPACK.gelqt!(A, view(L, 1:mb, 1:minmn))
        else
            A, T = YALAPACK.gelqt!(A, similar(A, mb, minmn))
        end
        Q = YALAPACK.gemlqt!('R', 'N', A, T, one!(Q))
    else
        A, τ = YALAPACK.gelqf!(A)
        if inplaceQ
            Q = YALAPACK.unglq!(A, τ)
        else
            Q = YALAPACK.unmlq!('R', 'N', A, τ, one!(Q))
        end
    end

    if positive # already fix Q even if we do not need R
        @inbounds for j in 1:n
            @simd for i in 1:minmn
                s = sign_safe(A[i, i])
                Q[i, j] *= s
            end
        end
    end

    if computeL
        L̃ = tril!(view(A, axes(L)...))
        if positive
            @inbounds for j in 1:minmn
                s = conj(sign_safe(L̃[j, j]))
                @simd for i in j:m
                    L̃[i, j] = L̃[i, j] * s
                end
            end
        end
        copyto!(L, L̃)
    end
    return L, Q
end

function _lapack_lq_null!(A::AbstractMatrix, Nᴴ::AbstractMatrix;
                          positive=false,
                          pivoted=false,
                          blocksize=YALAPACK.default_qr_blocksize(A))
    m, n = size(A)
    minmn = min(m, n)
    fill!(Nᴴ, zero(eltype(Nᴴ)))
    one!(view(Nᴴ, 1:(n - minmn), (minmn + 1):n))
    if blocksize > 1
        mb = min(minmn, blocksize)
        A, T = YALAPACK.gelqt!(A, similar(A, mb, minmn))
        Nᴴ = YALAPACK.gemlqt!('R', 'N', A, T, Nᴴ)
    else
        A, τ = YALAPACK.gelqf!(A)
        Nᴴ = YALAPACK.unmlq!('R', 'N', A, τ, Nᴴ)
    end
    return Nᴴ
end
