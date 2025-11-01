const IndexRange{T <: Integer} = Base.AbstractRange{T}

# Elementary Householder reflection
struct Householder{T, V <: AbstractVector, R <: IndexRange}
    β::T
    v::V
    r::R
end
Base.adjoint(H::Householder) = Householder(conj(H.β), H.v, H.r)

function householder(x::AbstractVector, r::IndexRange = axes(x, 1), k = first(r))
    i = findfirst(equalto(k), r)
    i == nothing && error("k = $k should be in the range r = $r")
    β, v, ν = _householder!(x[r], i)
    return Householder(β, v, r), ν
end
# Householder reflector h that zeros the elements A[r,col] (except for A[k,col]) upon lmul!(h,A)
function householder(A::AbstractMatrix, r::IndexRange, col::Int, k = first(r))
    i = findfirst(equalto(k), r)
    i == nothing && error("k = $k should be in the range r = $r")
    β, v, ν = _householder!(A[r, col], i)
    return Householder(β, v, r), ν
end
# Householder reflector that zeros the elements A[row,r] (except for A[row,k]) upon rmul!(A,h')
function householder(A::AbstractMatrix, row::Int, r::IndexRange, k = first(r))
    i = findfirst(equalto(k), r)
    i == nothing && error("k = $k should be in the range r = $r")
    β, v, ν = _householder!(conj!(A[row, r]), i)
    return Householder(β, v, r), ν
end

# generate Householder vector based on vector v, such that applying the reflection
# to v yields a vector with single non-zero element on position i, whose value is
# positive and thus equal to norm(v)
function _householder!(v::AbstractVector{T}, i::Int = 1) where {T}
    β::T = zero(T)
    @inbounds begin
        σ = abs2(zero(T))
        @simd for k in 1:(i - 1)
            σ += abs2(v[k])
        end
        @simd for k in (i + 1):length(v)
            σ += abs2(v[k])
        end
        vi = v[i]
        ν = sqrt(abs2(vi) + σ)

        if σ == 0 && vi == ν
            β = zero(vi)
        else
            if real(vi) < 0
                vi = vi - ν
            else
                vi = ((vi - conj(vi)) * ν - σ) / (conj(vi) + ν)
            end
            @simd for k in 1:(i - 1)
                v[k] /= vi
            end
            v[i] = 1
            @simd for k in (i + 1):length(v)
                v[k] /= vi
            end
            β = -conj(vi) / (ν)
        end
    end
    return β, v, ν
end

function LinearAlgebra.lmul!(H::Householder, x::AbstractVector)
    v = H.v
    r = H.r
    β = H.β
    β == 0 && return x
    @inbounds begin
        μ = conj(zero(v[1])) * zero(x[r[1]])
        i = 1
        @simd for j in r
            μ += conj(v[i]) * x[j]
            i += 1
        end
        μ *= β
        i = 1
        @simd for j in H.r
            x[j] -= μ * v[i]
            i += 1
        end
    end
    return x
end
function LinearAlgebra.lmul!(H::Householder, A::AbstractMatrix; cols = axes(A, 2))
    v = H.v
    r = H.r
    β = H.β
    β == 0 && return A
    @inbounds begin
        for k in cols
            μ = conj(zero(v[1])) * zero(A[r[1], k])
            i = 1
            @simd for j in r
                μ += conj(v[i]) * A[j, k]
                i += 1
            end
            μ *= β
            i = 1
            @simd for j in H.r
                A[j, k] -= μ * v[i]
                i += 1
            end
        end
    end
    return A
end
function LinearAlgebra.rmul!(A::AbstractMatrix, H::Householder; rows = axes(A, 1))
    v = H.v
    r = H.r
    β = H.β
    β == 0 && return A
    w = similar(A, length(rows))
    fill!(w, 0)
    all(in(axes(A, 2)), r) || error("Householder range r = $r not compatible with matrix A of size $(size(A))")
    @inbounds begin
        l = 1
        for k in r
            j = 1
            @simd for i in rows
                w[j] += A[i, k] * v[l]
                j += 1
            end
            l += 1
        end
        l = 1
        for k in r
            j = 1
            @simd for i in rows
                A[i, k] -= β * w[j] * conj(v[l])
                j += 1
            end
            l += 1
        end
    end
    return A
end
