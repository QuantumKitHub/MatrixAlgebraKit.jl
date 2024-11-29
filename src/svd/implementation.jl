struct DivideAndConquer
    full::Bool
end

struct QRIteration
    full::Bool
end

# preallocate output
function SVD(A::AbstractMatrix, alg::Union{DivideAndConquer,QRIteration})
    m, n = size(A)
    k = min(m, n)
    U = similar(A, (m, alg.full ? m : k))
    S = similar(A, real(eltype(A)), k)
    Vᴴ = similar(A, (alg.full ? k : n, n))
    return SVD(U, S, Vᴴ)
end

function check_svd_full_input(A, U, S, Vᴴ)
    m, n = size(A)
    minmn = min(m, n)
    size(U) == (m, m) ||
        throw(DimensionMismatch("`full=true` requires square U matrix with equal number of rows as A"))
    size(Vᴴ) == (n, n) ||
        throw(DimensionMismatch("`full=true` requires square Vᴴ matrix with equal number of columns as A"))
    size(S) == (minmn,) ||
        throw(DimensionMismatch("`full=true` requires vector S of length min(size(A)..."))
    return nothing
end

function check_svd_compact_input(A, U, S, Vᴴ)
    m, n = size(A)
    minmn = min(m, n)
    size(U) == (m, minmn) ||
        throw(DimensionMismatch("`full=false` requires square U matrix with equal number of rows as A"))
    size(Vᴴ) == (minmn, n) ||
        throw(DimensionMismatch("`full=false` requires square Vᴴ matrix with equal number of columns as A"))
    size(S) == (minmn,) ||
        throw(DimensionMismatch("`full=false` requires vector S of length min(size(A)..."))
    return nothing
end

function svd!(F::SVD, A::AbstractMatrix, alg::Union{DivideAndConquer,QRIteration})
    alg.full ? check_svd_full_input(A, F) : check_svd_compact_input(A, F)
    f_lapack = alg isa DivideAndConquer ? YALAPACK.gesdd! : YALAPACK.gesvd!
    f_lapack(A, F.U, getfield(F, :S), F.Vt)
    return F
end
