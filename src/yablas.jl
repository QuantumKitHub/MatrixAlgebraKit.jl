"""
    module YABLAS

Yet Another BLAS wrapper.
This module contains bindings for calling BLAS batched matrix multiplication
functionality from Julia using the CBLAS interface.

`gemm_batch_strided!` wraps `cblas_?gemm_batch_strided`, available in both
OpenBLAS and MKL. `gemm_batch!` wraps `cblas_?gemm_batch`, which is only
available in MKL (not standard OpenBLAS) and will error if called otherwise.

Both functions accept a `layout` keyword argument (`CblasColMajor` or
`CblasRowMajor`) to select storage order. For `CblasColMajor` (default), a
batched m×k matrix A is stored as a Julia array of size `(m, k, batch)`. For
`CblasRowMajor`, the same m×k matrix is stored as `(k, m, batch)` (i.e. rows
are the contiguous dimension). In both cases `stride(X, 1) == 1` is required.
"""
module YABLAS

using LinearAlgebra: BlasFloat, chkstride1, require_one_based_indexing

using LinearAlgebra.BLAS: libblastrampoline

# type aliases consistent with YALAPACK
const BlasMat{T <: BlasFloat} = StridedMatrix{T}
const BlasArr3{T <: BlasFloat} = StridedArray{T, 3}

# CBLAS layout/transpose constants
const CblasRowMajor = Int32(101)
const CblasColMajor = Int32(102)
const CblasNoTrans = Int32(111)

# =============================================================================
# Internal helpers
# =============================================================================

function _chkgemm_dims_colmajor(
        C::AbstractArray{<:Any, 3}, A::AbstractArray{<:Any, 3}, B::AbstractArray{<:Any, 3}
    )
    # col-major: A is (m, k, batch), B is (k, n, batch), C is (m, n, batch)
    m, k, bA = size(A)
    kB, n, bB = size(B)
    mC, nC, bC = size(C)
    k == kB || throw(DimensionMismatch(lazy"inner dimensions of A ($k) and B ($kB) must match"))
    m == mC || throw(DimensionMismatch(lazy"row count of A ($m) and C ($mC) must match"))
    n == nC || throw(DimensionMismatch(lazy"column count of B ($n) and C ($nC) must match"))
    bA == bB == bC || throw(DimensionMismatch(lazy"batch sizes of A ($bA), B ($bB), and C ($bC) must match"))
    return m, n, k, bA
end

function _chkgemm_dims_rowmajor(
        C::AbstractArray{<:Any, 3}, A::AbstractArray{<:Any, 3}, B::AbstractArray{<:Any, 3}
    )
    # row-major: A is (k, m, batch), B is (n, k, batch), C is (n, m, batch)
    k, m, bA = size(A)
    n, kB, bB = size(B)
    nC, mC, bC = size(C)
    k == kB || throw(DimensionMismatch(lazy"inner dimensions of A ($k) and B ($kB) must match"))
    m == mC || throw(DimensionMismatch(lazy"row count of A ($m) and C ($mC) must match"))
    n == nC || throw(DimensionMismatch(lazy"column count of B ($n) and C ($nC) must match"))
    bA == bB == bC || throw(DimensionMismatch(lazy"batch sizes of A ($bA), B ($bB), and C ($bC) must match"))
    return m, n, k, bA
end

function _chkgemm_dims_mat_colmajor(
        C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix
    )
    m, k = size(A)
    kB, n = size(B)
    mC, nC = size(C)
    k == kB || throw(DimensionMismatch(lazy"inner dimensions of A ($k) and B ($kB) must match"))
    m == mC || throw(DimensionMismatch(lazy"row count of A ($m) and C ($mC) must match"))
    n == nC || throw(DimensionMismatch(lazy"column count of B ($n) and C ($nC) must match"))
    return m, n, k
end

function _chkgemm_dims_mat_rowmajor(
        C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix
    )
    k, m = size(A)
    n, kB = size(B)
    nC, mC = size(C)
    k == kB || throw(DimensionMismatch(lazy"inner dimensions of A ($k) and B ($kB) must match"))
    m == mC || throw(DimensionMismatch(lazy"row count of A ($m) and C ($mC) must match"))
    n == nC || throw(DimensionMismatch(lazy"column count of B ($n) and C ($nC) must match"))
    return m, n, k
end

# =============================================================================
# gemm_batch_strided!
# Compute C[:,:,k] = alpha * A[:,:,k] * B[:,:,k] + beta * C[:,:,k] for k = 1:batch
# via cblas_?gemm_batch_strided (available in both OpenBLAS and MKL)
#
# layout = CblasColMajor (default): A is (m, k, batch), B is (k, n, batch),
#   C is (m, n, batch); stride(X, 1) == 1 required.
# layout = CblasRowMajor: A is (k, m, batch), B is (n, k, batch),
#   C is (n, m, batch); stride(X, 1) == 1 required.
#
# Real types pass alpha/beta by value; complex types pass them via Ref (pointer).
# =============================================================================

for (fname, elty) in (
        (:cblas_dgemm_batch_strided, :Float64),
        (:cblas_sgemm_batch_strided, :Float32),
        (:cblas_zgemm_batch_strided, :ComplexF64),
        (:cblas_cgemm_batch_strided, :ComplexF32),
    )
    is_real = elty in (:Float64, :Float32)
    alpha_ctype = is_real ? elty : :(Ptr{$elty})
    alpha_arg = is_real ? :($elty(alpha)) : :(Ref($elty(alpha)))
    beta_arg = is_real ? :($elty(beta)) : :(Ref($elty(beta)))
    @eval begin
        function gemm_batch_strided!(
                C::AbstractArray{$elty, 3}, A::AbstractArray{$elty, 3}, B::AbstractArray{$elty, 3},
                alpha::Union{$elty, Bool}, beta::Union{$elty, Bool};
                layout::Cint = CblasColMajor
            )
            require_one_based_indexing(A, B, C)
            chkstride1(A)
            chkstride1(B)
            chkstride1(C)
            if layout == CblasColMajor
                m, n, k, batch = _chkgemm_dims_colmajor(C, A, B)
            elseif layout == CblasRowMajor
                m, n, k, batch = _chkgemm_dims_rowmajor(C, A, B)
            else
                throw(ArgumentError(lazy"layout must be CblasColMajor ($CblasColMajor) or CblasRowMajor ($CblasRowMajor), got $layout"))
            end
            lda = max(1, stride(A, 2))
            ldb = max(1, stride(B, 2))
            ldc = max(1, stride(C, 2))
            ccall(
                ($(QuoteNode(fname)), libblastrampoline), Cvoid,
                (
                    Cint, Cint, Cint, Cint, Cint, Cint, $alpha_ctype,
                    Ptr{$elty}, Cint, Clonglong,
                    Ptr{$elty}, Cint, Clonglong,
                    $alpha_ctype, Ptr{$elty}, Cint, Clonglong,
                    Cint,
                ),
                layout, CblasNoTrans, CblasNoTrans, m, n, k, $alpha_arg,
                A, lda, stride(A, 3),
                B, ldb, stride(B, 3),
                $beta_arg, C, ldc, stride(C, 3),
                batch
            )
            return C
        end
    end
end

# =============================================================================
# gemm_batch!
# Compute Cs[k] = alpha * As[k] * Bs[k] + beta * Cs[k] for k = 1:batch
# via cblas_?gemm_batch (MKL only — errors if called with OpenBLAS)
#
# layout = CblasColMajor (default): As[k] is m×k, Bs[k] is k×n, Cs[k] is m×n;
#   stride(X, 1) == 1 required for each matrix.
# layout = CblasRowMajor: As[k] is stored (k, m), Bs[k] is stored (n, k),
#   Cs[k] is stored (n, m); stride(X, 1) == 1 required for each matrix.
#
# Real types use typed Ptr{T} for alpha/beta arrays and matrix pointers;
# complex types use Ptr{Cvoid} to match the void* CBLAS interface.
# =============================================================================

for (fname, elty) in (
        (:cblas_dgemm_batch, :Float64),
        (:cblas_sgemm_batch, :Float32),
        (:cblas_zgemm_batch, :ComplexF64),
        (:cblas_cgemm_batch, :ComplexF32),
    )
    is_real = elty in (:Float64, :Float32)
    ptr_elty = is_real ? elty : :Cvoid
    @eval begin
        function gemm_batch!(
                Cs::AbstractVector{<:AbstractMatrix{$elty}},
                As::AbstractVector{<:AbstractMatrix{$elty}},
                Bs::AbstractVector{<:AbstractMatrix{$elty}},
                alphas::AbstractVector{$elty}, betas::AbstractVector{$elty};
                layout::Cint = CblasColMajor
            )
            layout == CblasColMajor || layout == CblasRowMajor ||
                throw(ArgumentError(lazy"layout must be CblasColMajor ($CblasColMajor) or CblasRowMajor ($CblasRowMajor), got $layout"))
            batch = length(As)
            length(Bs) == batch || throw(DimensionMismatch(lazy"length of As ($batch) and Bs ($(length(Bs))) must match"))
            length(Cs) == batch || throw(DimensionMismatch(lazy"length of As ($batch) and Cs ($(length(Cs))) must match"))
            length(alphas) == batch || throw(DimensionMismatch(lazy"length of alphas ($(length(alphas))) must match batch size ($batch)"))
            length(betas) == batch || throw(DimensionMismatch(lazy"length of betas ($(length(betas))) must match batch size ($batch)"))
            transa = fill(CblasNoTrans, batch)
            m_arr = Vector{Cint}(undef, batch)
            n_arr = Vector{Cint}(undef, batch)
            k_arr = Vector{Cint}(undef, batch)
            lda_arr = Vector{Cint}(undef, batch)
            ldb_arr = Vector{Cint}(undef, batch)
            ldc_arr = Vector{Cint}(undef, batch)
            a_ptrs = Vector{Ptr{$ptr_elty}}(undef, batch)
            b_ptrs = Vector{Ptr{$ptr_elty}}(undef, batch)
            c_ptrs = Vector{Ptr{$ptr_elty}}(undef, batch)
            group_sizes = ones(Cint, batch)
            @inbounds for i in 1:batch
                require_one_based_indexing(As[i], Bs[i], Cs[i])
                chkstride1(As[i])
                chkstride1(Bs[i])
                chkstride1(Cs[i])
                if layout == CblasColMajor
                    m, n, k = _chkgemm_dims_mat_colmajor(Cs[i], As[i], Bs[i])
                else
                    m, n, k = _chkgemm_dims_mat_rowmajor(Cs[i], As[i], Bs[i])
                end
                m_arr[i] = m
                n_arr[i] = n
                k_arr[i] = k
                lda_arr[i] = max(1, stride(As[i], 2))
                ldb_arr[i] = max(1, stride(Bs[i], 2))
                ldc_arr[i] = max(1, stride(Cs[i], 2))
                a_ptrs[i] = pointer(As[i])
                b_ptrs[i] = pointer(Bs[i])
                c_ptrs[i] = pointer(Cs[i])
            end
            ccall(
                ($(QuoteNode(fname)), libblastrampoline), Cvoid,
                (
                    Cint, Ptr{Cint}, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                    Ptr{$ptr_elty}, Ptr{Ptr{$ptr_elty}}, Ptr{Cint},
                    Ptr{Ptr{$ptr_elty}}, Ptr{Cint},
                    Ptr{$ptr_elty}, Ptr{Ptr{$ptr_elty}}, Ptr{Cint},
                    Cint, Ptr{Cint},
                ),
                layout, transa, transa,
                m_arr, n_arr, k_arr,
                alphas, a_ptrs, lda_arr,
                b_ptrs, ldb_arr,
                betas, c_ptrs, ldc_arr,
                batch, group_sizes
            )
            return Cs
        end
    end
end

end
