"""
    module YABLAS

Yet Another BLAS wrapper.
This module contains bindings for calling BLAS batched matrix multiplication
functionality from Julia using the CBLAS interface.

`gemm_strided_batched!` wraps `cblas_?gemm_batch_strided` and `gemm_batched!`
wraps `cblas_?gemm_batch`. Both are MKL-only and will error if called with
a BLAS that does not support them (including standard OpenBLAS).

Both functions accept a `layout` keyword argument (`CblasColMajor` or
`CblasRowMajor`) to select storage order. For `CblasColMajor` (default), a
batched m×k matrix A is stored as a Julia array of size `(m, k, batch)`. For
`CblasRowMajor`, the same m×k matrix is stored as `(k, m, batch)` (i.e. rows
are the contiguous dimension). In both cases `stride(X, 1) == 1` is required.
"""
module YABLAS

using LinearAlgebra: BlasFloat, chkstride1, require_one_based_indexing, Adjoint, Transpose

using LinearAlgebra.BLAS: libblastrampoline

# CBLAS layout/transpose constants
const CblasRowMajor = Int32(101)
const CblasColMajor = Int32(102)
const CblasNoTrans = Int32(111)
const CblasTrans = Int32(112)
const CblasConjTrans = Int32(113)

# Convert a BLAS transpose Char to its CBLAS Cint constant
function _char_to_cblas_trans(c::Char)
    c == 'N' && return CblasNoTrans
    c == 'T' && return CblasTrans
    c == 'C' && return CblasConjTrans
    throw(ArgumentError(lazy"transpose character must be 'N', 'T', or 'C', got '$c'"))
end

# Map Julia wrappers to BLAS transpose characters ('N', 'T', 'C')
_trans_char(::AbstractMatrix) = 'N'
_trans_char(::Adjoint) = 'C'
_trans_char(::Transpose) = 'T'

# Extract the underlying matrix from a possibly-wrapped input
_cblas_parent(A::AbstractMatrix) = A
_cblas_parent(A::Union{Adjoint, Transpose}) = parent(A)

# Validate that layout is one of the two supported CBLAS constants.
function _check_layout(layout::Cint)
    layout == CblasColMajor || layout == CblasRowMajor ||
        throw(ArgumentError(lazy"layout must be CblasColMajor ($CblasColMajor) or CblasRowMajor ($CblasRowMajor), got $layout"))
    return nothing
end

# Infer mathematical M, N, K from stored sizes and trans characters, then check
# that the inner dimensions of A and B are compatible.
# Col-major: A stored (M,K) if notransA else (K,M); B stored (K,N) if notransB else (N,K).
# Row-major: A stored (K,M) if notransA else (M,K); B stored (N,K) if notransB else (K,N).
function _gemm_dims(layout::Cint, transa::Cint, transb::Cint, A, B)
    notransA = transa == CblasNoTrans
    notransB = transb == CblasNoTrans
    if layout == CblasColMajor
        m = notransA ? size(A, 1) : size(A, 2)
        k = notransA ? size(A, 2) : size(A, 1)
        n = notransB ? size(B, 2) : size(B, 1)
        kb = notransB ? size(B, 1) : size(B, 2)
    else
        m = notransA ? size(A, 2) : size(A, 1)
        k = notransA ? size(A, 1) : size(A, 2)
        n = notransB ? size(B, 1) : size(B, 2)
        kb = notransB ? size(B, 2) : size(B, 1)
    end
    k == kb || throw(DimensionMismatch(lazy"inner dimensions of A ($k) and B ($kb) must match"))
    return m, n, k
end

# Check that the first two dimensions of C match the expected output size.
# Col-major: C is (M, N); row-major: C is (N, M).
function _check_output_size(layout::Cint, m::Int, n::Int, C)
    r, c = layout == CblasColMajor ? (m, n) : (n, m)
    size(C, 1) == r && size(C, 2) == c ||
        throw(DimensionMismatch(lazy"C has size ($(size(C,1)), $(size(C,2))), expected ($r, $c)"))
    return nothing
end

@doc """
    gemm_strided_batched!(transA, transB, alpha, A, B, beta, C; layout=CblasColMajor)

Compute the strided batched matrix product
`C[:,:,k] = alpha * op(A[:,:,k]) * op(B[:,:,k]) + beta * C[:,:,k]` for all `k`,
where `op(X)` is determined by the transpose character:
`'N'` → identity, `'T'` → transpose, `'C'` → conjugate transpose.

# Arguments
- `transA`, `transB`: transpose character for `A` and `B` (`'N'`, `'T'`, or `'C'`).
- `alpha`, `beta`: scalar coefficients.
- `A`, `B`, `C`: 3D arrays with `stride(X, 1) == 1`.
- `layout`: `CblasColMajor` (default) or `CblasRowMajor`.

For `CblasColMajor` with `transA = 'N'`, `A` is stored as `(m, k, batch)`.
With `transA = 'T'` or `'C'`, `A` is stored as `(k, m, batch)`.

!!! warning
    Wraps `cblas_?gemm_batch_strided` and requires MKL; errors with standard OpenBLAS.
""" gemm_strided_batched!

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
        function gemm_strided_batched!(
                transA::Char, transB::Char,
                alpha::$elty,
                A::AbstractArray{$elty, 3}, B::AbstractArray{$elty, 3},
                beta::$elty,
                C::AbstractArray{$elty, 3};
                layout::Cint = CblasColMajor
            )
            require_one_based_indexing(A, B, C)
            chkstride1(A)
            chkstride1(B)
            chkstride1(C)
            _check_layout(layout)
            transa_int = _char_to_cblas_trans(transA)
            transb_int = _char_to_cblas_trans(transB)
            m, n, k = _gemm_dims(layout, transa_int, transb_int, A, B)
            batch = size(A, 3)
            batch == size(B, 3) || throw(DimensionMismatch(lazy"batch sizes of A ($batch) and B ($(size(B,3))) must match"))
            size(C, 3) == batch || throw(DimensionMismatch(lazy"batch size of C ($(size(C,3))) must match A ($batch)"))
            _check_output_size(layout, m, n, C)
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
                layout, transa_int, transb_int, m, n, k, $alpha_arg,
                A, lda, stride(A, 3),
                B, ldb, stride(B, 3),
                $beta_arg, C, ldc, stride(C, 3),
                batch
            )
            return C
        end
    end
end

@doc """
    gemm_batched!(transA, transB, alpha, A, B, beta, C; layout=CblasColMajor)

Compute the batched matrix product
`C[k] = alpha * op(A[k]) * op(B[k]) + beta * C[k]` for all `k`,
where `op(X)` is determined by the transpose character:
`'N'` → identity, `'T'` → transpose, `'C'` → conjugate transpose.


## Arguments
- `transA`, `transB`: transpose character for `A` and `B` (`'N'`, `'T'`, or `'C'`).
- `alpha`, `beta`: scalar coefficients.
- `A`, `B`, `C`: vectors of matrices with `stride(X[k], 1) == 1` for all `k`.
- `layout`: `CblasColMajor` (default) or `CblasRowMajor`.

For `CblasColMajor` with `transA = 'N'`, `A[k]` is stored as an `(m, k)` matrix (first index contiguous).
With `transA = 'T'` or `'C'`, `A[k]` is stored as `(k, m)`.

!!! warning
    Wraps `cblas_?gemm_batch` and requires MKL; errors with standard OpenBLAS.
""" gemm_batched!

for (fname, elty) in (
        (:cblas_dgemm_batch, :Float64),
        (:cblas_sgemm_batch, :Float32),
        (:cblas_zgemm_batch, :ComplexF64),
        (:cblas_cgemm_batch, :ComplexF32),
    )
    is_real = elty in (:Float64, :Float32)
    ptr_elty = is_real ? elty : :Cvoid
    @eval begin
        function gemm_batched!(
                transA::Char, transB::Char,
                alpha::$elty,
                A::AbstractVector{<:AbstractMatrix{$elty}},
                B::AbstractVector{<:AbstractMatrix{$elty}},
                beta::$elty,
                C::AbstractVector{<:AbstractMatrix{$elty}};
                layout::Cint = CblasColMajor
            )
            _check_layout(layout)
            transa_int = _char_to_cblas_trans(transA)
            transb_int = _char_to_cblas_trans(transB)
            batch = length(A)
            length(B) == batch ||
                throw(DimensionMismatch(lazy"length of A ($batch) and B ($(length(B))) must match"))
            length(C) == batch ||
                throw(DimensionMismatch(lazy"length of A ($batch) and C ($(length(C))) must match"))

            # Bundle all same-type arrays into single allocations to reduce heap pressure.
            # int_buf layout:    [transa | transb | m | n | k | lda | ldb | ldc | group_sizes]
            # scalar_buf layout: [alpha | beta]
            # ptr_buf layout:    [a_ptrs | b_ptrs | c_ptrs]
            int_buf = Vector{Cint}(undef, 9 * batch)
            scalar_buf = Vector{$elty}(undef, 2 * batch)
            ptr_buf = Vector{Ptr{$ptr_elty}}(undef, 3 * batch)

            @inbounds for i in 1:batch
                Ai = _cblas_parent(A[i])
                Bi = _cblas_parent(B[i])
                require_one_based_indexing(Ai, Bi, C[i])
                chkstride1(Ai); chkstride1(Bi); chkstride1(C[i])
                # _gemm_dims and lda must use the parent (untrансposed) storage
                m, n, k = _gemm_dims(layout, transa_int, transb_int, Ai, Bi)
                _check_output_size(layout, m, n, C[i])
                int_buf[i] = transa_int
                int_buf[batch + i] = transb_int
                int_buf[2 * batch + i] = m
                int_buf[3 * batch + i] = n
                int_buf[4 * batch + i] = k
                int_buf[5 * batch + i] = max(1, stride(Ai, 2))
                int_buf[6 * batch + i] = max(1, stride(Bi, 2))
                int_buf[7 * batch + i] = max(1, stride(C[i], 2))
                int_buf[8 * batch + i] = 1
                scalar_buf[i] = alpha
                scalar_buf[batch + i] = beta
                ptr_buf[i] = pointer(Ai)
                ptr_buf[batch + i] = pointer(Bi)
                ptr_buf[2 * batch + i] = pointer(C[i])
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
                layout,
                pointer(int_buf), pointer(int_buf, batch + 1),
                pointer(int_buf, 2 * batch + 1), pointer(int_buf, 3 * batch + 1), pointer(int_buf, 4 * batch + 1),
                Ptr{$ptr_elty}(pointer(scalar_buf)), pointer(ptr_buf), pointer(int_buf, 5 * batch + 1),
                pointer(ptr_buf, batch + 1), pointer(int_buf, 6 * batch + 1),
                Ptr{$ptr_elty}(pointer(scalar_buf, batch + 1)), pointer(ptr_buf, 2 * batch + 1), pointer(int_buf, 7 * batch + 1),
                batch, pointer(int_buf, 8 * batch + 1)
            )
            return C
        end
    end
end

@doc """
    gemm_grouped_batched!(transA, transB, alpha, A, B, beta, C; layout=CblasColMajor)

Compute the grouped batched matrix product
`C[g][k] = alpha[g] * op(A[g][k]) * op(B[g][k]) + beta[g] * C[g][k]`
for all groups `g` and batch indices `k` within each group,
where `op(X)` is determined by the transpose character:
`'N'` → identity, `'T'` → transpose, `'C'` → conjugate transpose.

Within each group all matrices must have the same dimensions and leading dimensions.
Different groups may have different dimensions, transpose characters, and scaling factors.

# Arguments
- `transA`, `transB`: vectors of transpose characters, one per group.
- `alpha`, `beta`: vectors of scalar coefficients, one per group.
- `A`, `B`, `C`: vectors of matrix groups; `A[g]` is a vector of matrices for group `g`,
  all satisfying `stride(X, 1) == 1`.
- `layout`: `CblasColMajor` (default) or `CblasRowMajor`.

!!! warning
    Wraps `cblas_?gemm_batch` and requires MKL; errors with standard OpenBLAS.
""" gemm_grouped_batched!

for (fname, elty) in (
        (:cblas_dgemm_batch, :Float64),
        (:cblas_sgemm_batch, :Float32),
        (:cblas_zgemm_batch, :ComplexF64),
        (:cblas_cgemm_batch, :ComplexF32),
    )
    is_real = elty in (:Float64, :Float32)
    ptr_elty = is_real ? elty : :Cvoid
    @eval begin
        function gemm_grouped_batched!(
                transA::AbstractVector{Char}, transB::AbstractVector{Char},
                alpha::AbstractVector{$elty},
                A::AbstractVector{<:AbstractVector{<:AbstractMatrix{$elty}}},
                B::AbstractVector{<:AbstractVector{<:AbstractMatrix{$elty}}},
                beta::AbstractVector{$elty},
                C::AbstractVector{<:AbstractVector{<:AbstractMatrix{$elty}}};
                layout::Cint = CblasColMajor
            )
            _check_layout(layout)
            ngroups = length(A)
            length(B) == ngroups ||
                throw(DimensionMismatch(lazy"length of A ($ngroups) and B ($(length(B))) must match"))
            length(C) == ngroups ||
                throw(DimensionMismatch(lazy"length of A ($ngroups) and C ($(length(C))) must match"))
            length(transA) == ngroups ||
                throw(DimensionMismatch(lazy"length of transA ($(length(transA))) must match number of groups ($ngroups)"))
            length(transB) == ngroups ||
                throw(DimensionMismatch(lazy"length of transB ($(length(transB))) must match number of groups ($ngroups)"))
            length(alpha) == ngroups ||
                throw(DimensionMismatch(lazy"length of alpha ($(length(alpha))) must match number of groups ($ngroups)"))
            length(beta) == ngroups ||
                throw(DimensionMismatch(lazy"length of beta ($(length(beta))) must match number of groups ($ngroups)"))

            total_batch = sum(length, A)

            transa_arr = Vector{Cint}(undef, ngroups)
            transb_arr = Vector{Cint}(undef, ngroups)
            m_arr = Vector{Cint}(undef, ngroups)
            n_arr = Vector{Cint}(undef, ngroups)
            k_arr = Vector{Cint}(undef, ngroups)
            lda_arr = Vector{Cint}(undef, ngroups)
            ldb_arr = Vector{Cint}(undef, ngroups)
            ldc_arr = Vector{Cint}(undef, ngroups)
            group_sizes = Vector{Cint}(undef, ngroups)
            a_ptrs = Vector{Ptr{$ptr_elty}}(undef, total_batch)
            b_ptrs = Vector{Ptr{$ptr_elty}}(undef, total_batch)
            c_ptrs = Vector{Ptr{$ptr_elty}}(undef, total_batch)

            ptr_offset = 0
            @inbounds for g in 1:ngroups
                Ag = A[g]; Bg = B[g]; Cg = C[g]
                ng = length(Ag)
                length(Bg) == ng ||
                    throw(DimensionMismatch(lazy"group $g: length of A[$g] ($ng) and B[$g] ($(length(Bg))) must match"))
                length(Cg) == ng ||
                    throw(DimensionMismatch(lazy"group $g: length of A[$g] ($ng) and C[$g] ($(length(Cg))) must match"))
                ng > 0 || throw(ArgumentError(lazy"group $g is empty"))

                transa_int = _char_to_cblas_trans(transA[g])
                transb_int = _char_to_cblas_trans(transB[g])
                transa_arr[g] = transa_int
                transb_arr[g] = transb_int
                group_sizes[g] = ng

                # Infer M, N, K from the first matrix in the group; validate all
                # subsequent matrices match its size and leading dimensions.
                A1 = Ag[1]; B1 = Bg[1]; C1 = Cg[1]
                m, n, k = _gemm_dims(layout, transa_int, transb_int, A1, B1)
                _check_output_size(layout, m, n, C1)
                m_arr[g] = m; n_arr[g] = n; k_arr[g] = k
                lda = max(1, stride(A1, 2))
                ldb = max(1, stride(B1, 2))
                ldc = max(1, stride(C1, 2))
                lda_arr[g] = lda; ldb_arr[g] = ldb; ldc_arr[g] = ldc

                for i in 1:ng
                    Ai = Ag[i]; Bi = Bg[i]; Ci = Cg[i]
                    require_one_based_indexing(Ai, Bi, Ci)
                    chkstride1(Ai); chkstride1(Bi); chkstride1(Ci)
                    size(Ai) == size(A1) ||
                        throw(DimensionMismatch(lazy"group $g: size of A[$g][$i] ($(size(Ai))) differs from A[$g][1] ($(size(A1)))"))
                    size(Bi) == size(B1) ||
                        throw(DimensionMismatch(lazy"group $g: size of B[$g][$i] ($(size(Bi))) differs from B[$g][1] ($(size(B1)))"))
                    size(Ci) == size(C1) ||
                        throw(DimensionMismatch(lazy"group $g: size of C[$g][$i] ($(size(Ci))) differs from C[$g][1] ($(size(C1)))"))
                    stride(Ai, 2) == stride(A1, 2) ||
                        throw(ArgumentError(lazy"group $g: all A matrices must have the same leading dimension"))
                    stride(Bi, 2) == stride(B1, 2) ||
                        throw(ArgumentError(lazy"group $g: all B matrices must have the same leading dimension"))
                    stride(Ci, 2) == stride(C1, 2) ||
                        throw(ArgumentError(lazy"group $g: all C matrices must have the same leading dimension"))
                    a_ptrs[ptr_offset + i] = pointer(Ai)
                    b_ptrs[ptr_offset + i] = pointer(Bi)
                    c_ptrs[ptr_offset + i] = pointer(Ci)
                end
                ptr_offset += ng
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
                layout, transa_arr, transb_arr,
                m_arr, n_arr, k_arr,
                alpha, a_ptrs, lda_arr,
                b_ptrs, ldb_arr,
                beta, c_ptrs, ldc_arr,
                ngroups, group_sizes
            )
            return C
        end
    end
end

end
