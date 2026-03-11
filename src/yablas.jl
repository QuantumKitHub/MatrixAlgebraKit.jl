"""
    module YABLAS

Yet Another BLAS wrapper.
This module contains bindings for calling BLAS batched matrix multiplication
functionality from Julia using the CBLAS interface.

`gemm_batch_strided!` wraps `cblas_?gemm_batch_strided`, available in both
OpenBLAS and MKL. `gemm_batch!` wraps `cblas_?gemm_batch`, which is only
available in MKL (not standard OpenBLAS) and will error if called otherwise.
"""
module YABLAS

using LinearAlgebra: BlasFloat
using LinearAlgebra.BLAS: libblastrampoline

# type aliases consistent with YALAPACK
const BlasMat{T <: BlasFloat} = StridedMatrix{T}
const BlasArr3{T <: BlasFloat} = StridedArray{T, 3}

# CBLAS layout/transpose constants
const CblasColMajor = Int32(102)
const CblasNoTrans = Int32(111)

# =============================================================================
# gemm_batch_strided!
# Compute C[:,:,k] = alpha * A[:,:,k] * B[:,:,k] + beta * C[:,:,k] for k = 1:batch
# via cblas_?gemm_batch_strided (available in both OpenBLAS and MKL)
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
    alpha_arg = is_real ? :alpha : :(Ref(alpha))
    beta_arg = is_real ? :beta : :(Ref(beta))
    @eval begin
        function gemm_batch_strided!(
                C::AbstractArray{$elty, 3}, A::AbstractArray{$elty, 3}, B::AbstractArray{$elty, 3},
                alpha::$elty, beta::$elty
            )
            m, k, batch = size(A)
            n = size(B, 2)
            ccall(
                ($(QuoteNode(fname)), libblastrampoline), Cvoid,
                (
                    Cint, Cint, Cint, Cint, Cint, Cint, $alpha_ctype,
                    Ptr{$elty}, Cint, Clonglong,
                    Ptr{$elty}, Cint, Clonglong,
                    $alpha_ctype, Ptr{$elty}, Cint, Clonglong,
                    Cint,
                ),
                CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, $alpha_arg,
                A, stride(A, 2), stride(A, 3),
                B, stride(B, 2), stride(B, 3),
                $beta_arg, C, stride(C, 2), stride(C, 3),
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
                alphas::AbstractVector{$elty}, betas::AbstractVector{$elty}
            )
            batch = length(As)
            transa = fill(CblasNoTrans, batch)
            m_arr = Cint[size(As[k], 1) for k in 1:batch]
            n_arr = Cint[size(Bs[k], 2) for k in 1:batch]
            k_arr = Cint[size(As[k], 2) for k in 1:batch]
            lda_arr = Cint[stride(As[k], 2) for k in 1:batch]
            ldb_arr = Cint[stride(Bs[k], 2) for k in 1:batch]
            ldc_arr = Cint[stride(Cs[k], 2) for k in 1:batch]
            a_ptrs = Ptr{$ptr_elty}[pointer(As[k]) for k in 1:batch]
            b_ptrs = Ptr{$ptr_elty}[pointer(Bs[k]) for k in 1:batch]
            c_ptrs = Ptr{$ptr_elty}[pointer(Cs[k]) for k in 1:batch]
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
                CblasColMajor, transa, transa,
                m_arr, n_arr, k_arr,
                alphas, a_ptrs, lda_arr,
                b_ptrs, ldb_arr,
                betas, c_ptrs, ldc_arr,
                batch, ones(Cint, batch)
            )
            return Cs
        end
    end
end

end
