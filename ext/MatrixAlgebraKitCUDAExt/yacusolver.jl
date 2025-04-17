module YACUSOLVER

using LinearAlgebra
using LinearAlgebra: BlasInt, checksquare, chkstride1, require_one_based_indexing
using LinearAlgebra.LAPACK: chkargsok, chklapackerror, chktrans, chkside, chkdiag, chkuplo

using CUDA
using CUDA.CUSOLVER: cusolverDnCreate

# QR methods are implemented with full access to allocated arrays, so we do not need to redo this:
using CUDA.CUSOLVER: geqrf!, ormqr!, orgqr!
const unmqr! = ormqr!
const ungqr! = orgqr!

# # Wrapper for SVD via QR Iteration
# for (bname, fname, elty, relty) in
#     ((:cusolverDnSgesvd_bufferSize, :cusolverDnSgesvd, :Float32, :Float32),
#      (:cusolverDnDgesvd_bufferSize, :cusolverDnDgesvd, :Float64, :Float64),
#      (:cusolverDnCgesvd_bufferSize, :cusolverDnCgesvd, :ComplexF32, :Float32),
#      (:cusolverDnZgesvd_bufferSize, :cusolverDnZgesvd, :ComplexF64, :Float64))
#     @eval begin
#         function gesvd!(A::StridedCuMatrix{$elty},
#                         S::StridedCuVector{$relty}=similar(A, $relty, min(size(A)...)),
#                         U::StridedCuMatrix{$elty}=similar(A, $elty, size(A, 1),
#                                                           min(size(A)...)),
#                         Vᴴ::StridedCuMatrix{$elty}=similar(A, $elty, min(size(A)...),
#                                                            size(A, 2)))
#             chkstride1(A, U, Vᴴ, S)
#             m, n = size(A)
#             (m < n) && throw(ArgumentError("CUSOLVER's gesvd requires m ≥ n"))
#             minmn = min(m, n)
#             lda = max(1, stride(A, 2))

#             if length(U) == 0
#                 jobu = 'N'
#             else
#                 size(U, 1) == m ||
#                     throw(DimensionMismatch("row size mismatch between A and U"))
#                 if size(U, 2) == minmn
#                     if U === A
#                         jobu = 'O'
#                     else
#                         jobu = 'S'
#                     end
#                 elseif size(U, 2) == m
#                     jobu = 'A'
#                 else
#                     throw(DimensionMismatch("invalid column size of U"))
#                 end
#             end
#             if length(Vᴴ) == 0
#                 jobvt = 'N'
#             else
#                 size(Vᴴ, 2) == n ||
#                     throw(DimensionMismatch("column size mismatch between A and Vᴴ"))
#                 if size(Vᴴ, 1) == minmn
#                     if Vᴴ === A
#                         jobvt = 'O'
#                     else
#                         jobvt = 'S'
#                     end
#                 elseif size(Vᴴ, 1) == n
#                     jobvt = 'A'
#                 else
#                     throw(DimensionMismatch("invalid row size of Vᴴ"))
#                 end
#             end
#             length(S) == minmn ||
#                 throw(DimensionMismatch("length mismatch between A and S"))

#             lda = max(1, stride(A, 2))
#             ldu = max(1, stride(U, 2))
#             ldv = max(1, stride(Vᴴ, 2))

#             dh = dense_handle()
#             function bufferSize()
#                 out = Ref{Cint}(0)
#                 $bname(dh, m, n, out)
#                 return out[] * sizeof($elty)
#             end
#             rwork = CuArray{$relty}(undef, min(m, n) - 1)
#             with_workspace(dh.workspace_gpu, bufferSize) do buffer
#                 return $fname(dh, jobu, jobvt, m, n, A, lda, S, U, ldu, Vᴴ, ldv,
#                               buffer, sizeof(buffer) ÷ sizeof($elty), rwork, dh.info)
#             end
#             unsafe_free!(rwork)

#             info = @allowscalar dh.info[1]
#             chkargsok(BlasInt(info))

#             return (S, U, Vᴴ)
#         end
#     end
# end

# # Wrapper for SVD via Jacobi
# for (bname, fname, elty, relty) in
#     ((:cusolverDnSgesvdj_bufferSize, :cusolverDnSgesvdj, :Float32, :Float32),
#      (:cusolverDnDgesvdj_bufferSize, :cusolverDnDgesvdj, :Float64, :Float64),
#      (:cusolverDnCgesvdj_bufferSize, :cusolverDnCgesvdj, :ComplexF32, :Float32),
#      (:cusolverDnZgesvdj_bufferSize, :cusolverDnZgesvdj, :ComplexF64, :Float64))
#     @eval begin
#         function gesvdj!(A::StridedCuMatrix{$elty},
#                          S::StridedCuVector{$relty}=similar(A, $relty, min(size(A)...)),
#                          U::StridedCuMatrix{$elty}=similar(A, $elty, size(A, 1),
#                                                            min(size(A)...)),
#                          Vᴴ::StridedCuMatrix{$elty}=similar(A, $elty, min(size(A)...),
#                                                             size(A, 2));
#                          tol::$relty=eps($relty),
#                          max_sweeps::Int=100)
#             chkstride1(A, U, Vᴴ, S)
#             m, n = size(A)
#             minmn = min(m, n)
#             lda = max(1, stride(A, 2))

#             if length(U) == 0 && length(Vᴴ) == 0
#                 jobz = 'N'
#                 econ = 0
#             else
#                 jobz = 'V'
#                 size(U, 1) == m ||
#                     throw(DimensionMismatch("row size mismatch between A and U"))
#                 size(Vᴴ, 2) == n ||
#                     throw(DimensionMismatch("column size mismatch between A and Vᴴ"))
#                 if size(U, 2) == size(Vᴴ, 1) == minmn
#                     econ = 1
#                 elseif size(U, 2) == m && size(Vᴴ, 1) == n
#                     econ = 0
#                 else
#                     throw(DimensionMismatch("invalid column size of U or row size of Vᴴ"))
#                 end
#             end
#             length(S) == minmn ||
#                 throw(DimensionMismatch("length mismatch between A and S"))

#             if jobz == 'N' # it seems we still need the memory for U and Vᴴ
#                 U = similar(A, $elty, m, minmn)
#                 V = similar(A, $elty, n, minmn)
#             else
#                 V = similar(Vᴴ')
#             end
#             lda = max(1, stride(A, 2))
#             ldu = max(1, stride(U, 2))
#             ldv = max(1, stride(V, 2))

#             params = Ref{gesvdjInfo_t}(C_NULL)
#             cusolverDnCreateGesvdjInfo(params)
#             cusolverDnXgesvdjSetTolerance(params[], tol)
#             cusolverDnXgesvdjSetMaxSweeps(params[], max_sweeps)
#             dh = dense_handle()

#             function bufferSize()
#                 out = Ref{Cint}(0)
#                 $bname(dh, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
#                        out, params[])
#                 return out[] * sizeof($elty)
#             end

#             with_workspace(dh.workspace_gpu, bufferSize) do buffer
#                 return $fname(dh, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
#                               buffer, sizeof(buffer) ÷ sizeof($elty), dh.info, params[])
#             end

#             info = @allowscalar dh.info[1]
#             chkargsok(BlasInt(info))

#             cusolverDnDestroyGesvdjInfo(params[])

#             if jobz != 'N'
#                 adjoint!(Vᴴ, V)
#             end
#             return U, S, Vᴴ
#         end
#     end
# end

# for (jname, bname, fname, elty, relty) in
#     ((:sygvd!, :cusolverDnSsygvd_bufferSize, :cusolverDnSsygvd, :Float32, :Float32),
#      (:sygvd!, :cusolverDnDsygvd_bufferSize, :cusolverDnDsygvd, :Float64, :Float64),
#      (:hegvd!, :cusolverDnChegvd_bufferSize, :cusolverDnChegvd, :ComplexF32, :Float32),
#      (:hegvd!, :cusolverDnZhegvd_bufferSize, :cusolverDnZhegvd, :ComplexF64, :Float64))
#     @eval begin
#         function $jname(itype::Int,
#                         jobz::Char,
#                         uplo::Char,
#                         A::StridedCuMatrix{$elty},
#                         B::StridedCuMatrix{$elty})
#             chkuplo(uplo)
#             nA, nB = checksquare(A, B)
#             if nB != nA
#                 throw(DimensionMismatch("Dimensions of A ($nA, $nA) and B ($nB, $nB) must match!"))
#             end
#             n = nA
#             lda = max(1, stride(A, 2))
#             ldb = max(1, stride(B, 2))
#             W = CuArray{$relty}(undef, n)
#             dh = dense_handle()

#             function bufferSize()
#                 out = Ref{Cint}(0)
#                 $bname(dh, itype, jobz, uplo, n, A, lda, B, ldb, W, out)
#                 return out[] * sizeof($elty)
#             end

#             with_workspace(dh.workspace_gpu, bufferSize) do buffer
#                 return $fname(dh, itype, jobz, uplo, n, A, lda, B, ldb, W,
#                               buffer, sizeof(buffer) ÷ sizeof($elty), dh.info)
#             end

#             info = @allowscalar dh.info[1]
#             chkargsok(BlasInt(info))

#             if jobz == 'N'
#                 return W
#             elseif jobz == 'V'
#                 return W, A, B
#             end
#         end
#     end
# end

# for (jname, bname, fname, elty, relty) in
#     ((:sygvj!, :cusolverDnSsygvj_bufferSize, :cusolverDnSsygvj, :Float32, :Float32),
#      (:sygvj!, :cusolverDnDsygvj_bufferSize, :cusolverDnDsygvj, :Float64, :Float64),
#      (:hegvj!, :cusolverDnChegvj_bufferSize, :cusolverDnChegvj, :ComplexF32, :Float32),
#      (:hegvj!, :cusolverDnZhegvj_bufferSize, :cusolverDnZhegvj, :ComplexF64, :Float64))
#     @eval begin
#         function $jname(itype::Int,
#                         jobz::Char,
#                         uplo::Char,
#                         A::StridedCuMatrix{$elty},
#                         B::StridedCuMatrix{$elty};
#                         tol::$relty=eps($relty),
#                         max_sweeps::Int=100)
#             chkuplo(uplo)
#             nA, nB = checksquare(A, B)
#             if nB != nA
#                 throw(DimensionMismatch("Dimensions of A ($nA, $nA) and B ($nB, $nB) must match!"))
#             end
#             n = nA
#             lda = max(1, stride(A, 2))
#             ldb = max(1, stride(B, 2))
#             W = CuArray{$relty}(undef, n)
#             params = Ref{syevjInfo_t}(C_NULL)
#             cusolverDnCreateSyevjInfo(params)
#             cusolverDnXsyevjSetTolerance(params[], tol)
#             cusolverDnXsyevjSetMaxSweeps(params[], max_sweeps)
#             dh = dense_handle()

#             function bufferSize()
#                 out = Ref{Cint}(0)
#                 $bname(dh, itype, jobz, uplo, n, A, lda, B, ldb, W,
#                        out, params[])
#                 return out[] * sizeof($elty)
#             end

#             with_workspace(dh.workspace_gpu, bufferSize) do buffer
#                 return $fname(dh, itype, jobz, uplo, n, A, lda, B, ldb, W,
#                               buffer, sizeof(buffer) ÷ sizeof($elty), dh.info, params[])
#             end

#             info = @allowscalar dh.info[1]
#             chkargsok(BlasInt(info))

#             cusolverDnDestroySyevjInfo(params[])

#             if jobz == 'N'
#                 return W
#             elseif jobz == 'V'
#                 return W, A, B
#             end
#         end
#     end
# end

# for (jname, bname, fname, elty, relty) in
#     ((:syevjBatched!, :cusolverDnSsyevjBatched_bufferSize, :cusolverDnSsyevjBatched,
#       :Float32, :Float32),
#      (:syevjBatched!, :cusolverDnDsyevjBatched_bufferSize, :cusolverDnDsyevjBatched,
#       :Float64, :Float64),
#      (:heevjBatched!, :cusolverDnCheevjBatched_bufferSize, :cusolverDnCheevjBatched,
#       :ComplexF32, :Float32),
#      (:heevjBatched!, :cusolverDnZheevjBatched_bufferSize, :cusolverDnZheevjBatched,
#       :ComplexF64, :Float64))
#     @eval begin
#         function $jname(jobz::Char,
#                         uplo::Char,
#                         A::StridedCuArray{$elty};
#                         tol::$relty=eps($relty),
#                         max_sweeps::Int=100)

#             # Set up information for the solver arguments
#             chkuplo(uplo)
#             n = checksquare(A)
#             lda = max(1, stride(A, 2))
#             batchSize = size(A, 3)
#             W = CuArray{$relty}(undef, n, batchSize)
#             params = Ref{syevjInfo_t}(C_NULL)

#             dh = dense_handle()
#             resize!(dh.info, batchSize)

#             # Initialize the solver parameters
#             cusolverDnCreateSyevjInfo(params)
#             cusolverDnXsyevjSetTolerance(params[], tol)
#             cusolverDnXsyevjSetMaxSweeps(params[], max_sweeps)

#             # Calculate the workspace size
#             function bufferSize()
#                 out = Ref{Cint}(0)
#                 $bname(dh, jobz, uplo, n, A, lda, W, out, params[], batchSize)
#                 return out[] * sizeof($elty)
#             end

#             # Run the solver
#             with_workspace(dh.workspace_gpu, bufferSize) do buffer
#                 return $fname(dh, jobz, uplo, n, A, lda, W, buffer,
#                               sizeof(buffer) ÷ sizeof($elty), dh.info, params[], batchSize)
#             end

#             # Copy the solver info and delete the device memory
#             info = @allowscalar collect(dh.info)

#             # Double check the solver's exit status
#             for i in 1:batchSize
#                 chkargsok(BlasInt(info[i]))
#             end

#             cusolverDnDestroySyevjInfo(params[])

#             # Return eigenvalues (in W) and possibly eigenvectors (in A)
#             if jobz == 'N'
#                 return W
#             elseif jobz == 'V'
#                 return W, A
#             end
#         end
#     end
# end

# for (fname, elty) in ((:cusolverDnSpotrsBatched, :Float32),
#                       (:cusolverDnDpotrsBatched, :Float64),
#                       (:cusolverDnCpotrsBatched, :ComplexF32),
#                       (:cusolverDnZpotrsBatched, :ComplexF64))
#     @eval begin
#         function potrsBatched!(uplo::Char,
#                                A::Vector{<:StridedCuMatrix{$elty}},
#                                B::Vector{<:StridedCuVecOrMat{$elty}})
#             if length(A) != length(B)
#                 throw(DimensionMismatch(""))
#             end
#             # Set up information for the solver arguments
#             chkuplo(uplo)
#             n = checksquare(A[1])
#             if size(B[1], 1) != n
#                 throw(DimensionMismatch("first dimension of B[i], $(size(B[1],1)), must match second dimension of A, $n"))
#             end
#             nrhs = size(B[1], 2)
#             # cuSOLVER's Remark 1: only nrhs=1 is supported.
#             if nrhs != 1
#                 throw(ArgumentError("cuSOLVER only supports vectors for B"))
#             end
#             lda = max(1, stride(A[1], 2))
#             ldb = max(1, stride(B[1], 2))
#             batchSize = length(A)

#             Aptrs = unsafe_batch(A)
#             Bptrs = unsafe_batch(B)

#             dh = dense_handle()

#             # Run the solver
#             $fname(dh, uplo, n, nrhs, Aptrs, lda, Bptrs, ldb, dh.info, batchSize)

#             # Copy the solver info and delete the device memory
#             info = @allowscalar dh.info[1]
#             chklapackerror(BlasInt(info))

#             return B
#         end
#     end
# end

# for (fname, elty) in ((:cusolverDnSpotrfBatched, :Float32),
#                       (:cusolverDnDpotrfBatched, :Float64),
#                       (:cusolverDnCpotrfBatched, :ComplexF32),
#                       (:cusolverDnZpotrfBatched, :ComplexF64))
#     @eval begin
#         function potrfBatched!(uplo::Char, A::Vector{<:StridedCuMatrix{$elty}})

#             # Set up information for the solver arguments
#             chkuplo(uplo)
#             n = checksquare(A[1])
#             lda = max(1, stride(A[1], 2))
#             batchSize = length(A)

#             Aptrs = unsafe_batch(A)

#             dh = dense_handle()
#             resize!(dh.info, batchSize)

#             # Run the solver
#             $fname(dh, uplo, n, Aptrs, lda, dh.info, batchSize)

#             # Copy the solver info and delete the device memory
#             info = @allowscalar collect(dh.info)

#             # Double check the solver's exit status
#             for i in 1:batchSize
#                 chkargsok(BlasInt(info[i]))
#             end

#             # info[i] > 0 means the leading minor of order info[i] is not positive definite
#             # LinearAlgebra.LAPACK does not throw Exception here
#             # to simplify calls to isposdef! and factorize
#             return A, info
#         end
#     end
# end

# # gesv
# function gesv!(X::CuVecOrMat{T}, A::CuMatrix{T}, B::CuVecOrMat{T}; fallback::Bool=true,
#                residual_history::Bool=false, irs_precision::String="AUTO",
#                refinement_solver::String="CLASSICAL",
#                maxiters::Int=0, maxiters_inner::Int=0, tol::Float64=0.0,
#                tol_inner=Float64 = 0.0) where {T<:BlasFloat}
#     params = CuSolverIRSParameters()
#     info = CuSolverIRSInformation()
#     n = checksquare(A)
#     nrhs = size(B, 2)
#     lda = max(1, stride(A, 2))
#     ldb = max(1, stride(B, 2))
#     ldx = max(1, stride(X, 2))
#     niters = Ref{Cint}()
#     dh = dense_handle()

#     if irs_precision == "AUTO"
#         (T == Float32) && (irs_precision = "R_32F")
#         (T == Float64) && (irs_precision = "R_64F")
#         (T == ComplexF32) && (irs_precision = "C_32F")
#         (T == ComplexF64) && (irs_precision = "C_64F")
#     else
#         (T == Float32) && (irs_precision ∈ ("R_32F", "R_16F", "R_16BF", "R_TF32") ||
#                            error("$irs_precision is not supported."))
#         (T == Float64) &&
#             (irs_precision ∈ ("R_64F", "R_32F", "R_16F", "R_16BF", "R_TF32") ||
#              error("$irs_precision is not supported."))
#         (T == ComplexF32) && (irs_precision ∈ ("C_32F", "C_16F", "C_16BF", "C_TF32") ||
#                               error("$irs_precision is not supported."))
#         (T == ComplexF64) &&
#             (irs_precision ∈ ("C_64F", "C_32F", "C_16F", "C_16BF", "C_TF32") ||
#              error("$irs_precision is not supported."))
#     end
#     cusolverDnIRSParamsSetSolverMainPrecision(params, T)
#     cusolverDnIRSParamsSetSolverLowestPrecision(params, irs_precision)
#     cusolverDnIRSParamsSetRefinementSolver(params, refinement_solver)
#     (tol != 0.0) && cusolverDnIRSParamsSetTol(params, tol)
#     (tol_inner != 0.0) && cusolverDnIRSParamsSetTolInner(params, tol_inner)
#     (maxiters != 0) && cusolverDnIRSParamsSetMaxIters(params, maxiters)
#     (maxiters_inner != 0) && cusolverDnIRSParamsSetMaxItersInner(params, maxiters_inner)
#     fallback ? cusolverDnIRSParamsEnableFallback(params) :
#     cusolverDnIRSParamsDisableFallback(params)
#     residual_history && cusolverDnIRSInfosRequestResidual(info)

#     function bufferSize()
#         buffer_size = Ref{Csize_t}(0)
#         cusolverDnIRSXgesv_bufferSize(dh, params, n, nrhs, buffer_size)
#         return buffer_size[]
#     end

#     with_workspace(dh.workspace_gpu, bufferSize) do buffer
#         return cusolverDnIRSXgesv(dh, params, info, n, nrhs, A, lda, B, ldb,
#                                   X, ldx, buffer, sizeof(buffer), niters, dh.info)
#     end

#     # Copy the solver flag and delete the device memory
#     flag = @allowscalar dh.info[1]
#     chklapackerror(BlasInt(flag))

#     return X, info
# end

# for (jname, bname, fname, elty, relty) in
#     ((:syevd!, :cusolverDnSsyevd_bufferSize, :cusolverDnSsyevd, :Float32, :Float32),
#      (:syevd!, :cusolverDnDsyevd_bufferSize, :cusolverDnDsyevd, :Float64, :Float64),
#      (:heevd!, :cusolverDnCheevd_bufferSize, :cusolverDnCheevd, :ComplexF32, :Float32),
#      (:heevd!, :cusolverDnZheevd_bufferSize, :cusolverDnZheevd, :ComplexF64, :Float64))
#     @eval begin
#         function $jname(jobz::Char,
#                         uplo::Char,
#                         A::StridedCuMatrix{$elty})
#             chkuplo(uplo)
#             n = checksquare(A)
#             lda = max(1, stride(A, 2))
#             W = CuArray{$relty}(undef, n)
#             dh = dense_handle()

#             function bufferSize()
#                 out = Ref{Cint}(0)
#                 $bname(dh, jobz, uplo, n, A, lda, W, out)
#                 return out[] * sizeof($elty)
#             end

#             with_workspace(dh.workspace_gpu, bufferSize) do buffer
#                 return $fname(dh, jobz, uplo, n, A, lda, W,
#                               buffer, sizeof(buffer) ÷ sizeof($elty), dh.info)
#             end

#             info = @allowscalar dh.info[1]
#             chkargsok(BlasInt(info))

#             if jobz == 'N'
#                 return W
#             elseif jobz == 'V'
#                 return W, A
#             end
#         end
#     end
# end

# Wrapper for Hermitian Eigenvalue Problem
function heevd!(jobz::Char, uplo::Char, A::StridedCuMatrix{T},
                W::StridedCuVector{T}) where {T<:BlasFloat}
    chkuplo(uplo)
    n = checksquare(A)
    lda = max(1, stride(A, 2))
    dh = dense_handle()

    function bufferSize()
        out = Ref{Cint}(0)
        cusolverDnSsyevd_bufferSize(dh, jobz, uplo, n, A, lda, W, out)
        return out[] * sizeof(T)
    end

    with_workspace(dh.workspace_gpu, bufferSize) do buffer
        return cusolverDnSsyevd(dh, jobz, uplo, n, A, lda, W, buffer,
                                sizeof(buffer) ÷ sizeof(T), dh.info)
    end

    info = @allowscalar dh.info[1]
    chkargsok(BlasInt(info))
    return W, A
end

# Wrapper for Non-Hermitian Eigenvalue Problem
function geevd!(jobvl::Char, jobvr::Char, A::StridedCuMatrix{T}, W::StridedCuVector{T},
                VL::StridedCuMatrix{T}, VR::StridedCuMatrix{T}) where {T<:BlasFloat}
    n = checksquare(A)
    lda = max(1, stride(A, 2))
    ldvl = max(1, stride(VL, 2))
    ldvr = max(1, stride(VR, 2))
    dh = dense_handle()

    function bufferSize()
        out = Ref{Cint}(0)
        cusolverDnSgeev_bufferSize(dh, jobvl, jobvr, n, A, lda, W, VL, ldvl, VR, ldvr, out)
        return out[] * sizeof(T)
    end

    with_workspace(dh.workspace_gpu, bufferSize) do buffer
        return cusolverDnSgeev(dh, jobvl, jobvr, n, A, lda, W, VL, ldvl, VR, ldvr, buffer,
                               sizeof(buffer) ÷ sizeof(T), dh.info)
    end

    info = @allowscalar dh.info[1]
    chkargsok(BlasInt(info))
    return W, VL, VR
end

# Wrapper for Randomized SVD (example implementation)
function randomized_svd!(A::StridedCuMatrix{T}, S::StridedCuVector{T},
                         U::StridedCuMatrix{T}, V::StridedCuMatrix{T},
                         rank::Int) where {T<:BlasFloat}
    # Example implementation for randomized SVD
    # Generate random projection matrix
    Omega = CuArray{T}(randn(size(A, 2), rank))
    Y = A * Omega
    Q, _ = qr(Y)
    B = Q' * A
    gesvdqr!(B, S, U, V)
    U = Q * U
    return S, U, V
end

end