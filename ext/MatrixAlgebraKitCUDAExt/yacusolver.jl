module YACUSOLVER

using LinearAlgebra
using LinearAlgebra: BlasInt, BlasFloat, BlasReal, checksquare, chkstride1, require_one_based_indexing
using LinearAlgebra.LAPACK: chkargsok, chklapackerror, chktrans, chkside, chkdiag, chkuplo

using CUDA
using CUDA: @allowscalar, i32
using CUDA.CUSOLVER

# QR methods are implemented with full access to allocated arrays, so we do not need to redo this:
using CUDA.CUSOLVER: geqrf!, ormqr!, orgqr!
const unmqr! = ormqr!
const ungqr! = orgqr!

# Wrapper for SVD via QR Iteration
for (bname, fname, elty, relty) in
    (
        (:cusolverDnSgesvd_bufferSize, :cusolverDnSgesvd, :Float32, :Float32),
        (:cusolverDnDgesvd_bufferSize, :cusolverDnDgesvd, :Float64, :Float64),
        (:cusolverDnCgesvd_bufferSize, :cusolverDnCgesvd, :ComplexF32, :Float32),
        (:cusolverDnZgesvd_bufferSize, :cusolverDnZgesvd, :ComplexF64, :Float64),
    )
    @eval begin
        function gesvd!(
                A::StridedCuMatrix{$elty},
                S::StridedCuVector{$relty} = similar(A, $relty, min(size(A)...)),
                U::StridedCuMatrix{$elty} = similar(A, $elty, size(A, 1), min(size(A)...)),
                Vᴴ::StridedCuMatrix{$elty} = similar(A, $elty, min(size(A)...), size(A, 2))
            )
            chkstride1(A, U, Vᴴ, S)
            m, n = size(A)
            (m < n) && throw(ArgumentError(lazy"CUSOLVER's gesvd requires m ($m) ≥ n ($n)"))
            minmn = min(m, n)
            if length(U) == 0
                jobu = 'N'
            else
                size(U, 1) == m ||
                    throw(DimensionMismatch("row size mismatch between A and U"))
                if size(U, 2) == minmn
                    if U === A
                        jobu = 'O'
                    else
                        jobu = 'S'
                    end
                elseif size(U, 2) == m
                    jobu = 'A'
                else
                    throw(DimensionMismatch("invalid column size of U"))
                end
            end
            if length(Vᴴ) == 0
                jobvt = 'N'
            else
                size(Vᴴ, 2) == n ||
                    throw(DimensionMismatch("column size mismatch between A and Vᴴ"))
                if size(Vᴴ, 1) == minmn
                    if Vᴴ === A
                        jobvt = 'O'
                    else
                        jobvt = 'S'
                    end
                elseif size(Vᴴ, 1) == n
                    jobvt = 'A'
                else
                    throw(DimensionMismatch("invalid row size of Vᴴ"))
                end
            end
            length(S) == minmn ||
                throw(DimensionMismatch("length mismatch between A and S"))

            lda = max(1, stride(A, 2))
            ldu = max(1, stride(U, 2))
            ldv = max(1, stride(Vᴴ, 2))

            dh = CUSOLVER.dense_handle()
            function bufferSize()
                out = Ref{Cint}(0)
                CUSOLVER.$bname(dh, m, n, out)
                return out[] * sizeof($elty)
            end
            rwork = CuArray{$relty}(undef, min(m, n) - 1)
            CUDA.with_workspace(dh.workspace_gpu, bufferSize) do buffer
                return CUSOLVER.$fname(
                    dh, jobu, jobvt, m, n,
                    A, lda, S, U, ldu, Vᴴ, ldv,
                    buffer, sizeof(buffer) ÷ sizeof($elty), rwork,
                    dh.info
                )
            end
            CUDA.unsafe_free!(rwork)

            info = @allowscalar dh.info[1]
            CUSOLVER.chkargsok(BlasInt(info))

            return (S, U, Vᴴ)
        end
    end
end

function Xgesvdp!(
        A::StridedCuMatrix{T},
        S::StridedCuVector = similar(A, real(T), min(size(A)...)),
        U::StridedCuMatrix{T} = similar(A, T, size(A, 1), min(size(A)...)),
        Vᴴ::StridedCuMatrix{T} = similar(A, T, min(size(A)...), size(A, 2));
        tol = norm(A) * eps(real(T))
    ) where {T <: BlasFloat}
    chkstride1(A, U, S, Vᴴ)
    m, n = size(A)
    minmn = min(m, n)
    if length(U) == length(Vᴴ) == 0
        jobz = 'N'
        econ = 1
    else
        jobz = 'V'
        size(U, 1) == m ||
            throw(DimensionMismatch("row size mismatch between A and U"))
        size(Vᴴ, 2) == n ||
            throw(DimensionMismatch("column size mismatch between A and Vᴴ"))
        if size(U, 2) == size(Vᴴ, 1) == minmn
            econ = 1
        elseif size(U, 2) == m && size(Vᴴ, 1) == n
            econ = 0
        else
            throw(DimensionMismatch("invalid column size of U or row size of Vᴴ"))
        end
    end
    R = eltype(S)
    length(S) == minmn ||
        throw(DimensionMismatch("length mismatch between A and S"))
    R == real(T) ||
        throw(ArgumentError("S does not have the matching real `eltype` of A"))

    Ṽ = similar(Vᴴ, (n, n))
    Ũ = (size(U) == (m, m)) ? U : similar(U, (m, m))
    lda = max(1, stride(A, 2))
    ldu = max(1, stride(Ũ, 2))
    ldv = max(1, stride(Ṽ, 2))
    h_err_sigma = Ref{Cdouble}(0)
    params = CUSOLVER.CuSolverParameters()
    dh = CUSOLVER.dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        CUSOLVER.cusolverDnXgesvdp_bufferSize(
            dh, params, jobz, econ, m, n,
            T, A, lda, R, S, T, Ũ, ldu, T, Ṽ, ldv,
            T, out_gpu, out_cpu
        )

        return out_gpu[], out_cpu[]
    end
    CUSOLVER.with_workspaces(
        dh.workspace_gpu, dh.workspace_cpu,
        bufferSize()...
    ) do buffer_gpu, buffer_cpu
        return CUSOLVER.cusolverDnXgesvdp(
            dh, params, jobz, econ, m, n,
            T, A, lda, R, S, T, Ũ, ldu, T, Ṽ, ldv,
            T, buffer_gpu, sizeof(buffer_gpu),
            buffer_cpu, sizeof(buffer_cpu),
            dh.info, h_err_sigma
        )
    end
    err = h_err_sigma[]
    if err > tol
        warn("Xgesvdp! did not attained requested tolerance: error = $err > tolerance = $tol")
    end

    flag = @allowscalar dh.info[1]
    CUSOLVER.chklapackerror(BlasInt(flag))
    if Ũ !== U && length(U) > 0
        U .= view(Ũ, 1:m, 1:size(U, 2))
    end
    if length(Vᴴ) > 0
        Vᴴ .= view(Ṽ', 1:size(Vᴴ, 1), 1:n)
    end
    Ũ !== U && CUDA.unsafe_free!(Ũ)
    CUDA.unsafe_free!(Ṽ)

    return S, U, Vᴴ
end

# Wrapper for SVD via Jacobi
for (bname, fname, elty, relty) in
    (
        (:cusolverDnSgesvdj_bufferSize, :cusolverDnSgesvdj, :Float32, :Float32),
        (:cusolverDnDgesvdj_bufferSize, :cusolverDnDgesvdj, :Float64, :Float64),
        (:cusolverDnCgesvdj_bufferSize, :cusolverDnCgesvdj, :ComplexF32, :Float32),
        (:cusolverDnZgesvdj_bufferSize, :cusolverDnZgesvdj, :ComplexF64, :Float64),
    )
    @eval begin
        #! format: off
        function gesvdj!(
                A::StridedCuMatrix{$elty},
                S::StridedCuVector{$relty} = similar(A, $relty, min(size(A)...)),
                U::StridedCuMatrix{$elty} = similar(A, $elty, size(A, 1), min(size(A)...)),
                Vᴴ::StridedCuMatrix{$elty} = similar(A, $elty, min(size(A)...), size(A, 2));
                tol::$relty = eps($relty),
                max_sweeps::Int = 100,
                kwargs...
            )
            #! format: on
            chkstride1(A, U, Vᴴ, S)
            m, n = size(A)
            minmn = min(m, n)

            if length(U) == 0 && length(Vᴴ) == 0
                jobz = 'N'
                econ = 0
            else
                jobz = 'V'
                size(U, 1) == m ||
                    throw(DimensionMismatch("row size mismatch between A and U"))
                size(Vᴴ, 2) == n ||
                    throw(DimensionMismatch("column size mismatch between A and Vᴴ"))
                if size(U, 2) == size(Vᴴ, 1) == minmn
                    econ = 1
                elseif size(U, 2) == m && size(Vᴴ, 1) == n
                    econ = 0
                else
                    throw(DimensionMismatch("invalid column size of U or row size of Vᴴ"))
                end
            end
            length(S) == minmn ||
                throw(DimensionMismatch("length mismatch between A and S"))

            Ṽ = (jobz == 'V') ? similar(Vᴴ') : similar(Vᴴ, (n, minmn))
            Ũ = (jobz == 'V') ? U : similar(U, (m, minmn))
            lda = max(1, stride(A, 2))
            ldu = max(1, stride(Ũ, 2))
            ldv = max(1, stride(Ṽ, 2))

            params = Ref{CUSOLVER.gesvdjInfo_t}(C_NULL)
            CUSOLVER.cusolverDnCreateGesvdjInfo(params)
            CUSOLVER.cusolverDnXgesvdjSetTolerance(params[], tol)
            CUSOLVER.cusolverDnXgesvdjSetMaxSweeps(params[], max_sweeps)
            dh = CUSOLVER.dense_handle()

            function bufferSize()
                out = Ref{Cint}(0)
                CUSOLVER.$bname(
                    dh, jobz, econ, m, n, A, lda, S, Ũ, ldu, Ṽ, ldv,
                    out, params[]
                )
                return out[] * sizeof($elty)
            end

            CUSOLVER.with_workspace(dh.workspace_gpu, bufferSize) do buffer
                return CUSOLVER.$fname(
                    dh, jobz, econ, m, n, A, lda, S, Ũ, ldu, Ṽ, ldv,
                    buffer, sizeof(buffer) ÷ sizeof($elty), dh.info,
                    params[]
                )
            end

            info = @allowscalar dh.info[1]
            CUSOLVER.chkargsok(BlasInt(info))

            CUSOLVER.cusolverDnDestroyGesvdjInfo(params[])

            if jobz == 'V'
                adjoint!(Vᴴ, Ṽ)
            end
            return S, U, Vᴴ
        end
    end
end

# Wrapper for randomized SVD
function Xgesvdr!(
        A::StridedCuMatrix{T},
        S::StridedCuVector = similar(A, real(T), min(size(A)...)),
        U::StridedCuMatrix{T} = similar(A, T, size(A, 1), min(size(A)...)),
        Vᴴ::StridedCuMatrix{T} = similar(A, T, min(size(A)...), size(A, 2));
        k::Int = length(S),
        p::Int = min(size(A)...) - k - 1,
        niters::Int = 1
    ) where {T <: BlasFloat}
    chkstride1(A, U, S, Vᴴ)
    m, n = size(A)
    minmn = min(m, n)
    jobu = length(U) == 0 ? 'N' : 'S'
    jobv = length(Vᴴ) == 0 ? 'N' : 'S'
    R = eltype(S)
    k < minmn || throw(DimensionMismatch("length of S ($k) must be less than the smaller dimension of A ($minmn)"))
    k + p < minmn || throw(DimensionMismatch("length of S ($k) plus oversampling ($p) must be less than the smaller dimension of A ($minmn)"))
    R == real(T) ||
        throw(ArgumentError("S does not have the matching real `eltype` of A"))

    Ṽ = similar(Vᴴ, (n, n))
    Ũ = (size(U) == (m, m)) ? U : similar(U, (m, m))
    lda = max(1, stride(A, 2))
    ldu = max(1, stride(Ũ, 2))
    ldv = max(1, stride(Ṽ, 2))
    params = CUSOLVER.CuSolverParameters()
    dh = CUSOLVER.dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        CUSOLVER.cusolverDnXgesvdr_bufferSize(
            dh, params, jobu, jobv, m, n, k, p, niters,
            T, A, lda, R, S, T, Ũ, ldu, T, Ṽ, ldv,
            T, out_gpu, out_cpu
        )

        return out_gpu[], out_cpu[]
    end
    CUSOLVER.with_workspaces(
        dh.workspace_gpu, dh.workspace_cpu,
        bufferSize()...
    ) do buffer_gpu, buffer_cpu
        return CUSOLVER.cusolverDnXgesvdr(
            dh, params, jobu, jobv, m, n, k, p, niters,
            T, A, lda, R, S, T, Ũ, ldu, T, Ṽ, ldv,
            T, buffer_gpu, sizeof(buffer_gpu),
            buffer_cpu, sizeof(buffer_cpu),
            dh.info
        )
    end

    flag = @allowscalar dh.info[1]
    CUSOLVER.chklapackerror(BlasInt(flag))
    if Ũ !== U && length(U) > 0
        U .= view(Ũ, 1:m, 1:size(U, 2))
    end
    if length(Vᴴ) > 0
        Vᴴ .= view(Ṽ', 1:size(Vᴴ, 1), 1:n)
    end
    Ũ !== U && CUDA.unsafe_free!(Ũ)
    CUDA.unsafe_free!(Ṽ)

    return S, U, Vᴴ
end

# Wrapper for general eigensolver
for (celty, elty) in ((:ComplexF32, :Float32), (:ComplexF64, :Float64), (:ComplexF32, :ComplexF32), (:ComplexF64, :ComplexF64))
    @eval begin
        function Xgeev!(A::StridedCuMatrix{$elty}, D::StridedCuVector{$celty}, V::StridedCuMatrix{$celty})
            require_one_based_indexing(A, V, D)
            chkstride1(A, V, D)
            n = checksquare(A)
            # TODO GPU appropriate version
            #chkfinite(A) # balancing routines don't support NaNs and Infs
            n == length(D) || throw(DimensionMismatch("length mismatch between A and D"))
            if length(V) == 0
                jobvr = 'N'
            elseif length(V) == n * n
                jobvr = 'V'
            else
                throw(DimensionMismatch("size of VR must match size of A"))
            end
            jobvl = 'N' # required by API for now (https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnxgeev)
            #=if length(VL) == 0
                jobvl = 'N'
            elseif length(VL) == n*n
                jobvl = 'V'
            else
                throw(DimensionMismatch("size of VL must match size of A"))
            end=#
            VL = similar(A, n, 0)
            lda = max(1, stride(A, 2))
            ldvl = max(1, stride(VL, 2))
            params = CUSOLVER.CuSolverParameters()
            dh = CUSOLVER.dense_handle()

            if $elty <: Real
                D2 = reinterpret($elty, D)
                # reuse memory, we will have to reorder afterwards to bring real and imaginary
                # components in the order as required for the Complex type
                VR = reinterpret($elty, V)
            else
                D2 = D
                VR = V
            end
            ldvr = max(1, stride(VR, 2))
            function bufferSize()
                out_cpu = Ref{Csize_t}(0)
                out_gpu = Ref{Csize_t}(0)
                CUSOLVER.cusolverDnXgeev_bufferSize(
                    dh, params, jobvl, jobvr, n, $elty, A,
                    lda, $elty, D2, $elty, VL, ldvl, $elty, VR, ldvr,
                    $elty, out_gpu, out_cpu
                )
                return out_gpu[], out_cpu[]
            end
            CUDA.with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
                CUSOLVER.cusolverDnXgeev(
                    dh, params, jobvl, jobvr, n, $elty, A, lda, $elty,
                    D2, $elty, VL, ldvl, $elty, VR, ldvr, $elty, buffer_gpu,
                    sizeof(buffer_gpu), buffer_cpu, sizeof(buffer_cpu), dh.info
                )
            end
            flag = @allowscalar dh.info[1]
            CUSOLVER.chkargsok(BlasInt(flag))
            if eltype(A) <: Real
                work = CuVector{$elty}(undef, n)
                DR = view(D2, 1:n)
                DI = view(D2, (n + 1):(2n))
                _reorder_realeigendecomposition!(D, DR, DI, work, VR, jobvr)
            end
            return D, V
        end
    end
end

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

for (bname, fname, elty, relty) in (
        (:(CUSOLVER.cusolverDnSsyevj_bufferSize), :(CUSOLVER.cusolverDnSsyevj), :Float32, :Float32),
        (:(CUSOLVER.cusolverDnDsyevj_bufferSize), :(CUSOLVER.cusolverDnDsyevj), :Float64, :Float64),
        (:(CUSOLVER.cusolverDnCheevj_bufferSize), :(CUSOLVER.cusolverDnCheevj), :ComplexF32, :Float32),
        (:(CUSOLVER.cusolverDnZheevj_bufferSize), :(CUSOLVER.cusolverDnZheevj), :ComplexF64, :Float64),
    )
    @eval begin
        function heevj!(
                A::StridedCuMatrix{$elty},
                W::StridedCuVector{$relty},
                V::StridedCuMatrix{$elty};
                uplo::Char = 'U',
                tol::$relty = eps($relty),
                max_sweeps::Int = 100
            )
            chkuplo(uplo)
            n = checksquare(A)
            lda = max(1, stride(A, 2))
            dh = CUSOLVER.dense_handle()
            length(W) == n || throw(DimensionMismatch("size mismatch between A and W"))
            if length(V) == 0
                jobz = 'N'
            else
                size(V) == (n, n) || throw(DimensionMismatch("size mismatch between A and V"))
                jobz = 'V'
            end
            params = Ref{CUSOLVER.syevjInfo_t}(C_NULL)
            CUSOLVER.cusolverDnCreateSyevjInfo(params)
            CUSOLVER.cusolverDnXsyevjSetTolerance(params[], tol)
            CUSOLVER.cusolverDnXsyevjSetMaxSweeps(params[], max_sweeps)
            function bufferSize()
                out = Ref{Cint}(0)
                $bname(dh, jobz, uplo, n, A, lda, W, out, params[])
                return out[] * sizeof($elty)
            end
            CUDA.with_workspace(dh.workspace_gpu, bufferSize) do buffer
                $fname(
                    dh, jobz, uplo, n, A, lda, W, buffer,
                    sizeof(buffer) ÷ sizeof($elty), dh.info, params[]
                )
            end

            info = @allowscalar dh.info[1]
            chkargsok(BlasInt(info))

            if jobz == 'V' && V !== A
                copy!(V, A)
            end
            return W, V
        end
    end
end

function heevd!(
        A::StridedCuMatrix{T},
        W::StridedCuVector{Tr},
        V::StridedCuMatrix{T};
        uplo::Char = 'U'
    ) where {T <: BlasFloat, Tr <: BlasReal}
    chkuplo(uplo)
    n = checksquare(A)
    lda = max(1, stride(A, 2))
    dh = CUSOLVER.dense_handle()
    length(W) == n || throw(DimensionMismatch("size mismatch between A and W"))
    if length(V) == 0
        jobz = 'N'
    else
        size(V) == (n, n) || throw(DimensionMismatch("size mismatch between A and V"))
        jobz = 'V'
    end

    params = CUSOLVER.CuSolverParameters()
    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        CUSOLVER.cusolverDnXsyevd_bufferSize(dh, params, jobz, uplo, n, T, A, lda, Tr, W, T, out_gpu, out_cpu)
        return out_gpu[], out_cpu[]
    end

    CUSOLVER.with_workspaces(
        dh.workspace_gpu, dh.workspace_cpu,
        bufferSize()...
    ) do buffer_gpu, buffer_cpu
        return CUSOLVER.cusolverDnXsyevd(
            dh, params, jobz, uplo, n, T, A, lda, Tr, W,
            T, buffer_gpu, sizeof(buffer_gpu), buffer_cpu,
            sizeof(buffer_cpu), dh.info
        )
    end

    info = @allowscalar dh.info[1]
    chkargsok(BlasInt(info))

    if jobz == 'V' && V !== A
        copy!(V, A)
    end
    return W, V
end

# device code is unreachable by coverage right now
# COV_EXCL_START
# TODO use a shmem array here
function _reorder_kernel_real(real_ev_ixs, VR::CuDeviceArray{T}, n::Int) where {T}
    grid_idx = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
    @inbounds if grid_idx <= length(real_ev_ixs)
        i = real_ev_ixs[grid_idx]
        for j in n:-1:1
            VR[2 * j, i] = zero(T)
            VR[2 * j - 1, i] = VR[j, i]
        end
    end
    return
end

function _reorder_kernel_complex(complex_ev_ixs, VR::CuDeviceArray{T}, n::Int) where {T}
    grid_idx = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
    @inbounds if grid_idx <= length(complex_ev_ixs)
        i = complex_ev_ixs[grid_idx]
        for j in n:-1:1
            VR[2 * j, i] = VR[j, i + 1]
            VR[2 * j - 1, i] = VR[j, i]
            VR[2 * j, i + 1] = -VR[j, i + 1]
            VR[2 * j - 1, i + 1] = VR[j, i]
        end
    end
    return
end
# COV_EXCL_STOP

function _reorder_realeigendecomposition!(W, WR, WI, work, VR, jobvr)
    # first reorder eigenvalues and recycle work as temporary buffer to efficiently implement the permutation
    copy!(work, WI)
    n = size(W, 1)
    @. W[1:n] = WR[1:n] + im * work[1:n]
    T = eltype(WR)
    return if jobvr == 'V' # also reorganise vectors
        real_ev_ixs = findall(isreal, W)
        _cmplx_ev_ixs = findall(!isreal, W) # these come in pairs, choose only the first of each pair
        complex_ev_ixs = view(_cmplx_ev_ixs, 1:2:length(_cmplx_ev_ixs))
        if !isempty(real_ev_ixs)
            real_threads = 128
            real_blocks = max(1, div(length(real_ev_ixs), real_threads))
            @cuda threads = real_threads blocks = real_blocks _reorder_kernel_real(real_ev_ixs, VR, n)
        end
        if !isempty(complex_ev_ixs)
            complex_threads = 128
            complex_blocks = max(1, div(length(complex_ev_ixs), complex_threads))
            @cuda threads = complex_threads blocks = complex_blocks _reorder_kernel_complex(complex_ev_ixs, VR, n)
        end
    end
end

end
