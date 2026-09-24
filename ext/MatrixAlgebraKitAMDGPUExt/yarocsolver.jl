module YArocSOLVER

using LinearAlgebra
using LinearAlgebra: BlasInt, BlasReal, BlasFloat, checksquare, chkstride1, require_one_based_indexing
using LinearAlgebra.LAPACK: chkargsok, chklapackerror, chktrans, chkside, chkdiag, chkuplo

using AMDGPU
using AMDGPU: @allowscalar
using AMDGPU.rocSOLVER
using AMDGPU.rocBLAS
using ..MatrixAlgebraKit: CHECK_LIBRARY_CALLS

# QR methods are implemented with full access to allocated arrays, so we do not need to redo this:
using AMDGPU.rocSOLVER: geqrf!, ormqr!, orgqr!
const unmqr! = ormqr!
const ungqr! = orgqr!

# Wrapper for SVD via QR Iteration
for (fname, elty, relty) in
    (
        (:rocsolver_sgesvd, :Float32, :Float32),
        (:rocsolver_dgesvd, :Float64, :Float64),
        (:rocsolver_cgesvd, :ComplexF32, :Float32),
        (:rocsolver_zgesvd, :ComplexF64, :Float64),
    )
    @eval begin
        function gesvd!(
                A::StridedROCMatrix{$elty},
                S::StridedROCVector{$relty} = similar(A, $relty, min(size(A)...)),
                U::StridedROCMatrix{$elty} = similar(A, $elty, size(A, 1), min(size(A)...)),
                Vᴴ::StridedROCMatrix{$elty} = similar(A, $elty, min(size(A)...), size(A, 2));
                check::Bool = CHECK_LIBRARY_CALLS[],
            )
            chkstride1(A, U, Vᴴ, S)
            m, n = size(A)
            (m < n) && throw(ArgumentError("rocSOLVER's gesvd requires m ≥ n"))
            minmn = min(m, n)
            if length(U) == 0
                jobu = rocSOLVER.rocblas_svect_none
            else
                size(U, 1) == m ||
                    throw(DimensionMismatch("row size mismatch between A and U"))
                if size(U, 2) == minmn
                    if U === A
                        jobu = rocSOLVER.rocblas_svect_overwrite
                    else
                        jobu = rocSOLVER.rocblas_svect_singular
                    end
                elseif size(U, 2) == m
                    jobu = rocSOLVER.rocblas_svect_all
                else
                    throw(DimensionMismatch("invalid column size of U"))
                end
            end
            if length(Vᴴ) == 0
                jobvt = rocSOLVER.rocblas_svect_none
            else
                size(Vᴴ, 2) == n ||
                    throw(DimensionMismatch("column size mismatch between A and Vᴴ"))
                if size(Vᴴ, 1) == minmn
                    if Vᴴ === A
                        jobvt = rocSOLVER.rocblas_svect_overwrite
                    else
                        jobvt = rocSOLVER.rocblas_svect_singular
                    end
                elseif size(Vᴴ, 1) == n
                    jobvt = rocSOLVER.rocblas_svect_all
                else
                    throw(DimensionMismatch("invalid row size of Vᴴ"))
                end
            end
            length(S) == minmn ||
                throw(DimensionMismatch("length mismatch between A and S"))

            lda = max(1, stride(A, 2))
            ldu = max(1, stride(U, 2))
            ldv = max(1, stride(Vᴴ, 2))

            rwork = ROCArray{$relty}(undef, minmn - 1)
            dh = rocBLAS.handle()
            dev_info = ROCVector{Cint}(undef, 1)
            rocSOLVER.$fname(
                dh, jobu, jobvt, m, n,
                A, lda, S, U, ldu, Vᴴ, ldv,
                rwork, convert(rocSOLVER.rocblas_workmode, 'I'),
                dev_info
            )
            AMDGPU.unsafe_free!(rwork)

            if check
                info = @allowscalar dev_info[1]
                rocSOLVER.chkargsok(BlasInt(info))
            end

            return (S, U, Vᴴ)
        end
    end
end

# Wrappers for batched SVD via QR Iteration
for (fname, elty, relty) in
    (
        (:rocsolver_sgesvd_batched, :Float32, :Float32),
        (:rocsolver_dgesvd_batched, :Float64, :Float64),
        (:rocsolver_cgesvd_batched, :ComplexF32, :Float32),
        (:rocsolver_zgesvd_batched, :ComplexF64, :Float64),
    )
    @eval begin
        function gesvd_batched!(
                A::AbstractVector{<:StridedROCMatrix{$elty}},
                S::StridedROCMatrix{$relty} = similar(first(A), $relty, (min(size(first(A))...), length(A))),
                U::StridedROCArray{$elty, 3} = similar(first(A), $elty, size(first(A), 1), min(size(first(A))...), length(A)),
                Vᴴ::StridedROCArray{$elty, 3} = similar(first(A), $elty, min(size(first(A))...), size(first(A), 2), length(A));
                check::Bool = CHECK_LIBRARY_CALLS[],
            )
            for A_ in A
                chkstride1(A_, U, Vᴴ, S)
            end
            m, n = size(first(A))
            (m < n) && throw(ArgumentError("rocSOLVER's gesvd_batched requires m ≥ n"))
            minmn = min(m, n)
            if length(U) == 0
                jobu = rocSOLVER.rocblas_svect_none
            else
                size(U, 1) == m ||
                    throw(DimensionMismatch("row size mismatch between A and U"))
                if size(U, 2) == minmn
                    if U === A # seems impossible?
                        jobu = rocSOLVER.rocblas_svect_overwrite
                    else
                        jobu = rocSOLVER.rocblas_svect_singular
                    end
                elseif size(U, 2) == m
                    jobu = rocSOLVER.rocblas_svect_all
                else
                    throw(DimensionMismatch("invalid column size of U"))
                end
            end
            if length(Vᴴ) == 0
                jobvt = rocSOLVER.rocblas_svect_none
            else
                size(Vᴴ, 2) == n ||
                    throw(DimensionMismatch("column size mismatch between A and Vᴴ"))
                if size(Vᴴ, 1) == minmn
                    if Vᴴ === A # seems impossible?
                        jobvt = rocSOLVER.rocblas_svect_overwrite
                    else
                        jobvt = rocSOLVER.rocblas_svect_singular
                    end
                elseif size(Vᴴ, 1) == n
                    jobvt = rocSOLVER.rocblas_svect_all
                else
                    throw(DimensionMismatch("invalid row size of Vᴴ"))
                end
            end
            length(S) == minmn * length(A) ||
                throw(DimensionMismatch("length mismatch between A and S"))

            lda = max(1, stride(first(A), 2))
            ldu = max(1, stride(U, 2))
            strideU = ldu * size(U, 2)
            ldv = max(1, stride(Vᴴ, 2))
            strideV = ldv * n
            strideS = minmn

            strideE = minmn - 1
            E = ROCArray{$relty}(undef, length(A) * strideE)
            dh = rocBLAS.handle()
            dev_info = ROCVector{Cint}(undef, length(A))
            pA = ROCVector(map(pointer, A))
            rocSOLVER.$fname(
                dh, jobu, jobvt, m, n, pA, lda,
                S, strideS, U, ldu, strideU, Vᴴ, ldv, strideV,
                E, strideE, convert(rocSOLVER.rocblas_workmode, 'I'),
                dev_info, length(A)
            )
            AMDGPU.unsafe_free!(pA)
            AMDGPU.unsafe_free!(E)
            if check
                foreach(rocSOLVER.chkargsok ∘ BlasInt, collect(dev_info))
            end

            return (S, U, Vᴴ)
        end
    end
end

for (fname, elty, relty) in
    (
        (:rocsolver_sgesvd_strided_batched, :Float32, :Float32),
        (:rocsolver_dgesvd_strided_batched, :Float64, :Float64),
        (:rocsolver_cgesvd_strided_batched, :ComplexF32, :Float32),
        (:rocsolver_zgesvd_strided_batched, :ComplexF64, :Float64),
    )
    @eval begin
        function gesvd_strided_batched!(
                A::StridedROCArray{$elty, 3},
                S::StridedROCMatrix{$relty} = similar(A, $relty, min(size(A, 1, size(A, 2))), size(A, 3)),
                U::StridedROCArray{$elty, 3} = similar(A, $elty, size(A, 1), min(size(A, 1), size(A, 2)), size(A, 3)),
                Vᴴ::StridedROCArray{$elty, 3} = similar(A, $elty, min(size(A, 1), size(A, 2)), size(A, 2), size(A, 3));
                check::Bool = CHECK_LIBRARY_CALLS[],
            )
            chkstride1(A, U, Vᴴ, S)
            m, n, batch_size = size(A)
            (m < n) && throw(ArgumentError("rocSOLVER's gesvd_strided_batched requires m ≥ n"))
            minmn = min(m, n)
            if length(U) == 0
                jobu = rocSOLVER.rocblas_svect_none
            else
                size(U, 1) == m ||
                    throw(DimensionMismatch("row size mismatch between A and U"))
                if size(U, 2) == minmn
                    if U === A
                        jobu = rocSOLVER.rocblas_svect_overwrite
                    else
                        jobu = rocSOLVER.rocblas_svect_singular
                    end
                elseif size(U, 2) == m
                    jobu = rocSOLVER.rocblas_svect_all
                else
                    throw(DimensionMismatch("invalid column size of U"))
                end
            end
            if length(Vᴴ) == 0
                jobvt = rocSOLVER.rocblas_svect_none
            else
                size(Vᴴ, 2) == n ||
                    throw(DimensionMismatch("column size mismatch between A and Vᴴ"))
                if size(Vᴴ, 1) == minmn
                    if Vᴴ === A
                        jobvt = rocSOLVER.rocblas_svect_overwrite
                    else
                        jobvt = rocSOLVER.rocblas_svect_singular
                    end
                elseif size(Vᴴ, 1) == n
                    jobvt = rocSOLVER.rocblas_svect_all
                else
                    throw(DimensionMismatch("invalid row size of Vᴴ"))
                end
            end
            length(S) == minmn * batch_size ||
                throw(DimensionMismatch("length mismatch between A and S"))

            lda = max(1, stride(A, 2))
            strideA = lda * n
            ldu = max(1, stride(U, 2))
            strideU = ldu * size(U, 2)
            ldv = max(1, stride(Vᴴ, 2))
            strideV = ldv * n
            strideS = minmn

            strideE = minmn - 1
            E = ROCArray{$relty}(undef, batch_size * strideE)
            dh = rocBLAS.handle()
            dev_info = ROCVector{Cint}(undef, batch_size)
            rocSOLVER.$fname(
                dh, jobu, jobvt, m, n, A, lda, strideA,
                S, strideS, U, ldu, strideU, Vᴴ, ldv, strideV,
                E, strideE, convert(rocSOLVER.rocblas_workmode, 'I'),
                dev_info, batch_size
            )
            AMDGPU.unsafe_free!(E)

            if check
                foreach(rocSOLVER.chkargsok ∘ BlasInt, collect(dev_info))
            end

            return (S, U, Vᴴ)
        end
    end
end

# Wrapper for SVD via DivideAndConquer
for (fname, elty, relty) in
    (
        (:rocsolver_sgesdd, :Float32, :Float32),
        (:rocsolver_dgesdd, :Float64, :Float64),
        (:rocsolver_cgesdd, :ComplexF32, :Float32),
        (:rocsolver_zgesdd, :ComplexF64, :Float64),
    )
    @eval begin
        function gesdd!(
                A::StridedROCMatrix{$elty},
                S::StridedROCVector{$relty} = similar(A, $relty, min(size(A)...)),
                U::StridedROCMatrix{$elty} = similar(A, $elty, size(A, 1), min(size(A)...)),
                Vᴴ::StridedROCMatrix{$elty} = similar(A, $elty, min(size(A)...), size(A, 2));
                check::Bool = CHECK_LIBRARY_CALLS[],
            )
            chkstride1(A, U, Vᴴ, S)
            m, n = size(A)
            minmn = min(m, n)
            if length(U) == 0
                jobu = rocSOLVER.rocblas_svect_none
            else
                size(U, 1) == m ||
                    throw(DimensionMismatch("row size mismatch between A and U"))
                if size(U, 2) == minmn
                    if U === A
                        jobu = rocSOLVER.rocblas_svect_overwrite
                    else
                        jobu = rocSOLVER.rocblas_svect_singular
                    end
                elseif size(U, 2) == m
                    jobu = rocSOLVER.rocblas_svect_all
                else
                    throw(DimensionMismatch("invalid column size of U"))
                end
            end
            if length(Vᴴ) == 0
                jobvt = rocSOLVER.rocblas_svect_none
            else
                size(Vᴴ, 2) == n ||
                    throw(DimensionMismatch("column size mismatch between A and Vᴴ"))
                if size(Vᴴ, 1) == minmn
                    if Vᴴ === A
                        jobvt = rocSOLVER.rocblas_svect_overwrite
                    else
                        jobvt = rocSOLVER.rocblas_svect_singular
                    end
                elseif size(Vᴴ, 1) == n
                    jobvt = rocSOLVER.rocblas_svect_all
                else
                    throw(DimensionMismatch("invalid row size of Vᴴ"))
                end
            end
            length(S) == minmn ||
                throw(DimensionMismatch("length mismatch between A and S"))

            lda = max(1, stride(A, 2))
            ldu = max(1, stride(U, 2))
            ldv = max(1, stride(Vᴴ, 2))

            dh = rocBLAS.handle()
            dev_info = ROCVector{Cint}(undef, 1)
            rocSOLVER.$fname(
                dh, jobu, jobvt, m, n,
                A, lda, S, U, ldu, Vᴴ, ldv,
                dev_info
            )

            if check
                info = @allowscalar dev_info[1]
                rocSOLVER.chkargsok(BlasInt(info))
            end
            return (S, U, Vᴴ)
        end
    end
end

# Wrapper for batched SVD via DivideAndConquer
for (fname, elty, relty) in
    (
        (:rocsolver_sgesdd_batched, :Float32, :Float32),
        (:rocsolver_dgesdd_batched, :Float64, :Float64),
        (:rocsolver_cgesdd_batched, :ComplexF32, :Float32),
        (:rocsolver_zgesdd_batched, :ComplexF64, :Float64),
    )
    @eval begin
        function gesdd_batched!(
                A::AbstractVector{<:StridedROCMatrix{$elty}},
                S::StridedROCMatrix{$relty} = similar(first(A), $relty, (min(size(first(A))...), length(A))),
                U::StridedROCArray{$elty, 3} = similar(first(A), $elty, size(first(A), 1), min(size(first(A))...), length(A)),
                Vᴴ::StridedROCArray{$elty, 3} = similar(first(A), $elty, min(size(first(A))...), size(first(A), 2), length(A));
                check::Bool = CHECK_LIBRARY_CALLS[],
            )
            for A_ in A
                chkstride1(A_, U, Vᴴ, S)
            end
            m, n = size(first(A))
            minmn = min(m, n)
            if length(U) == 0
                jobu = rocSOLVER.rocblas_svect_none
            else
                size(U, 1) == m ||
                    throw(DimensionMismatch("row size mismatch between A and U"))
                if size(U, 2) == minmn
                    if U === A # seems impossible?
                        jobu = rocSOLVER.rocblas_svect_overwrite
                    else
                        jobu = rocSOLVER.rocblas_svect_singular
                    end
                elseif size(U, 2) == m
                    jobu = rocSOLVER.rocblas_svect_all
                else
                    throw(DimensionMismatch("invalid column size of U"))
                end
            end
            if length(Vᴴ) == 0
                jobvt = rocSOLVER.rocblas_svect_none
            else
                size(Vᴴ, 2) == n ||
                    throw(DimensionMismatch("column size mismatch between A and Vᴴ"))
                if size(Vᴴ, 1) == minmn
                    if Vᴴ === A # seems impossible?
                        jobvt = rocSOLVER.rocblas_svect_overwrite
                    else
                        jobvt = rocSOLVER.rocblas_svect_singular
                    end
                elseif size(Vᴴ, 1) == n
                    jobvt = rocSOLVER.rocblas_svect_all
                else
                    throw(DimensionMismatch("invalid row size of Vᴴ"))
                end
            end
            length(S) == minmn * length(A) ||
                throw(DimensionMismatch("length mismatch between A and S"))

            lda = max(1, stride(first(A), 2))
            ldu = max(1, stride(U, 2))
            strideU = ldu * size(U, 2)
            ldv = max(1, stride(Vᴴ, 2))
            strideV = ldv * n
            strideS = minmn

            dh = rocBLAS.handle()
            dev_info = ROCVector{Cint}(undef, length(A))
            pA = ROCVector(map(pointer, A))
            rocSOLVER.$fname(
                dh, jobu, jobvt, m, n, pA, lda,
                S, strideS, U, ldu, strideU, Vᴴ, ldv, strideV,
                dev_info, length(A)
            )
            AMDGPU.unsafe_free!(pA)
            if check
                foreach(rocSOLVER.chkargsok ∘ BlasInt, collect(dev_info))
            end

            return (S, U, Vᴴ)
        end
    end
end

for (fname, elty, relty) in
    (
        (:rocsolver_sgesdd_strided_batched, :Float32, :Float32),
        (:rocsolver_dgesdd_strided_batched, :Float64, :Float64),
        (:rocsolver_cgesdd_strided_batched, :ComplexF32, :Float32),
        (:rocsolver_zgesdd_strided_batched, :ComplexF64, :Float64),
    )
    @eval begin
        function gesdd_strided_batched!(
                A::StridedROCArray{$elty, 3},
                S::StridedROCMatrix{$relty} = similar(first(A), $relty, (min(size(first(A))...), length(A))),
                U::StridedROCArray{$elty, 3} = similar(first(A), $elty, size(first(A), 1), min(size(first(A))...), length(A)),
                Vᴴ::StridedROCArray{$elty, 3} = similar(first(A), $elty, min(size(first(A))...), size(first(A), 2), length(A));
                check::Bool = CHECK_LIBRARY_CALLS[],
            )
            chkstride1(A, U, Vᴴ, S)
            m, n, batch_size = size(A)
            minmn = min(m, n)
            if length(U) == 0
                jobu = rocSOLVER.rocblas_svect_none
            else
                size(U, 1) == m ||
                    throw(DimensionMismatch("row size mismatch between A and U"))
                if size(U, 2) == minmn
                    if U === A
                        jobu = rocSOLVER.rocblas_svect_overwrite
                    else
                        jobu = rocSOLVER.rocblas_svect_singular
                    end
                elseif size(U, 2) == m
                    jobu = rocSOLVER.rocblas_svect_all
                else
                    throw(DimensionMismatch("invalid column size of U"))
                end
            end
            if length(Vᴴ) == 0
                jobvt = rocSOLVER.rocblas_svect_none
            else
                size(Vᴴ, 2) == n ||
                    throw(DimensionMismatch("column size mismatch between A and Vᴴ"))
                if size(Vᴴ, 1) == minmn
                    if Vᴴ === A
                        jobvt = rocSOLVER.rocblas_svect_overwrite
                    else
                        jobvt = rocSOLVER.rocblas_svect_singular
                    end
                elseif size(Vᴴ, 1) == n
                    jobvt = rocSOLVER.rocblas_svect_all
                else
                    throw(DimensionMismatch("invalid row size of Vᴴ"))
                end
            end
            length(S) == minmn * batch_size ||
                throw(DimensionMismatch("length mismatch between A and S"))

            lda = max(1, stride(A, 2))
            strideA = lda * n
            ldu = max(1, stride(U, 2))
            strideU = ldu * size(U, 2)
            ldv = max(1, stride(Vᴴ, 2))
            strideV = ldv * n
            strideS = minmn

            dh = rocBLAS.handle()
            dev_info = ROCVector{Cint}(undef, batch_size)
            rocSOLVER.$fname(
                dh, jobu, jobvt, m, n, A, lda, strideA,
                S, strideS, U, ldu, strideU, Vᴴ, ldv, strideV,
                dev_info, batch_size
            )
            if check
                foreach(rocSOLVER.chkargsok ∘ BlasInt, collect(dev_info))
            end
            return (S, U, Vᴴ)
        end
    end
end

# Wrapper for SVD via Jacobi
for (fname, elty, relty) in
    (
        (:rocsolver_sgesvdj, :Float32, :Float32),
        (:rocsolver_dgesvdj, :Float64, :Float64),
        (:rocsolver_cgesvdj, :ComplexF32, :Float32),
        (:rocsolver_zgesvdj, :ComplexF64, :Float64),
    )
    @eval begin
        function gesvdj!(
                A::StridedROCMatrix{$elty},
                S::StridedROCVector{$relty} = similar(A, $relty, min(size(A)...)),
                U::StridedROCMatrix{$elty} = similar(A, $elty, size(A, 1), min(size(A)...)),
                Vᴴ::StridedROCMatrix{$elty} = similar(A, $elty, min(size(A)...), size(A, 2));
                tol::$relty = eps($relty),
                max_sweeps::Int = 100,
                check::Bool = CHECK_LIBRARY_CALLS[],
            )
            chkstride1(A, U, Vᴴ, S)
            m, n = size(A)
            minmn = min(m, n)

            if length(U) == 0
                jobu = rocSOLVER.rocblas_svect_none
            else
                size(U, 1) == m ||
                    throw(DimensionMismatch("row size mismatch between A and U"))
                if size(U, 2) == minmn
                    if U === A
                        throw(ArgumentError("overwrite mode is not supported for gesvdj"))
                    else
                        jobu = rocSOLVER.rocblas_svect_singular
                    end
                elseif size(U, 2) == m
                    jobu = rocSOLVER.rocblas_svect_all
                else
                    throw(DimensionMismatch("invalid column size of U"))
                end
            end
            if length(Vᴴ) == 0
                jobvt = rocSOLVER.rocblas_svect_none
            else
                size(Vᴴ, 2) == n ||
                    throw(DimensionMismatch("column size mismatch between A and Vᴴ"))
                if size(Vᴴ, 1) == minmn
                    if Vᴴ === A
                        throw(ArgumentError("overwrite mode is not supported for gesvdj"))
                    else
                        jobvt = rocSOLVER.rocblas_svect_singular
                    end
                elseif size(Vᴴ, 1) == n
                    jobvt = rocSOLVER.rocblas_svect_all
                else
                    throw(DimensionMismatch("invalid row size of Vᴴ"))
                end
            end
            length(S) == minmn ||
                throw(DimensionMismatch("length mismatch between A and S"))

            lda = max(1, stride(A, 2))
            ldu = max(1, stride(U, 2))
            ldv = max(1, stride(Vᴴ, 2))
            dev_info = ROCVector{Cint}(undef, 1)
            dev_residual = ROCVector{$relty}(undef, 1)
            dev_n_sweeps = ROCVector{Cint}(undef, 1)

            dh = rocBLAS.handle()
            rocSOLVER.$fname(
                dh, jobu, jobvt, m, n, A, lda, tol,
                dev_residual, max_sweeps, dev_n_sweeps,
                S, U, ldu, Vᴴ, ldv, dev_info,
            )

            if check
                info = @allowscalar dev_info[1]
                rocSOLVER.chkargsok(BlasInt(info))
            end

            AMDGPU.unsafe_free!(dev_residual)
            AMDGPU.unsafe_free!(dev_n_sweeps)
            return (S, U, Vᴴ)
        end
    end
end

# Wrapper for batched SVD via Jacobi
for (fname, elty, relty) in
    (
        (:rocsolver_sgesvdj_batched, :Float32, :Float32),
        (:rocsolver_dgesvdj_batched, :Float64, :Float64),
        (:rocsolver_cgesvdj_batched, :ComplexF32, :Float32),
        (:rocsolver_zgesvdj_batched, :ComplexF64, :Float64),
    )
    @eval begin
        function gesvdj_batched!(
                A::AbstractVector{<:StridedROCMatrix{$elty}},
                S::StridedROCMatrix{$relty} = similar(first(A), $relty, (min(size(first(A))...), length(A))),
                U::StridedROCArray{$elty, 3} = similar(first(A), $elty, size(first(A), 1), min(size(first(A))...), length(A)),
                Vᴴ::StridedROCArray{$elty, 3} = similar(first(A), $elty, min(size(first(A))...), size(first(A), 2), length(A)),
                tol::$relty = eps($relty),
                max_sweeps::Int = 100,
                check::Bool = CHECK_LIBRARY_CALLS[],
            )
            for A_ in A
                chkstride1(A_, U, Vᴴ, S)
            end
            m, n = size(first(A))
            minmn = min(m, n)

            if length(U) == 0
                jobu = rocSOLVER.rocblas_svect_none
            else
                size(U, 1) == m ||
                    throw(DimensionMismatch("row size mismatch between A and U"))
                if size(U, 2) == minmn
                    if U === A
                        throw(ArgumentError("overwrite mode is not supported for gesvdj"))
                    else
                        jobu = rocSOLVER.rocblas_svect_singular
                    end
                elseif size(U, 2) == m
                    jobu = rocSOLVER.rocblas_svect_all
                else
                    throw(DimensionMismatch("invalid column size of U"))
                end
            end
            if length(Vᴴ) == 0
                jobvt = rocSOLVER.rocblas_svect_none
            else
                size(Vᴴ, 2) == n ||
                    throw(DimensionMismatch("column size mismatch between A and Vᴴ"))
                if size(Vᴴ, 1) == minmn
                    if Vᴴ === A
                        throw(ArgumentError("overwrite mode is not supported for gesvdj"))
                    else
                        jobvt = rocSOLVER.rocblas_svect_singular
                    end
                elseif size(Vᴴ, 1) == n
                    jobvt = rocSOLVER.rocblas_svect_all
                else
                    throw(DimensionMismatch("invalid row size of Vᴴ"))
                end
            end
            length(S) == minmn * length(A) ||
                throw(DimensionMismatch("length mismatch between A and S"))

            lda = max(1, stride(first(A), 2))
            ldu = max(1, stride(U, 2))
            strideU = ldu * size(U, 2)
            ldv = max(1, stride(Vᴴ, 2))
            strideV = ldv * n
            strideS = minmn
            dev_info = ROCVector{Cint}(undef, length(A))
            dev_residual = ROCVector{$relty}(undef, length(A))
            dev_n_sweeps = ROCVector{Cint}(undef, length(A))

            dh = rocBLAS.handle()
            pA = ROCVector(map(pointer, A))
            rocSOLVER.$fname(
                dh, jobu, jobvt, m, n, pA, lda, tol,
                dev_residual, max_sweeps, dev_n_sweeps,
                S, strideS, U, ldu, strideU, Vᴴ, ldv, strideV,
                dev_info, length(A)
            )
            if check
                foreach(rocSOLVER.chkargsok ∘ BlasInt, collect(dev_info))
            end
            AMDGPU.unsafe_free!(pA)
            AMDGPU.unsafe_free!(dev_residual)
            AMDGPU.unsafe_free!(dev_n_sweeps)
            return (S, U, Vᴴ)
        end
    end
end

for (fname, elty, relty) in
    (
        (:rocsolver_sgesvdj_strided_batched, :Float32, :Float32),
        (:rocsolver_dgesvdj_strided_batched, :Float64, :Float64),
        (:rocsolver_cgesvdj_strided_batched, :ComplexF32, :Float32),
        (:rocsolver_zgesvdj_strided_batched, :ComplexF64, :Float64),
    )
    @eval begin
        function gesvdj_strided_batched!(
                A::StridedROCArray{$elty, 3},
                S::StridedROCMatrix{$relty} = similar(A, $relty, min(size(A, 1, size(A, 2))), size(A, 3)),
                U::StridedROCArray{$elty, 3} = similar(A, $elty, size(A, 1), min(size(A, 1), size(A, 2)), size(A, 3)),
                Vᴴ::StridedROCArray{$elty, 3} = similar(A, $elty, min(size(A, 1), size(A, 2)), size(A, 2), size(A, 3));
                tol::$relty = eps($relty),
                max_sweeps::Int = 100,
                check::Bool = CHECK_LIBRARY_CALLS[],
            )
            chkstride1(A, U, Vᴴ, S)
            m, n, batch_size = size(A)
            minmn = min(m, n)

            if length(U) == 0
                jobu = rocSOLVER.rocblas_svect_none
            else
                size(U, 1) == m ||
                    throw(DimensionMismatch("row size mismatch between A and U"))
                if size(U, 2) == minmn
                    if U === A
                        throw(ArgumentError("overwrite mode is not supported for gesvdj"))
                    else
                        jobu = rocSOLVER.rocblas_svect_singular
                    end
                elseif size(U, 2) == m
                    jobu = rocSOLVER.rocblas_svect_all
                else
                    throw(DimensionMismatch("invalid column size of U"))
                end
            end
            if length(Vᴴ) == 0
                jobvt = rocSOLVER.rocblas_svect_none
            else
                size(Vᴴ, 2) == n ||
                    throw(DimensionMismatch("column size mismatch between A and Vᴴ"))
                if size(Vᴴ, 1) == minmn
                    if Vᴴ === A
                        throw(ArgumentError("overwrite mode is not supported for gesvdj"))
                    else
                        jobvt = rocSOLVER.rocblas_svect_singular
                    end
                elseif size(Vᴴ, 1) == n
                    jobvt = rocSOLVER.rocblas_svect_all
                else
                    throw(DimensionMismatch("invalid row size of Vᴴ"))
                end
            end
            length(S) == minmn * batch_size ||
                throw(DimensionMismatch("length mismatch between A and S"))

            lda = max(1, stride(A, 2))
            strideA = lda * n
            ldu = max(1, stride(U, 2))
            strideU = ldu * size(U, 2)
            ldv = max(1, stride(Vᴴ, 2))
            strideV = ldv * n
            strideS = minmn
            dev_info = ROCVector{Cint}(undef, batch_size)
            dev_residual = ROCVector{$relty}(undef, batch_size)
            dev_n_sweeps = ROCVector{Cint}(undef, batch_size)

            dh = rocBLAS.handle()
            rocSOLVER.$fname(
                dh, jobu, jobvt, m, n, A, lda, strideA, tol,
                dev_residual, max_sweeps, dev_n_sweeps,
                S, strideS, U, ldu, strideU, Vᴴ, ldv, strideV,
                dev_info, batch_size
            )

            if check
                foreach(rocSOLVER.chkargsok ∘ BlasInt, collect(dev_info))
            end
            AMDGPU.unsafe_free!(dev_residual)
            AMDGPU.unsafe_free!(dev_n_sweeps)
            return (S, U, Vᴴ)
        end
    end
end

# `gesvdx` computes all singular values, those in the half-open interval `[vl, vu)`, or those
# with an index in `irange`. Unlike the other algorithms, it never forms the full `U` and `Vᴴ`,
# so the only job modes are `singular` and `none`.
function _gesvdx_range(::Type{T}, kwargs) where {T <: Real}
    if haskey(kwargs, :irange)
        irange = convert(UnitRange{Int}, kwargs[:irange])
        return rocSOLVER.rocblas_srange_index, zero(T), zero(T), first(irange), last(irange)
    elseif haskey(kwargs, :vl) || haskey(kwargs, :vu)
        vl = convert(T, get(kwargs, :vl, -Inf))
        vu = convert(T, get(kwargs, :vu, +Inf))
        return rocSOLVER.rocblas_srange_value, vl, vu, 0, 0
    else
        return rocSOLVER.rocblas_srange_all, zero(T), zero(T), 0, 0
    end
end

function _gesvdx_jobs(U, Vᴴ, m::Integer, n::Integer, maxnsv::Integer)
    if length(U) == 0
        jobu = rocSOLVER.rocblas_svect_none
    else
        size(U, 1) == m ||
            throw(DimensionMismatch("row size mismatch between A ($m) and U ($(size(U, 1)))"))
        size(U, 2) >= maxnsv ||
            throw(DimensionMismatch("invalid column size of U"))
        jobu = rocSOLVER.rocblas_svect_singular
    end
    if length(Vᴴ) == 0
        jobvt = rocSOLVER.rocblas_svect_none
    else
        size(Vᴴ, 2) == n ||
            throw(DimensionMismatch("column size mismatch between A ($n) and Vᴴ ($(size(Vᴴ, 2)))"))
        size(Vᴴ, 1) >= maxnsv ||
            throw(DimensionMismatch("invalid row size of Vᴴ"))
        jobvt = rocSOLVER.rocblas_svect_singular
    end
    return jobu, jobvt
end

"""
    _gesvdx_zero_unconverged!(S, nsv)

Zero the entries of `S` that `gesvdx` did not write.
"""
function _gesvdx_zero_unconverged!(S::StridedROCVector, nsv::ROCVector{Cint})
    nv = @allowscalar Int(nsv[1])
    nv < length(S) && fill!(view(S, (nv + 1):length(S)), zero(eltype(S)))
    return S
end
function _gesvdx_zero_unconverged!(S::StridedROCMatrix, nsv::ROCVector{Cint})
    minmn = size(S, 1)
    nvs = Array(nsv)
    all(==(minmn), nvs) && return S          # nothing omitted, skip the per-batch fills
    for (b, nv) in pairs(nvs)
        nv < minmn && fill!(view(S, (nv + 1):minmn, b), zero(eltype(S)))
    end
    return S
end

# Wrapper for SVD via Bisection
for (fname, elty, relty) in
    (
        (:rocsolver_sgesvdx, :Float32, :Float32),
        (:rocsolver_dgesvdx, :Float64, :Float64),
        (:rocsolver_cgesvdx, :ComplexF32, :Float32),
        (:rocsolver_zgesvdx, :ComplexF64, :Float64),
    )
    @eval begin
        function gesvdx!(
                A::StridedROCMatrix{$elty},
                S::StridedROCVector{$relty} = similar(A, $relty, min(size(A)...)),
                U::StridedROCMatrix{$elty} = similar(A, $elty, size(A, 1), min(size(A)...)),
                Vᴴ::StridedROCMatrix{$elty} = similar(A, $elty, min(size(A)...), size(A, 2));
                check::Bool = CHECK_LIBRARY_CALLS[],
                kwargs...
            )
            chkstride1(A, U, Vᴴ, S)
            m, n = size(A)
            minmn = min(m, n)
            srange, vl, vu, il, iu = _gesvdx_range($relty, kwargs)
            maxnsv = srange == rocSOLVER.rocblas_srange_index ? iu - il + 1 : minmn
            jobu, jobvt = _gesvdx_jobs(U, Vᴴ, m, n, maxnsv)
            length(S) == minmn ||
                throw(DimensionMismatch("length mismatch between A ($minmn) and S ($(length(S)))"))

            lda = max(1, stride(A, 2))
            ldu = max(1, stride(U, 2))
            ldv = max(1, stride(Vᴴ, 2))
            ifail = ROCVector{Cint}(undef, minmn)
            nsv = ROCVector{Cint}(undef, 1)
            dh = rocBLAS.handle()
            dev_info = ROCVector{Cint}(undef, 1)
            rocSOLVER.$fname(
                dh, jobu, jobvt, srange, m, n,
                A, lda, vl, vu, il, iu, nsv,
                S, U, ldu, Vᴴ, ldv, ifail,
                dev_info
            )
            if check
                info = @allowscalar dev_info[1]
                rocSOLVER.chkargsok(BlasInt(info))
            end
            # Zero the entries of `S` that `gesvdx` did not write.
            _gesvdx_zero_unconverged!(S, nsv)

            AMDGPU.unsafe_free!(nsv)
            AMDGPU.unsafe_free!(ifail)
            AMDGPU.unsafe_free!(dev_info)
            return (S, U, Vᴴ)
        end
    end
end

# Wrappers for batched SVD via Bisection
for (fname, elty, relty) in
    (
        (:rocsolver_sgesvdx_batched, :Float32, :Float32),
        (:rocsolver_dgesvdx_batched, :Float64, :Float64),
        (:rocsolver_cgesvdx_batched, :ComplexF32, :Float32),
        (:rocsolver_zgesvdx_batched, :ComplexF64, :Float64),
    )
    @eval begin
        function gesvdx_batched!(
                A::AbstractVector{<:StridedROCMatrix{$elty}},
                S::StridedROCMatrix{$relty} = similar(first(A), $relty, (min(size(first(A))...), length(A))),
                U::StridedROCArray{$elty, 3} = similar(first(A), $elty, size(first(A), 1), min(size(first(A))...), length(A)),
                Vᴴ::StridedROCArray{$elty, 3} = similar(first(A), $elty, min(size(first(A))...), size(first(A), 2), length(A));
                check::Bool = CHECK_LIBRARY_CALLS[],
                kwargs...
            )
            for A_ in A
                chkstride1(A_, U, Vᴴ, S)
            end
            m, n = size(first(A))
            minmn = min(m, n)
            batch_count = length(A)
            srange, vl, vu, il, iu = _gesvdx_range($relty, kwargs)
            maxnsv = srange == rocSOLVER.rocblas_srange_index ? iu - il + 1 : minmn
            jobu, jobvt = _gesvdx_jobs(U, Vᴴ, m, n, maxnsv)
            length(S) == minmn * batch_count ||
                throw(DimensionMismatch("length mismatch between A and S"))

            lda = max(1, stride(first(A), 2))
            ldu = max(1, stride(U, 2))
            strideU = ldu * size(U, 2)
            ldv = max(1, stride(Vᴴ, 2))
            strideV = ldv * n
            strideS = minmn
            strideF = minmn

            dh = rocBLAS.handle()
            nsv = ROCVector{Cint}(undef, batch_count)
            ifail = ROCVector{Cint}(undef, minmn * batch_count)
            dev_info = ROCVector{Cint}(undef, batch_count)
            pA = ROCVector(map(pointer, A))
            rocSOLVER.$fname(
                dh, jobu, jobvt, srange, m, n, pA, lda,
                vl, vu, il, iu, nsv,
                S, strideS, U, ldu, strideU, Vᴴ, ldv, strideV,
                ifail, strideF, dev_info, batch_count
            )
            AMDGPU.unsafe_free!(pA)

            if check
                rocSOLVER.chkargsok.(BlasInt.(collect(dev_info)))
            end
            _gesvdx_zero_unconverged!(S, nsv)

            AMDGPU.unsafe_free!(nsv)
            AMDGPU.unsafe_free!(ifail)
            AMDGPU.unsafe_free!(dev_info)
            return (S, U, Vᴴ)
        end
    end
end

for (fname, elty, relty) in
    (
        (:rocsolver_sgesvdx_strided_batched, :Float32, :Float32),
        (:rocsolver_dgesvdx_strided_batched, :Float64, :Float64),
        (:rocsolver_cgesvdx_strided_batched, :ComplexF32, :Float32),
        (:rocsolver_zgesvdx_strided_batched, :ComplexF64, :Float64),
    )
    @eval begin
        function gesvdx_strided_batched!(
                A::StridedROCArray{$elty, 3},
                S::StridedROCMatrix{$relty} = similar(A, $relty, (min(size(A, 1), size(A, 2)), size(A, 3))),
                U::StridedROCArray{$elty, 3} = similar(A, $elty, size(A, 1), min(size(A, 1), size(A, 2)), size(A, 3)),
                Vᴴ::StridedROCArray{$elty, 3} = similar(A, $elty, min(size(A, 1), size(A, 2)), size(A, 2), size(A, 3));
                check::Bool = CHECK_LIBRARY_CALLS[],
                kwargs...
            )
            chkstride1(A, U, Vᴴ, S)
            m, n, batch_count = size(A)
            minmn = min(m, n)
            srange, vl, vu, il, iu = _gesvdx_range($relty, kwargs)
            maxnsv = srange == rocSOLVER.rocblas_srange_index ? iu - il + 1 : minmn
            jobu, jobvt = _gesvdx_jobs(U, Vᴴ, m, n, maxnsv)
            length(S) == minmn * batch_count ||
                throw(DimensionMismatch("length mismatch between A and S"))

            lda = max(1, stride(A, 2))
            strideA = stride(A, 3)
            ldu = max(1, stride(U, 2))
            strideU = ldu * size(U, 2)
            ldv = max(1, stride(Vᴴ, 2))
            strideV = ldv * n
            strideS = minmn
            strideF = minmn

            dh = rocBLAS.handle()
            nsv = ROCVector{Cint}(undef, batch_count)
            ifail = ROCVector{Cint}(undef, minmn * batch_count)
            dev_info = ROCVector{Cint}(undef, batch_count)
            rocSOLVER.$fname(
                dh, jobu, jobvt, srange, m, n, A, lda, strideA,
                vl, vu, il, iu, nsv,
                S, strideS, U, ldu, strideU, Vᴴ, ldv, strideV,
                ifail, strideF, dev_info, batch_count
            )
            if check
                rocSOLVER.chkargsok.(BlasInt.(collect(dev_info)))
            end
            _gesvdx_zero_unconverged!(S, nsv)

            AMDGPU.unsafe_free!(nsv)
            AMDGPU.unsafe_free!(ifail)
            AMDGPU.unsafe_free!(dev_info)
            return (S, U, Vᴴ)
        end
    end
end

# for (jname, bname, fname, elty, relty) in
#     ((:sygvd!, :rocsolverDnSsygvd_bufferSize, :rocsolverDnSsygvd, :Float32, :Float32),
#      (:sygvd!, :rocsolverDnDsygvd_bufferSize, :rocsolverDnDsygvd, :Float64, :Float64),
#      (:hegvd!, :rocsolverDnChegvd_bufferSize, :rocsolverDnChegvd, :ComplexF32, :Float32),
#      (:hegvd!, :rocsolverDnZhegvd_bufferSize, :rocsolverDnZhegvd, :ComplexF64, :Float64))
#     @eval begin
#         function $jname(itype::Int,
#                         jobz::Char,
#                         uplo::Char,
#                         A::StridedROCMatrix{$elty},
#                         B::StridedROCMatrix{$elty};
#                         check::Bool = CHECK_LIBRARY_CALLS[],
#                        )
#             chkuplo(uplo)
#             nA, nB = checksquare(A, B)
#             if nB != nA
#                 throw(DimensionMismatch("Dimensions of A ($nA, $nA) and B ($nB, $nB) must match!"))
#             end
#             n = nA
#             lda = max(1, stride(A, 2))
#             ldb = max(1, stride(B, 2))
#             W = CuArray{$relty}(undef, n)
#             dh = rocBLAS.handle()

#             function bufferSize()
#                 out = Ref{Cint}(0)
#                 $bname(dh, itype, jobz, uplo, n, A, lda, B, ldb, W, out)
#                 return out[] * sizeof($elty)
#             end

#             with_workspace(dh.workspace_gpu, bufferSize) do buffer
#                 return $fname(dh, itype, jobz, uplo, n, A, lda, B, ldb, W,
#                               buffer, sizeof(buffer) ÷ sizeof($elty), dh.info)
#             end

#             if check
#                 info = @allowscalar dh.info[1]
#                 chkargsok(BlasInt(info))
#             end
#             if jobz == 'N'
#                 return W
#             elseif jobz == 'V'
#                 return W, A, B
#             end
#         end
#     end
# end

# for (jname, bname, fname, elty, relty) in
#     ((:sygvj!, :rocsolverDnSsygvj_bufferSize, :rocsolverDnSsygvj, :Float32, :Float32),
#      (:sygvj!, :rocsolverDnDsygvj_bufferSize, :rocsolverDnDsygvj, :Float64, :Float64),
#      (:hegvj!, :rocsolverDnChegvj_bufferSize, :rocsolverDnChegvj, :ComplexF32, :Float32),
#      (:hegvj!, :rocsolverDnZhegvj_bufferSize, :rocsolverDnZhegvj, :ComplexF64, :Float64))
#     @eval begin
#         function $jname(itype::Int,
#                         jobz::Char,
#                         uplo::Char,
#                         A::StridedROCMatrix{$elty},
#                         B::StridedROCMatrix{$elty};
#                         tol::$relty=eps($relty),
#                         max_sweeps::Int=100;
#                         check::Bool = CHECK_LIBRARY_CALLS[],
#                        )
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
#             rocsolverDnCreateSyevjInfo(params)
#             rocsolverDnXsyevjSetTolerance(params[], tol)
#             rocsolverDnXsyevjSetMaxSweeps(params[], max_sweeps)
#             dh = rocBLAS.handle()

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
#             if check
#                 info = @allowscalar dh.info[1]
#                 chkargsok(BlasInt(info))
#             end

#             rocsolverDnDestroySyevjInfo(params[])

#             if jobz == 'N'
#                 return W
#             elseif jobz == 'V'
#                 return W, A, B
#             end
#         end
#     end
# end

# for (jname, bname, fname, elty, relty) in
#     ((:syevjBatched!, :rocsolverDnSsyevjBatched_bufferSize, :rocsolverDnSsyevjBatched,
#       :Float32, :Float32),
#      (:syevjBatched!, :rocsolverDnDsyevjBatched_bufferSize, :rocsolverDnDsyevjBatched,
#       :Float64, :Float64),
#      (:heevjBatched!, :rocsolverDnCheevjBatched_bufferSize, :rocsolverDnCheevjBatched,
#       :ComplexF32, :Float32),
#      (:heevjBatched!, :rocsolverDnZheevjBatched_bufferSize, :rocsolverDnZheevjBatched,
#       :ComplexF64, :Float64))
#     @eval begin
#         function $jname(jobz::Char,
#                         uplo::Char,
#                         A::StridedROCArray{$elty};
#                         tol::$relty=eps($relty),
#                         max_sweeps::Int=100;
#                         check::Bool = CHECK_LIBRARY_CALLS[],
#                        )

#             # Set up information for the solver arguments
#             chkuplo(uplo)
#             n = checksquare(A)
#             lda = max(1, stride(A, 2))
#             batchSize = size(A, 3)
#             W = CuArray{$relty}(undef, n, batchSize)
#             params = Ref{syevjInfo_t}(C_NULL)

#             dh = rocBLAS.handle()
#             resize!(dh.info, batchSize)

#             # Initialize the solver parameters
#             rocsolverDnCreateSyevjInfo(params)
#             rocsolverDnXsyevjSetTolerance(params[], tol)
#             rocsolverDnXsyevjSetMaxSweeps(params[], max_sweeps)

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

#             if check
#                 # Copy the solver info and delete the device memory
#                 info = collect(dh.info)
#
#                 # Double check the solver's exit status
#                 for i in 1:batchSize
#                     chkargsok(BlasInt(info[i]))
#                end
#            end
#             rocsolverDnDestroySyevjInfo(params[])

#             # Return eigenvalues (in W) and possibly eigenvectors (in A)
#             if jobz == 'N'
#                 return W
#             elseif jobz == 'V'
#                 return W, A
#             end
#         end
#     end
# end

# for (fname, elty) in ((:rocsolverDnSpotrsBatched, :Float32),
#                       (:rocsolverDnDpotrsBatched, :Float64),
#                       (:rocsolverDnCpotrsBatched, :ComplexF32),
#                       (:rocsolverDnZpotrsBatched, :ComplexF64))
#     @eval begin
#         function potrsBatched!(uplo::Char,
#                                A::Vector{<:StridedROCMatrix{$elty}},
#                                B::Vector{<:StridedROCVecOrMat{$elty}};
#                                check::Bool = CHECK_LIBRARY_CALLS[],
#                               )
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

#             dh = rocBLAS.handle()

#             # Run the solver
#             $fname(dh, uplo, n, nrhs, Aptrs, lda, Bptrs, ldb, dh.info, batchSize)
#             if check
#                 # Copy the solver info and delete the device memory
#                 info = @allowscalar dh.info[1]
#                 chklapackerror(BlasInt(info))
#             end
#             return B
#         end
#     end
# end

# for (fname, elty) in ((:rocsolverDnSpotrfBatched, :Float32),
#                       (:rocsolverDnDpotrfBatched, :Float64),
#                       (:rocsolverDnCpotrfBatched, :ComplexF32),
#                       (:rocsolverDnZpotrfBatched, :ComplexF64))
#     @eval begin
#         function potrfBatched!(uplo::Char, A::Vector{<:StridedROCMatrix{$elty}}; check::Bool = CHECK_LIBRARY_CALLS[],)

#             # Set up information for the solver arguments
#             chkuplo(uplo)
#             n = checksquare(A[1])
#             lda = max(1, stride(A[1], 2))
#             batchSize = length(A)

#             Aptrs = unsafe_batch(A)

#             dh = rocBLAS.handle()
#             resize!(dh.info, batchSize)

#             # Run the solver
#             $fname(dh, uplo, n, Aptrs, lda, dh.info, batchSize)

#             if check
#                 # Copy the solver info and delete the device memory
#                 info = collect(dh.info)

#                 # Double check the solver's exit status
#                 for i in 1:batchSize
#                     chkargsok(BlasInt(info[i]))
#                 end
#             end
#             # info[i] > 0 means the leading minor of order info[i] is not positive definite
#             # LinearAlgebra.LAPACK does not throw Exception here
#             # to simplify calls to isposdef! and factorize
#             return A, info
#         end
#     end
# end

# # gesv
# function gesv!(X::ROCVecOrMat{T}, A::ROCMatrix{T}, B::ROCVecOrMat{T}; fallback::Bool=true,
#                residual_history::Bool=false, irs_precision::String="AUTO",
#                refinement_solver::String="CLASSICAL",
#                maxiters::Int=0, maxiters_inner::Int=0, tol::Float64=0.0,
#                tol_inner=Float64 = 0.0,
#                check::Bool = CHECK_LIBRARY_CALLS[],
#                ) where {T<:BlasFloat}
#     params = CuSolverIRSParameters()
#     info = CuSolverIRSInformation()
#     n = checksquare(A)
#     nrhs = size(B, 2)
#     lda = max(1, stride(A, 2))
#     ldb = max(1, stride(B, 2))
#     ldx = max(1, stride(X, 2))
#     niters = Ref{Cint}()
#     dh = rocBLAS.handle()

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
#     rocsolverDnIRSParamsSetSolverMainPrecision(params, T)
#     rocsolverDnIRSParamsSetSolverLowestPrecision(params, irs_precision)
#     rocsolverDnIRSParamsSetRefinementSolver(params, refinement_solver)
#     (tol != 0.0) && rocsolverDnIRSParamsSetTol(params, tol)
#     (tol_inner != 0.0) && rocsolverDnIRSParamsSetTolInner(params, tol_inner)
#     (maxiters != 0) && rocsolverDnIRSParamsSetMaxIters(params, maxiters)
#     (maxiters_inner != 0) && rocsolverDnIRSParamsSetMaxItersInner(params, maxiters_inner)
#     fallback ? rocsolverDnIRSParamsEnableFallback(params) :
#     rocsolverDnIRSParamsDisableFallback(params)
#     residual_history && rocsolverDnIRSInfosRequestResidual(info)

#     function bufferSize()
#         buffer_size = Ref{Csize_t}(0)
#         rocsolverDnIRSXgesv_bufferSize(dh, params, n, nrhs, buffer_size)
#         return buffer_size[]
#     end

#     with_workspace(dh.workspace_gpu, bufferSize) do buffer
#         return rocsolverDnIRSXgesv(dh, params, info, n, nrhs, A, lda, B, ldb,
#                                   X, ldx, buffer, sizeof(buffer), niters, dh.info)
#     end

#     # Copy the solver flag and delete the device memory
#     flag = @allowscalar dh.info[1]
#     chklapackerror(BlasInt(flag))

#     return X, info
# end

for (heevd, heev, heevx, heevj, elty, relty) in
    (
        (:(rocSOLVER.rocsolver_ssyevd), :(rocSOLVER.rocsolver_ssyev), :(rocSOLVER.rocsolver_ssyevx), :(rocSOLVER.rocsolver_ssyevj), :Float32, :Float32),
        (:(rocSOLVER.rocsolver_dsyevd), :(rocSOLVER.rocsolver_dsyev), :(rocSOLVER.rocsolver_dsyevx), :(rocSOLVER.rocsolver_dsyevj), :Float64, :Float64),
        (:(rocSOLVER.rocsolver_cheevd), :(rocSOLVER.rocsolver_cheev), :(rocSOLVER.rocsolver_cheevx), :(rocSOLVER.rocsolver_cheevj), :ComplexF32, :Float32),
        (:(rocSOLVER.rocsolver_zheevd), :(rocSOLVER.rocsolver_zheev), :(rocSOLVER.rocsolver_zheevx), :(rocSOLVER.rocsolver_zheevj), :ComplexF64, :Float64),
    )
    @eval begin
        function heevd!(
                A::StridedROCMatrix{$elty},
                W::StridedROCVector{$relty},
                V::StridedROCMatrix{$elty};
                uplo::Char = 'U',
                check::Bool = CHECK_LIBRARY_CALLS[],
            )
            chkuplo(uplo)
            n = checksquare(A)
            lda = max(1, stride(A, 2))
            length(W) == n || throw(DimensionMismatch("size mismatch between A and W"))
            if length(V) == 0
                jobz = rocSOLVER.rocblas_evect_none
            else
                size(V) == (n, n) || throw(DimensionMismatch("size mismatch between A and V"))
                jobz = rocSOLVER.rocblas_evect_original
            end
            dh = rocBLAS.handle()
            work = ROCVector{$relty}(undef, n)
            dev_info = ROCVector{Cint}(undef, 1)
            roc_uplo = convert(rocSOLVER.rocblas_fill, uplo)
            $heevd(dh, jobz, roc_uplo, n, A, lda, W, work, dev_info)

            if check
                info = @allowscalar dev_info[1]
                chkargsok(BlasInt(info))
            end

            if jobz == rocSOLVER.rocblas_evect_original && V !== A
                copy!(V, A)
            end
            return W, V
        end
        function heev!(
                A::StridedROCMatrix{$elty},
                W::StridedROCVector{$relty},
                V::StridedROCMatrix{$elty};
                uplo::Char = 'U',
                check::Bool = CHECK_LIBRARY_CALLS[],
            )
            chkuplo(uplo)
            n = checksquare(A)
            lda = max(1, stride(A, 2))
            length(W) == n || throw(DimensionMismatch("size mismatch between A and W"))
            if length(V) == 0
                jobz = rocSOLVER.rocblas_evect_none
            else
                size(V) == (n, n) || throw(DimensionMismatch("size mismatch between A and V"))
                jobz = rocSOLVER.rocblas_evect_original
            end
            dh = rocBLAS.handle()
            work = ROCVector{$relty}(undef, n)
            dev_info = ROCVector{Cint}(undef, 1)
            roc_uplo = convert(rocSOLVER.rocblas_fill, uplo)
            $heev(dh, jobz, roc_uplo, n, A, lda, W, work, dev_info)

            if check
                info = @allowscalar dev_info[1]
                chkargsok(BlasInt(info))
            end

            if jobz == rocSOLVER.rocblas_evect_original && V !== A
                copy!(V, A)
            end
            return W, V
        end
        function heevx!(
                A::StridedROCMatrix{$elty},
                W::StridedROCVector{$relty},
                V::StridedROCMatrix{$elty};
                uplo::Char = 'U',
                check::Bool = CHECK_LIBRARY_CALLS[],
                kwargs...
            )
            chkuplo(uplo)
            n = checksquare(A)
            lda = max(1, stride(A, 2))
            length(W) == n || throw(DimensionMismatch("size mismatch between A and W"))
            if haskey(kwargs, :irange)
                il = first(kwargs[:irange])
                iu = last(kwargs[:irange])
                vl = vu = zero($relty)
                range = rocSOLVER.rocblas_erange_index
            elseif haskey(kwargs, :vl) || haskey(kwargs, :vu)
                vl = convert($relty, get(kwargs, :vl, -Inf))
                vu = convert($relty, get(kwargs, :vu, +Inf))
                il = iu = 0
                range = rocSOLVER.rocblas_erange_value
            else
                il = iu = 0
                vl = vu = zero($relty)
                range = rocSOLVER.rocblas_erange_all
            end
            if length(V) == 0
                jobz = rocSOLVER.rocblas_evect_none
            else
                size(V) == (n, n) || throw(DimensionMismatch("size mismatch between A and V"))
                jobz = rocSOLVER.rocblas_evect_original
            end
            dh = rocBLAS.handle()
            abstol = -one($relty)
            nev = ROCVector{Cint}(undef, 1)
            ldv = max(1, stride(V, 2))
            ifail = ROCVector{Cint}(undef, n)
            dev_info = ROCVector{Cint}(undef, 1)
            roc_uplo = convert(rocSOLVER.rocblas_fill, uplo)
            $heevx(dh, jobz, range, roc_uplo, n, A, lda, vl, vu, il, iu, abstol, nev, W, V, ldv, ifail, dev_info)

            if check
                info = @allowscalar dev_info[1]
                chkargsok(BlasInt(info))
            end
            m = @allowscalar nev[1]
            return W, V, m
        end
        function heevj!(
                A::StridedROCMatrix{$elty},
                W::StridedROCVector{$relty},
                V::StridedROCMatrix{$elty};
                uplo::Char = 'U',
                tol::$relty = eps($relty),
                max_sweeps::Int = 100,
                sort::Char = 'N',
                check::Bool = CHECK_LIBRARY_CALLS[],
            )
            chkuplo(uplo)
            n = checksquare(A)
            lda = max(1, stride(A, 2))
            length(W) == n || throw(DimensionMismatch("size mismatch between A and W"))
            if length(V) == 0
                jobz = rocSOLVER.rocblas_evect_none
            else
                size(V) == (n, n) || throw(DimensionMismatch("size mismatch between A and V"))
                jobz = rocSOLVER.rocblas_evect_original
            end
            dh = rocBLAS.handle()
            dev_info = ROCVector{Cint}(undef, 1)
            residual = ROCVector{$relty}(undef, 1)
            n_sweeps = ROCVector{Cint}(undef, 1)
            roc_uplo = convert(rocSOLVER.rocblas_fill, uplo)
            roc_sort = sort == 'N' ? rocSOLVER.rocblas_esort_none : rocSOLVER.rocblas_esort_ascending
            $heevj(dh, roc_sort, jobz, roc_uplo, n, A, lda, tol, residual, max_sweeps, n_sweeps, W, dev_info)

            if check
                info = @allowscalar dev_info[1]
                chkargsok(BlasInt(info))
            end

            if jobz == rocSOLVER.rocblas_evect_original && V !== A
                copy!(V, A)
            end
            return W, V
        end
    end
end

end
