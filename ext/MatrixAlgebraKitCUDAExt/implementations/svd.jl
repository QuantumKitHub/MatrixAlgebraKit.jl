const CUSOLVER_SVDAlgorithm = Union{CUSOLVER_QRIteration,
                                    CUSOLVER_SVDPolar,
                                    CUSOLVER_Jacobi}

# CUSOLVER SVD implementation
function MatrixAlgebraKit.svd_full!(A::CuMatrix, USVᴴ, alg::CUSOLVER_SVDAlgorithm)
    check_input(svd_full!, A, USVᴴ)
    U, S, Vᴴ = USVᴴ
    fill!(S, zero(eltype(S)))
    m, n = size(A)
    minmn = min(m, n)
    if alg isa CUSOLVER_QRIteration
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_QRIteration does not accept any keyword arguments"))
        YACUSOLVER.gesvd!(A, view(S, 1:minmn, 1), U, Vᴴ)
    elseif alg isa CUSOLVER_SVDPolar
        YACUSOLVER.Xgesvdp!(A, view(S, 1:minmn, 1), U, Vᴴ; alg.kwargs...)
    elseif alg isa CUSOLVER_Jacobi
        YACUSOLVER.gesvdj!(A, view(S, 1:minmn, 1), U, Vᴴ; alg.kwargs...)
        # elseif alg isa LAPACK_Bisection
        #     throw(ArgumentError("LAPACK_Bisection is not supported for full SVD"))
        # elseif alg isa LAPACK_Jacobi
        #     throw(ArgumentError("LAPACK_Bisection is not supported for full SVD"))
    else
        throw(ArgumentError("Unsupported SVD algorithm"))
    end
    diagview(S) .= view(S, 1:minmn, 1)
    view(S, 2:minmn, 1) .= zero(eltype(S))
    # TODO: make this controllable using a `gaugefix` keyword argument
    for j in 1:max(m, n)
        if j <= minmn
            u = view(U, :, j)
            v = view(Vᴴ, j, :)
            s = conj(sign(_argmaxabs(u)))
            u .*= s
            v .*= conj(s)
        elseif j <= m
            u = view(U, :, j)
            s = conj(sign(_argmaxabs(u)))
            u .*= s
        else
            v = view(Vᴴ, j, :)
            s = conj(sign(_argmaxabs(v)))
            v .*= s
        end
    end
    return USVᴴ
end

function MatrixAlgebraKit.svd_compact!(A::CuMatrix, USVᴴ, alg::CUSOLVER_SVDAlgorithm)
    check_input(svd_compact!, A, USVᴴ)
    U, S, Vᴴ = USVᴴ
    if alg isa CUSOLVER_QRIteration
        isempty(alg.kwargs) ||
            throw(ArgumentError("CUSOLVER_QRIteration does not accept any keyword arguments"))
        YACUSOLVER.gesvd!(A, S.diag, U, Vᴴ)
    elseif alg isa CUSOLVER_SVDPolar
        YACUSOLVER.Xgesvdp!(A, S.diag, U, Vᴴ; alg.kwargs...)
    elseif alg isa CUSOLVER_Jacobi
        YACUSOLVER.gesvdj!(A, S.diag, U, Vᴴ; alg.kwargs...)
        # elseif alg isa LAPACK_DivideAndConquer
        #     isempty(alg.kwargs) ||
        #         throw(ArgumentError("LAPACK_DivideAndConquer does not accept any keyword arguments"))
        #     YALAPACK.gesdd!(A, S.diag, U, Vᴴ)
        # elseif alg isa LAPACK_Bisection
        #     YALAPACK.gesvdx!(A, S.diag, U, Vᴴ; alg.kwargs...)
    else
        throw(ArgumentError("Unsupported SVD algorithm"))
    end
    # TODO: make this controllable using a `gaugefix` keyword argument
    for j in 1:size(U, 2)
        u = view(U, :, j)
        v = view(Vᴴ, j, :)
        s = conj(sign(_argmaxabs(u)))
        u .*= s
        v .*= conj(s)
    end
    return USVᴴ
end
_argmaxabs(x) = reduce(_largest, x; init=zero(eltype(x)))
_largest(x, y) = abs(x) < abs(y) ? y : x

function MatrixAlgebraKit.svd_vals!(A::CuMatrix, S, alg::CUSOLVER_SVDAlgorithm)
    check_input(svd_vals!, A, S)
    U, Vᴴ = similar(A, (0, 0)), similar(A, (0, 0))
    if alg isa CUSOLVER_QRIteration
        isempty(alg.kwargs) ||
            throw(ArgumentError("CUSOLVER_QRIteration does not accept any keyword arguments"))
        YACUSOLVER.gesvd!(A, S, U, Vᴴ)
    elseif alg isa CUSOLVER_SVDPolar
        YACUSOLVER.Xgesvdp!(A, S, U, Vᴴ; alg.kwargs...)
    elseif alg isa CUSOLVER_Jacobi
        YACUSOLVER.gesvdj!(A, S, U, Vᴴ; alg.kwargs...)
        # elseif alg isa LAPACK_DivideAndConquer
        #     isempty(alg.kwargs) ||
        #         throw(ArgumentError("LAPACK_DivideAndConquer does not accept any keyword arguments"))
        #     YALAPACK.gesdd!(A, S, U, Vᴴ)
        # elseif alg isa LAPACK_Bisection
        #     YALAPACK.gesvdx!(A, S, U, Vᴴ; alg.kwargs...)
        # elseif alg isa LAPACK_Jacobi
        #     isempty(alg.kwargs) ||
        #         throw(ArgumentError("LAPACK_Jacobi does not accept any keyword arguments"))
        #     YALAPACK.gesvj!(A, S, U, Vᴴ)
    else
        throw(ArgumentError("Unsupported SVD algorithm"))
    end
    return S
end