# Inputs
# ------
function copy_input(::typeof(svd_full), A)
    return copy!(similar(A, float(eltype(A))), A)
end
copy_input(::typeof(svd_compact), A) = copy_input(svd_full, A)
copy_input(::typeof(svd_vals), A) = copy_input(svd_full, A)
copy_input(::typeof(svd_trunc), A) = copy_input(svd_compact, A)

# TODO: many of these checks are happening again in the LAPACK routines
function check_input(::typeof(svd_full!), A::AbstractMatrix, USVᴴ, ::AbstractAlgorithm)
    m, n = size(A)
    U, S, Vᴴ = USVᴴ
    @assert U isa AbstractMatrix && S isa AbstractMatrix && Vᴴ isa AbstractMatrix
    @check_size(U, (m, m))
    @check_scalar(U, A)
    @check_size(S, (m, n))
    @check_scalar(S, A, real)
    @check_size(Vᴴ, (n, n))
    @check_scalar(Vᴴ, A)
    return nothing
end
function check_input(::typeof(svd_compact!), A::AbstractMatrix, USVᴴ, ::AbstractAlgorithm)
    m, n = size(A)
    minmn = min(m, n)
    U, S, Vᴴ = USVᴴ
    @assert U isa AbstractMatrix && S isa Diagonal && Vᴴ isa AbstractMatrix
    @check_size(U, (m, minmn))
    @check_scalar(U, A)
    @check_size(S, (minmn, minmn))
    @check_scalar(S, A, real)
    @check_size(Vᴴ, (minmn, n))
    @check_scalar(Vᴴ, A)
    return nothing
end
function check_input(::typeof(svd_vals!), A::AbstractMatrix, S, ::AbstractAlgorithm)
    m, n = size(A)
    minmn = min(m, n)
    @assert S isa AbstractVector
    @check_size(S, (minmn,))
    @check_scalar(S, A, real)
    return nothing
end

# Outputs
# -------
function initialize_output(::typeof(svd_full!), A::AbstractMatrix, ::AbstractAlgorithm)
    m, n = size(A)
    U = similar(A, (m, m))
    S = similar(A, real(eltype(A)), (m, n)) # TODO: Rectangular diagonal type?
    Vᴴ = similar(A, (n, n))
    return (U, S, Vᴴ)
end
function initialize_output(::typeof(svd_compact!), A::AbstractMatrix, ::AbstractAlgorithm)
    m, n = size(A)
    minmn = min(m, n)
    U = similar(A, (m, minmn))
    S = Diagonal(similar(A, real(eltype(A)), (minmn,)))
    Vᴴ = similar(A, (minmn, n))
    return (U, S, Vᴴ)
end
function initialize_output(::typeof(svd_vals!), A::AbstractMatrix, ::AbstractAlgorithm)
    return similar(A, real(eltype(A)), (min(size(A)...),))
end
function initialize_output(::typeof(svd_trunc!), A::AbstractMatrix, alg::TruncatedAlgorithm)
    return initialize_output(svd_compact!, A, alg.alg)
end


# Implementation
# --------------
function svd_full!(A::AbstractMatrix, USVᴴ, alg::LAPACK_SVDAlgorithm)
    check_input(svd_full!, A, USVᴴ, alg)
    U, S, Vᴴ = USVᴴ
    fill!(S, zero(eltype(S)))
    m, n = size(A)
    minmn = min(m, n)
    if minmn == 0
        one!(U)
        zero!(S)
        one!(Vᴴ)
        return USVᴴ
    end
    if alg isa LAPACK_QRIteration
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_QRIteration does not accept any keyword arguments"))
        YALAPACK.gesvd!(A, view(S, 1:minmn, 1), U, Vᴴ)
    elseif alg isa LAPACK_DivideAndConquer
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_DivideAndConquer does not accept any keyword arguments"))
        YALAPACK.gesdd!(A, view(S, 1:minmn, 1), U, Vᴴ)
    elseif alg isa LAPACK_Bisection
        throw(ArgumentError("LAPACK_Bisection is not supported for full SVD"))
    elseif alg isa LAPACK_Jacobi
        throw(ArgumentError("LAPACK_Bisection is not supported for full SVD"))
    else
        throw(ArgumentError("Unsupported SVD algorithm"))
    end
    for i in 2:minmn
        S[i, i] = S[i, 1]
        S[i, 1] = zero(eltype(S))
    end
    # TODO: make this controllable using a `gaugefix` keyword argument
    gaugefix!(svd_full!, U, S, Vᴴ, m, n)
    return USVᴴ
end

function svd_compact!(A::AbstractMatrix, USVᴴ, alg::LAPACK_SVDAlgorithm)
    check_input(svd_compact!, A, USVᴴ, alg)
    U, S, Vᴴ = USVᴴ
    if alg isa LAPACK_QRIteration
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_QRIteration does not accept any keyword arguments"))
        YALAPACK.gesvd!(A, S.diag, U, Vᴴ)
    elseif alg isa LAPACK_DivideAndConquer
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_DivideAndConquer does not accept any keyword arguments"))
        YALAPACK.gesdd!(A, S.diag, U, Vᴴ)
    elseif alg isa LAPACK_Bisection
        YALAPACK.gesvdx!(A, S.diag, U, Vᴴ; alg.kwargs...)
    elseif alg isa LAPACK_Jacobi
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_Jacobi does not accept any keyword arguments"))
        YALAPACK.gesvj!(A, S.diag, U, Vᴴ)
    else
        throw(ArgumentError("Unsupported SVD algorithm"))
    end
    # TODO: make this controllable using a `gaugefix` keyword argument
    gaugefix!(svd_compact!, U, S, Vᴴ, size(A)...)
    return USVᴴ
end

function svd_vals!(A::AbstractMatrix, S, alg::LAPACK_SVDAlgorithm)
    check_input(svd_vals!, A, S, alg)
    U, Vᴴ = similar(A, (0, 0)), similar(A, (0, 0))
    if alg isa LAPACK_QRIteration
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_QRIteration does not accept any keyword arguments"))
        YALAPACK.gesvd!(A, S, U, Vᴴ)
    elseif alg isa LAPACK_DivideAndConquer
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_DivideAndConquer does not accept any keyword arguments"))
        YALAPACK.gesdd!(A, S, U, Vᴴ)
    elseif alg isa LAPACK_Bisection
        YALAPACK.gesvdx!(A, S, U, Vᴴ; alg.kwargs...)
    elseif alg isa LAPACK_Jacobi
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_Jacobi does not accept any keyword arguments"))
        YALAPACK.gesvj!(A, S, U, Vᴴ)
    else
        throw(ArgumentError("Unsupported SVD algorithm"))
    end
    return S
end

function svd_trunc!(A::AbstractMatrix, USVᴴ, alg::TruncatedAlgorithm)
    USVᴴ′ = svd_compact!(A, USVᴴ, alg.alg)
    return truncate!(svd_trunc!, USVᴴ′, alg.trunc)
end

### GPU logic
# placed here to avoid code duplication since much of the logic is replicable across
# CUDA and AMDGPU
###
const CUSOLVER_SVDAlgorithm = Union{CUSOLVER_QRIteration,
                                    CUSOLVER_SVDPolar,
                                    CUSOLVER_Jacobi,
                                    CUSOLVER_Randomized}
const ROCSOLVER_SVDAlgorithm = Union{ROCSOLVER_QRIteration,
                                     ROCSOLVER_Jacobi}
const GPU_SVDAlgorithm = Union{CUSOLVER_SVDAlgorithm, ROCSOLVER_SVDAlgorithm}

const GPU_QRIteration = Union{CUSOLVER_QRIteration, ROCSOLVER_QRIteration}
const GPU_SVDPolar = Union{CUSOLVER_SVDPolar}
const GPU_Jacobi = Union{CUSOLVER_Jacobi, ROCSOLVER_Jacobi}
const GPU_Randomized = Union{CUSOLVER_Randomized}

function check_input(::typeof(svd_trunc!), A::AbstractMatrix, USVᴴ, alg::CUSOLVER_Randomized)
    m, n = size(A)
    minmn = min(m, n)
    U, S, Vᴴ = USVᴴ
    @assert U isa AbstractMatrix && S isa Diagonal && Vᴴ isa AbstractMatrix
    @check_size(U, (m, m))
    @check_scalar(U, A)
    @check_size(S, (minmn, minmn))
    @check_scalar(S, A, real)
    @check_size(Vᴴ, (n, n))
    @check_scalar(Vᴴ, A)
    return nothing
end

function initialize_output(::typeof(svd_trunc!), A::AbstractMatrix, alg::TruncatedAlgorithm{<:CUSOLVER_Randomized})
    m, n = size(A)
    minmn = min(m, n)
    U = similar(A, (m, m))
    S = Diagonal(similar(A, real(eltype(A)), (minmn,)))
    Vᴴ = similar(A, (n, n))
    return (U, S, Vᴴ)
end

_gpu_gesvd!(A::AbstractMatrix, S::AbstractVector, U::AbstractMatrix, Vᴴ::AbstractMatrix) = throw(MethodError(_gpu_gesvd!, (A, S, U, Vᴴ)))
_gpu_Xgesvdp!(A::AbstractMatrix, S::AbstractVector, U::AbstractMatrix, Vᴴ::AbstractMatrix; kwargs...) = throw(MethodError(_gpu_Xgesvdp!, (A, S, U, Vᴴ)))
_gpu_Xgesvdr!(A::AbstractMatrix, S::AbstractVector, U::AbstractMatrix, Vᴴ::AbstractMatrix; kwargs...) = throw(MethodError(_gpu_Xgesvdr!, (A, S, U, Vᴴ)))
_gpu_gesvdj!(A::AbstractMatrix, S::AbstractVector, U::AbstractMatrix, Vᴴ::AbstractMatrix; kwargs...) = throw(MethodError(_gpu_gesvdj!, (A, S, U, Vᴴ)))
# GPU SVD implementation
function MatrixAlgebraKit.svd_full!(A::AbstractMatrix, USVᴴ, alg::GPU_SVDAlgorithm)
    check_input(svd_full!, A, USVᴴ, alg)
    U, S, Vᴴ = USVᴴ
    fill!(S, zero(eltype(S)))
    m, n = size(A)
    minmn = min(m, n)
    if alg isa GPU_QRIteration
        isempty(alg.kwargs) ||
            throw(ArgumentError("GPU_QRIteration does not accept any keyword arguments"))
        _gpu_gesvd!(A, view(S, 1:minmn, 1), U, Vᴴ)
    elseif alg isa GPU_SVDPolar
        _gpu_Xgesvdp!(A, view(S, 1:minmn, 1), U, Vᴴ; alg.kwargs...)
    elseif alg isa GPU_Jacobi
        _gpu_gesvdj!(A, view(S, 1:minmn, 1), U, Vᴴ; alg.kwargs...)
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
    gaugefix!(svd_full!, U, S, Vᴴ, m, n)
    return USVᴴ
end

function svd_trunc!(A::AbstractMatrix, USVᴴ, alg::TruncatedAlgorithm{<:GPU_Randomized})
    check_input(svd_trunc!, A, USVᴴ, alg.alg)
    U, S, Vᴴ = USVᴴ
    _gpu_Xgesvdr!(A, S.diag, U, Vᴴ; alg.alg.kwargs...)
    # TODO: make this controllable using a `gaugefix` keyword argument
    gaugefix!(svd_trunc!, U, S, Vᴴ, size(A)...)
    return truncate!(svd_trunc!, USVᴴ, alg.trunc)
end

function MatrixAlgebraKit.svd_compact!(A::AbstractMatrix, USVᴴ, alg::GPU_SVDAlgorithm)
    check_input(svd_compact!, A, USVᴴ, alg)
    U, S, Vᴴ = USVᴴ
    if alg isa GPU_QRIteration
        isempty(alg.kwargs) ||
            throw(ArgumentError("GPU_QRIteration does not accept any keyword arguments"))
        _gpu_gesvd!(A, S.diag, U, Vᴴ)
    elseif alg isa GPU_SVDPolar
        _gpu_Xgesvdp!(A, S.diag, U, Vᴴ; alg.kwargs...)
    elseif alg isa GPU_Jacobi
        _gpu_gesvdj!(A, S.diag, U, Vᴴ; alg.kwargs...)
    else
        throw(ArgumentError("Unsupported SVD algorithm"))
    end
    # TODO: make this controllable using a `gaugefix` keyword argument
    gaugefix!(svd_compact!, U, S, Vᴴ, size(A)...) 
    return USVᴴ
end
_argmaxabs(x) = reduce(_largest, x; init=zero(eltype(x)))
_largest(x, y) = abs(x) < abs(y) ? y : x

function MatrixAlgebraKit.svd_vals!(A::AbstractMatrix, S, alg::GPU_SVDAlgorithm)
    check_input(svd_vals!, A, S, alg)
    U, Vᴴ = similar(A, (0, 0)), similar(A, (0, 0))
    if alg isa GPU_QRIteration
        isempty(alg.kwargs) ||
            throw(ArgumentError("GPU_QRIteration does not accept any keyword arguments"))
        _gpu_gesvd!(A, S, U, Vᴴ)
    elseif alg isa GPU_SVDPolar
        _gpu_Xgesvdp!(A, S, U, Vᴴ; alg.kwargs...)
    elseif alg isa GPU_Jacobi
        _gpu_gesvdj!(A, S, U, Vᴴ; alg.kwargs...)
    else
        throw(ArgumentError("Unsupported SVD algorithm"))
    end
    return S
end
