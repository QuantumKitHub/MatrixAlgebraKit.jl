# Inputs
# ------
function copy_input(::typeof(eigh_full), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
copy_input(::typeof(eigh_vals), A) = copy_input(eigh_full, A)
copy_input(::typeof(eigh_trunc), A) = copy_input(eigh_full, A)

copy_input(::typeof(eigh_full), A::Diagonal) = copy(A)

function check_input(::typeof(eigh_full!), A::AbstractMatrix, DV, ::AbstractAlgorithm)
    m, n = size(A)
    m == n || throw(DimensionMismatch("square input matrix expected"))
    D, V = DV
    @assert D isa Diagonal && V isa AbstractMatrix
    @check_size(D, (m, m))
    @check_scalar(D, A, real)
    @check_size(V, (m, m))
    @check_scalar(V, A)
    return nothing
end
function check_input(::typeof(eigh_vals!), A::AbstractMatrix, D, ::AbstractAlgorithm)
    m, n = size(A)
    m == n || throw(DimensionMismatch("square input matrix expected"))
    @assert D isa AbstractVector
    @check_size(D, (n,))
    @check_scalar(D, A, real)
    return nothing
end

function check_input(::typeof(eigh_full!), A::AbstractMatrix, DV, ::DiagonalAlgorithm)
    m, n = size(A)
    @assert m == n && isdiag(A)
    @assert (eltype(A) <: Real && issymmetric(A)) || ishermitian(A)
    D, V = DV
    @assert D isa Diagonal && V isa Diagonal
    @check_size(D, (m, m))
    @check_scalar(D, A, real)
    @check_size(V, (m, m))
    @check_scalar(V, A)
    return nothing
end
function check_input(::typeof(eigh_vals!), A::AbstractMatrix, D, ::DiagonalAlgorithm)
    m, n = size(A)
    @assert m == n && isdiag(A)
    @assert (eltype(A) <: Real && issymmetric(A)) || ishermitian(A)
    @assert D isa AbstractVector
    @check_size(D, (n,))
    @check_scalar(D, A, real)
    return nothing
end

# Outputs
# -------
function initialize_output(::typeof(eigh_full!), A::AbstractMatrix, ::AbstractAlgorithm)
    n = size(A, 1) # square check will happen later
    D = Diagonal(similar(A, real(eltype(A)), n))
    V = similar(A, (n, n))
    return (D, V)
end
function initialize_output(::typeof(eigh_vals!), A::AbstractMatrix, ::AbstractAlgorithm)
    n = size(A, 1) # square check will happen later
    D = similar(A, real(eltype(A)), n)
    return D
end
function initialize_output(::typeof(eigh_trunc!), A, alg::TruncatedAlgorithm)
    return initialize_output(eigh_full!, A, alg.alg)
end

function initialize_output(::typeof(eigh_full!), A::Diagonal, ::DiagonalAlgorithm)
    return eltype(A) <: Real ? A : similar(A, real(eltype(A))), similar(A)
end
function initialize_output(::typeof(eigh_vals!), A::Diagonal, ::DiagonalAlgorithm)
    return eltype(A) <: Real ? diagview(A) : similar(A, real(eltype(A)), size(A, 1))
end

# Implementation
# --------------
function eigh_full!(A::AbstractMatrix, DV, alg::LAPACK_EighAlgorithm)
    check_input(eigh_full!, A, DV, alg)
    D, V = DV
    Dd = D.diag
    if alg isa LAPACK_MultipleRelativelyRobustRepresentations
        YALAPACK.heevr!(A, Dd, V; alg.kwargs...)
    elseif alg isa LAPACK_DivideAndConquer
        YALAPACK.heevd!(A, Dd, V; alg.kwargs...)
    elseif alg isa LAPACK_Simple
        YALAPACK.heev!(A, Dd, V; alg.kwargs...)
    else # alg isa LAPACK_Expert
        YALAPACK.heevx!(A, Dd, V; alg.kwargs...)
    end
    # TODO: make this controllable using a `gaugefix` keyword argument
    V = gaugefix!(V)
    return D, V
end

function eigh_vals!(A::AbstractMatrix, D, alg::LAPACK_EighAlgorithm)
    check_input(eigh_vals!, A, D, alg)
    V = similar(A, (size(A, 1), 0))
    if alg isa LAPACK_MultipleRelativelyRobustRepresentations
        YALAPACK.heevr!(A, D, V; alg.kwargs...)
    elseif alg isa LAPACK_DivideAndConquer
        YALAPACK.heevd!(A, D, V; alg.kwargs...)
    elseif alg isa LAPACK_QRIteration # == LAPACK_Simple
        YALAPACK.heev!(A, D, V; alg.kwargs...)
    else # alg isa LAPACK_Bisection == LAPACK_Expert
        YALAPACK.heevx!(A, D, V; alg.kwargs...)
    end
    return D
end

function eigh_trunc!(A, DV, alg::TruncatedAlgorithm)
    D, V = eigh_full!(A, DV, alg.alg)
    DVtrunc, ind = truncate(eigh_trunc!, (D, V), alg.trunc)
    return DVtrunc..., truncation_error!(diagview(D), ind)
end

# Diagonal logic
# --------------
function eigh_full!(A::Diagonal, DV, alg::DiagonalAlgorithm)
    check_input(eigh_full!, A, DV, alg)
    D, V = DV
    D === A || (diagview(D) .= real.(diagview(A)))
    one!(V)
    return D, V
end

function eigh_vals!(A::Diagonal, D, alg::DiagonalAlgorithm)
    check_input(eigh_vals!, A, D, alg)
    Ad = diagview(A)
    D === Ad || (D .= real.(Ad))
    return D
end

# GPU logic
# ---------
_gpu_heevj!(A::AbstractMatrix, Dd::AbstractVector, V::AbstractMatrix; kwargs...) =
    throw(MethodError(_gpu_heevj!, (A, Dd, V)))
_gpu_heevd!(A::AbstractMatrix, Dd::AbstractVector, V::AbstractMatrix; kwargs...) =
    throw(MethodError(_gpu_heevd!, (A, Dd, V)))
_gpu_heev!(A::AbstractMatrix, Dd::AbstractVector, V::AbstractMatrix; kwargs...) =
    throw(MethodError(_gpu_heev!, (A, Dd, V)))
_gpu_heevx!(A::AbstractMatrix, Dd::AbstractVector, V::AbstractMatrix; kwargs...) =
    throw(MethodError(_gpu_heevx!, (A, Dd, V)))

function eigh_full!(A::AbstractMatrix, DV, alg::GPU_EighAlgorithm)
    check_input(eigh_full!, A, DV, alg)
    D, V = DV
    Dd = D.diag
    if alg isa GPU_Jacobi
        _gpu_heevj!(A, Dd, V; alg.kwargs...)
    elseif alg isa GPU_DivideAndConquer
        _gpu_heevd!(A, Dd, V; alg.kwargs...)
    elseif alg isa GPU_QRIteration # alg isa GPU_QRIteration == GPU_Simple
        _gpu_heev!(A, Dd, V; alg.kwargs...)
    elseif alg isa GPU_Bisection # alg isa GPU_Bisection == GPU_Expert
        _gpu_heevx!(A, Dd, V; alg.kwargs...)
    else
        throw(ArgumentError("Unsupported eigh algorithm"))
    end
    # TODO: make this controllable using a `gaugefix` keyword argument
    V = gaugefix!(V)
    return D, V
end

function eigh_vals!(A::AbstractMatrix, D, alg::GPU_EighAlgorithm)
    check_input(eigh_vals!, A, D, alg)
    V = similar(A, (size(A, 1), 0))
    if alg isa GPU_Jacobi
        _gpu_heevj!(A, D, V; alg.kwargs...)
    elseif alg isa GPU_DivideAndConquer
        _gpu_heevd!(A, D, V; alg.kwargs...)
    elseif alg isa GPU_QRIteration
        _gpu_heev!(A, D, V; alg.kwargs...)
    elseif alg isa GPU_Bisection
        _gpu_heevx!(A, D, V; alg.kwargs...)
    else
        throw(ArgumentError("Unsupported eigh algorithm"))
    end
    return D
end
