# Inputs
# ------
function copy_input(::typeof(eigh_full), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
copy_input(::typeof(eigh_vals), A) = copy_input(eigh_full, A)
copy_input(::Union{typeof(eigh_trunc), typeof(eigh_trunc_no_error)}, A) = copy_input(eigh_full, A)

copy_input(::typeof(eigh_full), A::Diagonal) = copy(A)

check_hermitian(A, ::AbstractAlgorithm) = check_hermitian(A)
check_hermitian(A, alg::Algorithm) = check_hermitian(A; atol = get(alg.kwargs, :hermitian_tol, default_hermitian_tol(A)))
function check_hermitian(A; atol::Real = default_hermitian_tol(A), rtol::Real = 0)
    m, n = size(A)
    m == n || throw(DimensionMismatch("square input matrix expected"))
    ishermitian(A; atol, rtol) ||
        throw(DomainError(A, "Hermitian matrix was expected. Use `project_hermitian` to project onto the nearest hermitian matrix."))
    return nothing
end

function check_input(::typeof(eigh_full!), A::AbstractMatrix, DV, alg::AbstractAlgorithm)
    check_hermitian(A, alg)
    D, V = DV
    m = size(A, 1)
    @assert D isa Diagonal && V isa AbstractMatrix
    @check_size(D, (m, m))
    @check_scalar(D, A, real)
    @check_size(V, (m, m))
    @check_scalar(V, A)
    return nothing
end
function check_input(::typeof(eigh_vals!), A::AbstractMatrix, D, alg::AbstractAlgorithm)
    check_hermitian(A, alg)
    m = size(A, 1)
    @assert D isa AbstractVector
    @check_size(D, (m,))
    @check_scalar(D, A, real)
    return nothing
end

function check_input(::typeof(eigh_full!), A::AbstractMatrix, DV, alg::DiagonalAlgorithm)
    check_hermitian(A, alg)
    @assert isdiag(A)
    m = size(A, 1)
    D, V = DV
    @assert D isa Diagonal
    @check_size(D, (m, m))
    @check_scalar(D, A, real)
    @check_size(V, (m, m))
    @check_scalar(V, A)
    return nothing
end

function check_input(::typeof(eigh_vals!), A::AbstractMatrix, D, alg::DiagonalAlgorithm)
    check_hermitian(A, alg)
    @assert isdiag(A)
    m = size(A, 1)
    @assert D isa AbstractVector
    @check_size(D, (m,))
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
function initialize_output(::Union{typeof(eigh_trunc!), typeof(eigh_trunc_no_error!)}, A, alg::TruncatedAlgorithm)
    return initialize_output(eigh_full!, A, alg.alg)
end

function initialize_output(::typeof(eigh_full!), A::Diagonal, ::DiagonalAlgorithm)
    return eltype(A) <: Real ? A : similar(A, real(eltype(A))), similar(A, size(A)...)
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

    do_gauge_fix = get(alg.kwargs, :fixgauge, default_fixgauge())::Bool
    alg_kwargs = Base.structdiff(alg.kwargs, NamedTuple{(:fixgauge,)})

    if alg isa LAPACK_MultipleRelativelyRobustRepresentations
        YALAPACK.heevr!(A, Dd, V; alg_kwargs...)
    elseif alg isa LAPACK_DivideAndConquer
        YALAPACK.heevd!(A, Dd, V; alg_kwargs...)
    elseif alg isa LAPACK_Simple
        YALAPACK.heev!(A, Dd, V; alg_kwargs...)
    else # alg isa LAPACK_Expert
        YALAPACK.heevx!(A, Dd, V; alg_kwargs...)
    end

    do_gauge_fix && (V = gaugefix!(eigh_full!, V))

    return D, V
end

function eigh_vals!(A::AbstractMatrix, D, alg::LAPACK_EighAlgorithm)
    check_input(eigh_vals!, A, D, alg)
    V = similar(A, (size(A, 1), 0))

    alg_kwargs = Base.structdiff(alg.kwargs, NamedTuple{(:fixgauge,)})

    if alg isa LAPACK_MultipleRelativelyRobustRepresentations
        YALAPACK.heevr!(A, D, V; alg_kwargs...)
    elseif alg isa LAPACK_DivideAndConquer
        YALAPACK.heevd!(A, D, V; alg_kwargs...)
    elseif alg isa LAPACK_QRIteration # == LAPACK_Simple
        YALAPACK.heev!(A, D, V; alg_kwargs...)
    else # alg isa LAPACK_Bisection == LAPACK_Expert
        YALAPACK.heevx!(A, D, V; alg_kwargs...)
    end

    return D
end

function eigh_trunc!(A, DV, alg::TruncatedAlgorithm)
    D, V = eigh_full!(A, DV, alg.alg)
    DVtrunc, ind = truncate(eigh_trunc!, (D, V), alg.trunc)
    return DVtrunc..., truncation_error!(diagview(D), ind)
end

function eigh_trunc_no_error!(A, DV, alg::TruncatedAlgorithm)
    D, V = eigh_full!(A, DV, alg.alg)
    DVtrunc, ind = truncate(eigh_trunc!, (D, V), alg.trunc)
    return DVtrunc
end

permute_V_cols!(V, I::Vector{Int}) = Base.permutecols!!(V, I)

# Diagonal logic
# --------------
function eigh_full!(A::Diagonal, DV, alg::DiagonalAlgorithm)
    check_input(eigh_full!, A, DV, alg)
    D, V = DV
    I = sortperm(diagview(A); by = real)
    if D === A
        sort!(diagview(A))
    else
        diagview(D) .= real.(diagview(A))[I]
    end
    zero!(V)
    V = permute_V_cols!(V, I)
    return D, V
end

function eigh_vals!(A::Diagonal, D, alg::DiagonalAlgorithm)
    check_input(eigh_vals!, A, D, alg)
    Ad = diagview(A)
    if D === Ad
        sort!(Ad)
    else
        D .= real.(Ad)
        sort!(D)
    end
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

    do_gauge_fix = get(alg.kwargs, :fixgauge, default_fixgauge())::Bool
    alg_kwargs = Base.structdiff(alg.kwargs, NamedTuple{(:fixgauge,)})

    if alg isa GPU_Jacobi
        _gpu_heevj!(A, Dd, V; alg_kwargs...)
    elseif alg isa GPU_DivideAndConquer
        _gpu_heevd!(A, Dd, V; alg_kwargs...)
    elseif alg isa GPU_QRIteration # alg isa GPU_QRIteration == GPU_Simple
        _gpu_heev!(A, Dd, V; alg_kwargs...)
    elseif alg isa GPU_Bisection # alg isa GPU_Bisection == GPU_Expert
        _gpu_heevx!(A, Dd, V; alg_kwargs...)
    else
        throw(ArgumentError("Unsupported eigh algorithm"))
    end

    do_gauge_fix && (V = gaugefix!(eigh_full!, V))

    return D, V
end

function eigh_vals!(A::AbstractMatrix, D, alg::GPU_EighAlgorithm)
    check_input(eigh_vals!, A, D, alg)
    V = similar(A, (size(A, 1), 0))

    alg_kwargs = Base.structdiff(alg.kwargs, NamedTuple{(:fixgauge,)})

    if alg isa GPU_Jacobi
        _gpu_heevj!(A, D, V; alg_kwargs...)
    elseif alg isa GPU_DivideAndConquer
        _gpu_heevd!(A, D, V; alg_kwargs...)
    elseif alg isa GPU_QRIteration
        _gpu_heev!(A, D, V; alg_kwargs...)
    elseif alg isa GPU_Bisection
        _gpu_heevx!(A, D, V; alg_kwargs...)
    else
        throw(ArgumentError("Unsupported eigh algorithm"))
    end

    return D
end
