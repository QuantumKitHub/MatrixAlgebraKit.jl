# Inputs
# ------
function copy_input(::typeof(eig_full), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
copy_input(::typeof(eig_vals), A) = copy_input(eig_full, A)
copy_input(::typeof(eig_trunc), A) = copy_input(eig_full, A)

copy_input(::typeof(eig_full), A::Diagonal) = copy(A)

function check_input(::typeof(eig_full!), A::AbstractMatrix, DV, ::AbstractAlgorithm)
    m, n = size(A)
    m == n || throw(DimensionMismatch("square input matrix expected"))
    D, V = DV
    @assert D isa Diagonal && V isa AbstractMatrix
    @check_size(D, (m, m))
    @check_scalar(D, A, complex)
    @check_size(V, (m, m))
    @check_scalar(V, A, complex)
    return nothing
end
function check_input(::typeof(eig_vals!), A::AbstractMatrix, D, ::AbstractAlgorithm)
    m, n = size(A)
    m == n || throw(DimensionMismatch("square input matrix expected"))
    @assert D isa AbstractVector
    @check_size(D, (n,))
    @check_scalar(D, A, complex)
    return nothing
end

function check_input(::typeof(eig_full!), A::AbstractMatrix, DV, ::DiagonalAlgorithm)
    m, n = size(A)
    @assert m == n && isdiag(A)
    D, V = DV
    @assert D isa Diagonal && V isa Diagonal
    @check_size(D, (m, m))
    @check_size(V, (m, m))
    # Diagonal doesn't need to promote to complex scalartype since we know it is diagonalizable
    @check_scalar(D, A)
    @check_scalar(V, A)
    return nothing
end
function check_input(::typeof(eig_vals!), A::AbstractMatrix, D, ::DiagonalAlgorithm)
    m, n = size(A)
    @assert m == n && isdiag(A)
    @assert D isa AbstractVector
    @check_size(D, (n,))
    # Diagonal doesn't need to promote to complex scalartype since we know it is diagonalizable
    @check_scalar(D, A)
    return nothing
end

# Outputs
# -------
function initialize_output(::typeof(eig_full!), A::AbstractMatrix, ::AbstractAlgorithm)
    n = size(A, 1) # square check will happen later
    Tc = complex(eltype(A))
    D = Diagonal(similar(A, Tc, n))
    V = similar(A, Tc, (n, n))
    return (D, V)
end
function initialize_output(::typeof(eig_vals!), A::AbstractMatrix, ::AbstractAlgorithm)
    n = size(A, 1) # square check will happen later
    Tc = complex(eltype(A))
    D = similar(A, Tc, n)
    return D
end
function initialize_output(::typeof(eig_trunc!), A, alg::TruncatedAlgorithm)
    DV = initialize_output(eig_full!, A, alg.alg)
    ϵ = similar(A, real(eltype(A)), alg.compute_error)
    return (DV..., ϵ)
end

function initialize_output(::typeof(eig_full!), A::Diagonal, ::DiagonalAlgorithm)
    return A, similar(A)
end
function initialize_output(::typeof(eig_vals!), A::Diagonal, ::DiagonalAlgorithm)
    return diagview(A)
end

# Implementation
# --------------
function eig_full!(A::AbstractMatrix, DV, alg::LAPACK_EigAlgorithm)
    check_input(eig_full!, A, DV, alg)
    D, V = DV

    do_gauge_fix = get(alg.kwargs, :fixgauge, default_fixgauge())::Bool
    alg_kwargs = Base.structdiff(alg.kwargs, NamedTuple{(:fixgauge,)})

    if alg isa LAPACK_Simple
        isempty(alg_kwargs) ||
            throw(ArgumentError("invalid keyword arguments for LAPACK_Simple"))
        YALAPACK.geev!(A, D.diag, V)
    else # alg isa LAPACK_Expert
        YALAPACK.geevx!(A, D.diag, V; alg_kwargs...)
    end

    do_gauge_fix && (V = gaugefix!(eig_full!, V))

    return D, V
end

function eig_vals!(A::AbstractMatrix, D, alg::LAPACK_EigAlgorithm)
    check_input(eig_vals!, A, D, alg)
    V = similar(A, complex(eltype(A)), (size(A, 1), 0))

    alg_kwargs = Base.structdiff(alg.kwargs, NamedTuple{(:fixgauge,)})

    if alg isa LAPACK_Simple
        isempty(alg_kwargs) ||
            throw(ArgumentError("invalid keyword arguments for LAPACK_Simple"))
        YALAPACK.geev!(A, D, V)
    else # alg isa LAPACK_Expert
        YALAPACK.geevx!(A, D, V; alg_kwargs...)
    end

    return D
end

function eig_trunc!(A, DVϵ, alg::TruncatedAlgorithm)
    D, V, ϵ = DVϵ
    D, V = eig_full!(A, (D, V), alg.alg)
    DVtrunc, ind = truncate(eig_trunc!, (D, V), alg.trunc)
    if !isempty(ϵ)
        ϵ .= truncation_error!(diagview(D), ind)
    end
    return DVtrunc..., ϵ
end

# Diagonal logic
# --------------
function eig_full!(A::Diagonal, (D, V)::Tuple{Diagonal, Diagonal}, alg::DiagonalAlgorithm)
    check_input(eig_full!, A, (D, V), alg)
    D === A || copy!(D, A)
    one!(V)
    return D, V
end

function eig_vals!(A::Diagonal, D::AbstractVector, alg::DiagonalAlgorithm)
    check_input(eig_vals!, A, D, alg)
    Ad = diagview(A)
    D === Ad || copy!(D, Ad)
    return D
end

# GPU logic
# ---------
_gpu_geev!(A, D, V) = throw(MethodError(_gpu_geev!, (A, D, V)))

function eig_full!(A::AbstractMatrix, DV, alg::GPU_EigAlgorithm)
    check_input(eig_full!, A, DV, alg)
    D, V = DV

    do_gauge_fix = get(alg.kwargs, :fixgauge, default_fixgauge())::Bool
    alg_kwargs = Base.structdiff(alg.kwargs, NamedTuple{(:fixgauge,)})

    if alg isa GPU_Simple
        isempty(alg_kwargs) || @warn "invalid keyword arguments for GPU_Simple"
        _gpu_geev!(A, D.diag, V)
    end

    do_gauge_fix && (V = gaugefix!(eig_full!, V))

    return D, V
end

function eig_vals!(A::AbstractMatrix, D, alg::GPU_EigAlgorithm)
    check_input(eig_vals!, A, D, alg)
    V = similar(A, complex(eltype(A)), (size(A, 1), 0))

    alg_kwargs = Base.structdiff(alg.kwargs, NamedTuple{(:fixgauge,)})

    if alg isa GPU_Simple
        isempty(alg_kwargs) || @warn "invalid keyword arguments for GPU_Simple"
        _gpu_geev!(A, D, V)
    end

    return D
end
