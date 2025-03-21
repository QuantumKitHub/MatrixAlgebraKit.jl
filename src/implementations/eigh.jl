# Inputs
# ------
function copy_input(::typeof(eigh_full), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
function copy_input(::typeof(eigh_vals), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
copy_input(::typeof(eigh_trunc), A) = copy_input(eigh_full, A)

function check_input(::typeof(eigh_full!), A::AbstractMatrix, DV)
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
function check_input(::typeof(eigh_vals!), A::AbstractMatrix, D)
    m, n = size(A)
    @assert D isa AbstractVector
    @check_size(D, (n,))
    @check_scalar(D, A, real)
    return nothing
end

# Outputs
# -------
function initialize_output(::typeof(eigh_full!), A::AbstractMatrix, ::LAPACK_EighAlgorithm)
    n = size(A, 1) # square check will happen later
    D = Diagonal(similar(A, real(eltype(A)), n))
    V = similar(A, (n, n))
    return (D, V)
end
function initialize_output(::typeof(eigh_vals!), A::AbstractMatrix, ::LAPACK_EighAlgorithm)
    n = size(A, 1) # square check will happen later
    D = similar(A, real(eltype(A)), n)
    return D
end
function initialize_output(::typeof(eigh_trunc!), A::AbstractMatrix,
                           alg::TruncatedAlgorithm)
    return initialize_output(eigh_full!, A, alg.alg)
end

# Implementation
# --------------
function eigh_full!(A::AbstractMatrix, DV, alg::LAPACK_EighAlgorithm)
    check_input(eigh_full!, A, DV)
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
    for j in 1:size(V, 2)
        v = view(V, :, j)
        s = conj(sign(argmax(abs, v)))
        v .*= s
    end
    return D, V
end

function eigh_vals!(A::AbstractMatrix, D, alg::LAPACK_EighAlgorithm)
    check_input(eigh_vals!, A, D)
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

function eigh_trunc!(A::AbstractMatrix, DV, alg::TruncatedAlgorithm)
    D, V = eigh_full!(A, DV, alg.alg)
    return truncate!(eigh_trunc!, (D, V), alg.trunc)
end
