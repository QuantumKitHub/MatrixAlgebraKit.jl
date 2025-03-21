# Inputs
# ------
function copy_input(::typeof(eig_full), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
function copy_input(::typeof(eig_vals), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
copy_input(::typeof(eig_trunc), A) = copy_input(eig_full, A)

function check_input(::typeof(eig_full!), A::AbstractMatrix, DV)
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
function check_input(::typeof(eig_vals!), A::AbstractMatrix, D)
    m, n = size(A)
    m == n || throw(DimensionMismatch("square input matrix expected"))
    @assert D isa AbstractVector
    @check_size(D, (n,))
    @check_scalar(D, A, complex)
    return nothing
end

# Outputs
# -------
function initialize_output(::typeof(eig_full!), A::AbstractMatrix, ::LAPACK_EigAlgorithm)
    n = size(A, 1) # square check will happen later
    Tc = complex(eltype(A))
    D = Diagonal(similar(A, Tc, n))
    V = similar(A, Tc, (n, n))
    return (D, V)
end
function initialize_output(::typeof(eig_vals!), A::AbstractMatrix, ::LAPACK_EigAlgorithm)
    n = size(A, 1) # square check will happen later
    Tc = complex(eltype(A))
    D = similar(A, Tc, n)
    return D
end
function initialize_output(::typeof(eig_trunc!), A::AbstractMatrix, alg::TruncatedAlgorithm)
    return initialize_output(eig_full!, A, alg.alg)
end

# Implementation
# --------------
# actual implementation
function eig_full!(A::AbstractMatrix, DV, alg::LAPACK_EigAlgorithm)
    check_input(eig_full!, A, DV)
    D, V = DV
    if alg isa LAPACK_Simple
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_Simple (geev) does not accept any keyword arguments"))
        YALAPACK.geev!(A, D.diag, V)
    else # alg isa LAPACK_Expert
        YALAPACK.geevx!(A, D.diag, V; alg.kwargs...)
    end
    # TODO: make this controllable using a `gaugefix` keyword argument
    for j in 1:size(V, 2)
        v = view(V, :, j)
        s = conj(sign(argmax(abs, v)))
        v .*= s
    end
    return D, V
end

function eig_vals!(A::AbstractMatrix, D, alg::LAPACK_EigAlgorithm)
    check_input(eig_vals!, A, D)
    V = similar(A, complex(eltype(A)), (size(A, 1), 0))
    if alg isa LAPACK_Simple
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_Simple (geev) does not accept any keyword arguments"))
        YALAPACK.geev!(A, D, V)
    else # alg isa LAPACK_Expert
        YALAPACK.geevx!(A, D, V; alg.kwargs...)
    end
    return D
end

function eig_trunc!(A::AbstractMatrix, DV, alg::TruncatedAlgorithm)
    D, V = eig_full!(A, DV, alg.alg)
    return truncate!(eig_trunc!, (D, V), alg.trunc)
end
