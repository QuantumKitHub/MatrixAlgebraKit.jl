# TODO: export? or not export but mark as public ?
function eig!(A::AbstractMatrix, args...; kwargs...)
    return eig_full!(A, args...; kwargs...)
end

function eig_full!(A::AbstractMatrix, DV=eig_full_init(A); kwargs...)
    return eig_full!(A, DV, default_algorithm(eig_full!, A; kwargs...))
end
function eig_vals!(A::AbstractMatrix, D=eig_vals_init(A); kwargs...)
    return eig_vals!(A, D, default_algorithm(eig_vals!, A; kwargs...))
end
function eig_trunc!(A::AbstractMatrix; kwargs...)
    return eig_trunc!(A, default_algorithm(eig_trunc!, A; kwargs...))
end

function eig_full_init(A::AbstractMatrix)
    n = size(A, 1) # square check will happen later
    D = Diagonal(similar(A, eltype(A), n))
    V = similar(A, (n, n))
    return (D, V)
end
function eig_vals_init(A::AbstractMatrix)
    n = size(A, 1) # square check will happen later
    D = similar(A, eltype(A), n)
    return D
end

function default_algorithm(::typeof(eig_full!), A::AbstractMatrix; kwargs...)
    return default_eig_algorithm(A; kwargs...)
end
function default_algorithm(::typeof(eig_vals!), A::AbstractMatrix; kwargs...)
    return default_eig_algorithm(A; kwargs...)
end
function default_algorithm(::typeof(eig_trunc!), A::AbstractMatrix; kwargs...)
    return default_eig_algorithm(A; kwargs...)
end

function default_eig_algorithm(A::StridedMatrix{T}; kwargs...) where {T<:BlasFloat}
    return LAPACK_RobustRepresentations(; kwargs...)
end

function check_eig_full_input(A::AbstractMatrix, DV)
    m, n = size(A)
    m == n || throw(ArgumentError("Eigenvalue decompsition requires square input matrix"))
    D, V = DV
    (V isa AbstractMatrix && eltype(V) == eltype(A) && size(V) == (m, m)) ||
        throw(ArgumentError("`eig_full!` requires square V matrix with same size and `eltype` as A"))
    (D isa Diagonal && eltype(D) == eltype(A) && size(D) == (m, m)) ||
        throw(ArgumentError("`eig_full!` requires Diagonal matrix D with same size and `eltype` as A"))
    return nothing
end
function check_eig_vals_input(A::AbstractMatrix, (D, V))
    m, n = size(A)
    m == n || throw(ArgumentError("Eigenvalue decompsition requires square input matrix"))
    size(D) == (n,) && eltype(D) == eltype(A) ||
        throw(ArgumentError("Eigenvalue vector `D` must have length equal to size(A, 1) and same `eltype` as A"))
    return nothing
end

const LAPACK_EighAlgorithm = Union{LAPACK_RobustRepresentations,LAPACK_QRIteration,
                                   LAPACK_DivideAndConquer}
function eig_full!(A::AbstractMatrix, DV, alg::LAPACK_EighAlgorithm)
    check_eig_full_input(A, DV)
    D, V = DV
    if alg isa LAPACK_RobustRepresentations
        YALAPACK.heevr!(A, D, V; alg.kwargs...)
    elseif alg isa LAPACK_DivideAndConquer
        YALAPACK.heevd!(A, D, V; alg.kwargs...)
    else
        YALAPACK.heev!(A, D, V; alg.kwargs...)
    end
    return D, V
end

function eig_vals!(A::AbstractMatrix, D, alg::LAPACK_EighAlgorithm)
    check_eig_vals_input(A, D)
    V = similar(A, (size(A, 1), 0))
    if alg isa LAPACK_RobustRepresentations
        YALAPACK.heevr!(A, D, V; alg.kwargs...)
    elseif alg isa LAPACK_DivideAndConquer
        YALAPACK.heevd!(A, D, V; alg.kwargs...)
    else
        YALAPACK.heev!(A, D, V; alg.kwargs...)
    end
    return D, V
end

# for eig_trunc!, it doesn't make sense to preallocate D and V as we don't know their sizes
function eig_trunc!(A::AbstractMatrix, alg::LAPACK_EighAlgorithm,
                    trunc::TruncationStrategy)
    DV = eig_full_init(A)
    D, V = eig_full!(A, DV, alg)

    Dd = D.diag
    ind = findtruncated(Dd, trunc)
    V′ = V[:, ind]
    D′ = Diagonal(Dd[ind])
    return (D′, V′)
end
