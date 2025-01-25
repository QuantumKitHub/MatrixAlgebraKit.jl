# TODO: export? or not export but mark as public ?
function eigh!(A::AbstractMatrix, args...; kwargs...)
    return eigh_full!(A, args...; kwargs...)
end

# copy input
function copy_input(::typeof(eigh_full), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
function copy_input(::typeof(eigh_vals), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end

# initialize output
function initialize_output(::typeof(eigh_full!), A::AbstractMatrix)
    n = size(A, 1) # square check will happen later
    D = Diagonal(similar(A, real(eltype(A)), n))
    V = similar(A, (n, n))
    return (D, V)
end
function initialize_output(::typeof(eigh_vals!), A::AbstractMatrix)
    n = size(A, 1) # square check will happen later
    D = similar(A, real(eltype(A)), n)
    return D
end

# select default algorithm
function select_algorithm(::typeof(eigh_full!), A::AbstractMatrix; kwargs...)
    return default_eigh_algorithm(A; kwargs...)
end
function select_algorithm(::typeof(eigh_vals!), A::AbstractMatrix; kwargs...)
    return default_eigh_algorithm(A; kwargs...)
end

function default_eigh_algorithm(A::StridedMatrix{T}; kwargs...) where {T<:BlasFloat}
    return LAPACK_MultipleRelativelyRobustRepresentations(; kwargs...)
end

# check input
function check_input(::typeof(eigh_full!), A::AbstractMatrix, DV)
    m, n = size(A)
    m == n || throw(ArgumentError("Eigenvalue decompsition requires square input matrix"))
    D, V = DV
    (V isa AbstractMatrix && eltype(V) == eltype(A) && size(V) == (m, m)) ||
        throw(ArgumentError("`eigh_full!` requires square V matrix with same size and `eltype` as A"))
    (D isa Diagonal && eltype(D) == real(eltype(A)) && size(D) == (m, m)) ||
        throw(ArgumentError("`eigh_full!` requires Diagonal matrix D with same size as A with a real `eltype`"))
    return nothing
end
function check_input(::typeof(eigh_vals!), A::AbstractMatrix, D)
    m, n = size(A)
    m == n || throw(ArgumentError("Eigenvalue decompsition requires square input matrix"))
    (size(D) == (n,) && eltype(D) == real(eltype(A))) ||
        throw(ArgumentError("Eigenvalue vector `D` must have length equal to size(A, 1) with a real `eltype`"))
    return nothing
end

# actual implementation
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
    return D, V
end

# for eigh_trunc!, it doesn't make sense to preallocate D and V as we don't know their sizes
# function select_algorithm(::typeof(eigh_trunc!), A::AbstractMatrix; kwargs...)
#     return default_eigh_algorithm(A; kwargs...)
# end

# function eigh_trunc!(A::AbstractMatrix, alg::LAPACK_EighAlgorithm,
#                      trunc::TruncationStrategy)
#     DV = eigh_full_init(A)
#     D, V = eigh_full!(A, DV, alg)

#     Dd = D.diag
#     ind = findtruncated(Dd, trunc)
#     V′ = V[:, ind]
#     D′ = Diagonal(Dd[ind])
#     return (D′, V′)
# end
