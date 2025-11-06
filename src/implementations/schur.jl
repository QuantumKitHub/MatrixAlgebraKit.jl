# Inputs
# ------
copy_input(::typeof(schur_full), A) = copy_input(eig_full, A)
copy_input(::typeof(schur_vals), A) = copy_input(eig_vals, A)

# check input
function check_input(::typeof(schur_full!), A::AbstractMatrix, TZv, ::AbstractAlgorithm)
    m, n = size(A)
    m == n || throw(DimensionMismatch("square input matrix expected"))
    T, Z, vals = TZv
    @assert T isa AbstractMatrix && Z isa AbstractMatrix && vals isa AbstractVector
    @check_size(T, (m, m))
    @check_scalar(T, A)
    @check_size(Z, (m, m))
    @check_scalar(Z, A)
    @check_size(vals, (n,))
    @check_scalar(vals, A, complex)
    return nothing
end
function check_input(::typeof(schur_vals!), A::AbstractMatrix, vals, ::AbstractAlgorithm)
    m, n = size(A)
    m == n || throw(DimensionMismatch("square input matrix expected"))
    @assert vals isa AbstractVector
    @check_size(vals, (n,))
    @check_scalar(vals, A, complex)
    return nothing
end

function check_input(::typeof(schur_full!), A::AbstractMatrix, TZv, ::DiagonalAlgorithm)
    m, n = size(A)
    @assert m == n && isdiag(A)
    T, Z, vals = TZv
    @assert vals isa AbstractVector && Z isa Diagonal
    @check_scalar(T, A)
    @check_size(Z, (m, m))
    @check_scalar(Z, A)
    @check_size(vals, (n,))
    # Diagonal doesn't need to promote to complex scalartype since we know it is diagonalizable
    @check_scalar(vals, A)
    return nothing
end
function check_input(::typeof(schur_vals!), A::AbstractMatrix, vals, ::DiagonalAlgorithm)
    m, n = size(A)
    @assert m == n && isdiag(A)
    @assert vals isa AbstractVector
    @check_size(vals, (n,))
    # Diagonal doesn't need to promote to complex scalartype since we know it is diagonalizable
    @check_scalar(vals, A)
    return nothing
end

# Outputs
# -------
function initialize_output(::typeof(schur_full!), A::AbstractMatrix, ::AbstractAlgorithm)
    n = size(A, 1) # square check will happen later
    Z = similar(A, (n, n))
    vals = similar(A, complex(eltype(A)), n)
    return (A, Z, vals)
end
function initialize_output(::typeof(schur_vals!), A::AbstractMatrix, ::AbstractAlgorithm)
    n = size(A, 1) # square check will happen later
    vals = similar(A, complex(eltype(A)), n)
    return vals
end
function initialize_output(::typeof(schur_full!), A::Diagonal, ::DiagonalAlgorithm)
    n = size(A, 1)
    Z = similar(A)
    vals = similar(A, eltype(A), n)
    return (A, Z, vals)
end
function initialize_output(::typeof(schur_vals!), A::Diagonal, ::DiagonalAlgorithm)
    n = size(A, 1)
    vals = similar(A, eltype(A), n)
    return vals
end

# Implementation
# --------------
function schur_full!(A::AbstractMatrix, TZv, alg::LAPACK_EigAlgorithm)
    check_input(schur_full!, A, TZv, alg)
    T, Z, vals = TZv
    if alg isa LAPACK_Simple
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_Simple Schur (gees) does not accept any keyword arguments"))
        YALAPACK.gees!(A, Z, vals)
    else # alg isa LAPACK_Expert
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_Expert Schur (geesx) does not accept any keyword arguments"))
        YALAPACK.geesx!(A, Z, vals)
    end
    T === A || copy!(T, A)
    return T, Z, vals
end

function schur_vals!(A::AbstractMatrix, vals, alg::LAPACK_EigAlgorithm)
    check_input(schur_vals!, A, vals, alg)
    Z = similar(A, eltype(A), (size(A, 1), 0))
    if alg isa LAPACK_Simple
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_Simple (gees) does not accept any keyword arguments"))
        YALAPACK.gees!(A, Z, vals)
    else # alg isa LAPACK_Expert
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_Expert (geesx) does not accept any keyword arguments"))
        YALAPACK.geesx!(A, Z, vals)
    end
    return vals
end

# Diagonal logic
# --------------
function schur_full!(A::Diagonal, (T, Z, vals)::Tuple{Diagonal, Diagonal, <:AbstractVector}, alg::DiagonalAlgorithm)
    check_input(schur_full!, A, (T, Z, vals), alg)
    copy!(vals, diagview(A))
    one!(Z)
    T === A || copy!(T, A)
    return T, Z, vals
end

function schur_vals!(A::Diagonal, vals::AbstractVector, alg::DiagonalAlgorithm)
    check_input(schur_vals!, A, vals, alg)
    Ad = diagview(A)
    vals === Ad || copy!(vals, Ad)
    return vals
end
