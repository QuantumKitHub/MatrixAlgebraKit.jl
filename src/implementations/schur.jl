# Inputs
# ------
copy_input(::typeof(schur_full), A::AbstractMatrix) = copy_input(eig_full, A)
copy_input(::typeof(schur_vals), A::AbstractMatrix) = copy_input(eig_vals, A)

# check input
function check_input(::typeof(schur_full!), A::AbstractMatrix, TZv)
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
function check_input(::typeof(schur_vals!), A::AbstractMatrix, vals)
    m, n = size(A)
    m == n || throw(DimensionMismatch("square input matrix expected"))
    @assert vals isa AbstractVector
    @check_size(vals, (n,))
    @check_scalar(vals, A, complex)
    return nothing
end

# Outputs
# -------
function initialize_output(::typeof(schur_full!), A::AbstractMatrix, ::LAPACK_EigAlgorithm)
    n = size(A, 1) # square check will happen later
    Z = similar(A, (n, n))
    vals = similar(A, complex(eltype(A)), n)
    return (A, Z, vals)
end
function initialize_output(::typeof(schur_vals!), A::AbstractMatrix, ::LAPACK_EigAlgorithm)
    n = size(A, 1) # square check will happen later
    vals = similar(A, complex(eltype(A)), n)
    return vals
end

# Implementation
# --------------
function schur_full!(A::AbstractMatrix, TZv, alg::LAPACK_EigAlgorithm)
    check_input(schur_full!, A, TZv)
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
    check_input(schur_vals!, A, vals)
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
