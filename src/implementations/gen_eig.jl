# Inputs
# ------
function copy_input(::typeof(gen_eig_full), A::AbstractMatrix, B::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A), copy!(similar(B, float(eltype(B))), B)
end
copy_input(::typeof(gen_eig_vals), A, B) = copy_input(gen_eig_full, A, B)

@noinline function _check_gen_eig_size(A, B)
    m = size(A, 1)
    n = size(B, 1)
    m == n || throw(DimensionMismatch(lazy"Expected matching input sizes, dimensions are $m and $n"))
    return m
end

function check_input(::typeof(gen_eig_full!), A::AbstractMatrix, B::AbstractMatrix, WV, ::AbstractAlgorithm)
    ma = LinearAlgebra.checksquare(A)
    mb = LinearAlgebra.checksquare(B)
    ma == mb || throw(DimensionMismatch(lazy"Expected matching input sizes, dimensions are $ma and $mb"))
    W, V = WV
    @assert W isa Diagonal && V isa AbstractMatrix
    @check_size(W, (ma, ma))
    @check_scalar(W, A, complex)
    @check_scalar(W, B, complex)
    @check_size(V, (ma, ma))
    @check_scalar(V, A, complex)
    @check_scalar(V, B, complex)
    return nothing
end
function check_input(::typeof(gen_eig_vals!), A::AbstractMatrix, B::AbstractMatrix, W, ::AbstractAlgorithm)
    ma = LinearAlgebra.checksquare(A)
    mb = LinearAlgebra.checksquare(B)
    ma == mb || throw(DimensionMismatch(lazy"Expected matching input sizes, dimensions are $ma and $mb"))
    @assert W isa AbstractVector
    @check_size(W, (na,))
    @check_scalar(W, A, complex)
    @check_scalar(W, B, complex)
    return nothing
end

# Outputs
# -------
function initialize_output(::typeof(gen_eig_full!), A::AbstractMatrix, B::AbstractMatrix, ::LAPACK_EigAlgorithm)
    n = size(A, 1) # square check will happen later
    Tc = complex(eltype(A))
    W = Diagonal(similar(A, Tc, n))
    V = similar(A, Tc, (n, n))
    return (W, V)
end
function initialize_output(::typeof(gen_eig_vals!), A::AbstractMatrix, B::AbstractMatrix, ::LAPACK_EigAlgorithm)
    n = size(A, 1) # square check will happen later
    Tc = complex(eltype(A))
    D = similar(A, Tc, n)
    return D
end

# Implementation
# --------------
# actual implementation
function gen_eig_full!(A::AbstractMatrix, B::AbstractMatrix, WV, alg::LAPACK_EigAlgorithm)
    check_input(gen_eig_full!, A, B, WV, alg)
    W, V = WV

    do_gauge_fix = get(alg.kwargs, :fixgauge, default_fixgauge())::Bool
    alg_kwargs = Base.structdiff(alg.kwargs, NamedTuple{(:fixgauge,)})

    if alg isa LAPACK_Simple
        isempty(alg_kwargs) ||
            throw(ArgumentError("invalid keyword arguments for LAPACK_Simple"))
        YALAPACK.ggev!(A, B, W.diag, V, similar(W.diag, eltype(A)))
    else # alg isa LAPACK_Expert
        throw(ArgumentError("LAPACK_Expert is not supported for ggev"))
    end

    do_gauge_fix && (V = gaugefix!(gen_eig_full!, V))

    return W, V
end

function gen_eig_vals!(A::AbstractMatrix, B::AbstractMatrix, W, alg::LAPACK_EigAlgorithm)
    check_input(gen_eig_vals!, A, B, W, alg)
    V = similar(A, complex(eltype(A)), (size(A, 1), 0))

    alg_kwargs = Base.structdiff(alg.kwargs, NamedTuple{(:fixgauge,)})

    if alg isa LAPACK_Simple
        isempty(alg_kwargs) ||
            throw(ArgumentError("invalid keyword arguments for LAPACK_Simple"))
        YALAPACK.ggev!(A, B, W, V, similar(W, eltype(A)))
    else # alg isa LAPACK_Expert
        throw(ArgumentError("LAPACK_Expert is not supported for ggev"))
    end

    return W
end
