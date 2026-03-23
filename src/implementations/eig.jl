# Inputs
# ------
function copy_input(::typeof(eig_full), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
copy_input(::typeof(eig_vals), A) = copy_input(eig_full, A)
copy_input(::Union{typeof(eig_trunc), typeof(eig_trunc_no_error)}, A) = copy_input(eig_full, A)

copy_input(::typeof(eig_full), A::Diagonal) = copy(A)

function check_input(::typeof(eig_full!), A::AbstractMatrix, DV, ::AbstractAlgorithm)
    m = LinearAlgebra.checksquare(A)
    D, V = DV
    @assert D isa Diagonal && V isa AbstractMatrix
    @check_size(D, (m, m))
    @check_scalar(D, A, complex)
    @check_size(V, (m, m))
    @check_scalar(V, A, complex)
    return nothing
end
function check_input(::typeof(eig_vals!), A::AbstractMatrix, D, ::AbstractAlgorithm)
    m = LinearAlgebra.checksquare(A)
    @assert D isa AbstractVector
    @check_size(D, (m,))
    @check_scalar(D, A, complex)
    return nothing
end

function check_input(::typeof(eig_full!), A::AbstractMatrix, DV, ::DiagonalAlgorithm)
    m = LinearAlgebra.checksquare(A)
    isdiag(A) || throw(DimensionMismatch("diagonal input matrix expected"))
    D, V = DV
    @assert D isa Diagonal && V isa AbstractMatrix
    @check_size(D, (m, m))
    @check_scalar(D, A, complex)
    @check_size(V, (m, m))
    @check_scalar(V, A, complex)
    return nothing
end
function check_input(::typeof(eig_vals!), A::AbstractMatrix, D, ::DiagonalAlgorithm)
    m = LinearAlgebra.checksquare(A)
    isdiag(A) || throw(DimensionMismatch("diagonal input matrix expected"))
    @assert D isa AbstractVector
    @check_size(D, (m,))
    @check_scalar(D, A, complex)
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
function initialize_output(::Union{typeof(eig_trunc!), typeof(eig_trunc_no_error!)}, A, alg::TruncatedAlgorithm)
    return initialize_output(eig_full!, A, alg.alg)
end

function initialize_output(::typeof(eig_full!), A::Diagonal, ::DiagonalAlgorithm)
    T = eltype(A)
    Tc = complex(T)
    D = T <: Complex ? A : Diagonal(similar(A, Tc, size(A, 1)))
    return D, similar(A, Tc, size(A))
end
function initialize_output(::typeof(eig_vals!), A::Diagonal, ::DiagonalAlgorithm)
    T = eltype(A)
    return T <: Complex ? diagview(A) : similar(A, complex(T), size(A, 1))
end

# DefaultAlgorithm intercepts
# ---------------------------
for f! in (:eig_full!, :eig_vals!, :eig_trunc!, :eig_trunc_no_error!)
    @eval function $f!(A::AbstractMatrix, alg::DefaultAlgorithm)
        return $f!(A, select_algorithm($f!, A, nothing; alg.kwargs...))
    end
    @eval function $f!(A::AbstractMatrix, out, alg::DefaultAlgorithm)
        return $f!(A, out, select_algorithm($f!, A, nothing; alg.kwargs...))
    end
end

# ==========================
#      IMPLEMENTATIONS
# ==========================

geev!(driver::Driver, args...; kwargs...) = throw(ArgumentError("$driver does not provide `geev!`"))
function geevx!(driver::Driver, A, Dd, V; kwargs...)
    @warn "$driver does not provide `geevx!`, falling back to `geev!`" maxlog = 1
    return geev!(driver, A, Dd, V)
end

# LAPACK implementations
for f! in (:geev!, :geevx!)
    @eval $f!(::LAPACK, args...; kwargs...) = YALAPACK.$f!(args...; kwargs...)
end

# driver dispatch
@inline eig_full_qr_iteration!(A, DV; driver::Driver = DefaultDriver(), kwargs...) =
    eig_full_qr_iteration!(driver, A, DV; kwargs...)
@inline eig_vals_qr_iteration!(A, D; driver::Driver = DefaultDriver(), kwargs...) =
    eig_vals_qr_iteration!(driver, A, D; kwargs...)

@inline eig_full_qr_iteration!(::DefaultDriver, A, DV; kwargs...) =
    eig_full_qr_iteration!(default_driver(QRIteration, A), A, DV; kwargs...)
@inline eig_vals_qr_iteration!(::DefaultDriver, A, D; kwargs...) =
    eig_vals_qr_iteration!(default_driver(QRIteration, A), A, D; kwargs...)

# Implementation
function eig_full_qr_iteration!(
        driver::Driver, A, DV;
        fixgauge::Bool = default_fixgauge(), scale::Bool = true, permute::Bool = true
    )
    D, V = DV
    Dd = diagview(D)
    (scale & permute) ? geev!(driver, A, Dd, V) : geevx!(driver, A, Dd, V; scale, permute)
    fixgauge && gaugefix!(eig_full!, V)
    return DV
end
function eig_vals_qr_iteration!(
        driver::Driver, A, D;
        fixgauge::Bool = default_fixgauge(), scale::Bool = true, permute::Bool = true
    )
    V = similar(A, complex(eltype(A)), (size(A, 1), 0))
    (scale & permute) ? geev!(driver, A, D, V) : geevx!(driver, A, D, V; scale, permute)
    return D
end

# Top-level QRIteration dispatch
function eig_full!(A::AbstractMatrix, DV, alg::QRIteration)
    check_input(eig_full!, A, DV, alg)
    eig_full_qr_iteration!(A, DV; alg.kwargs...)
    return DV
end
function eig_vals!(A::AbstractMatrix, D, alg::QRIteration)
    check_input(eig_vals!, A, D, alg)
    eig_vals_qr_iteration!(A, D; alg.kwargs...)
    return D
end

function eig_trunc!(A, DV, alg::TruncatedAlgorithm)
    D, V = eig_full!(A, DV, alg.alg)
    DVtrunc, ind = truncate(eig_trunc!, (D, V), alg.trunc)
    return DVtrunc..., truncation_error!(diagview(D), ind)
end

function eig_trunc_no_error!(A, DV, alg::TruncatedAlgorithm)
    D, V = eig_full!(A, DV, alg.alg)
    DVtrunc, ind = truncate(eig_trunc!, (D, V), alg.trunc)
    return DVtrunc
end

# Diagonal logic
# --------------
eig_sortby(x::T) where {T <: Number} = T <: Complex ? (real(x), imag(x)) : x
function eig_full!(A::Diagonal, DV, alg::DiagonalAlgorithm)
    check_input(eig_full!, A, DV, alg)
    D, V = DV
    diagA = diagview(A)
    I = sortperm(diagA; by = eig_sortby)
    if D === A
        permute!(diagA, I)
    else
        diagview(D) .= view(diagA, I)
    end
    zero!(V)
    n = size(A, 1)
    I .+= (0:(n - 1)) .* n
    V[I] .= Ref(one(eltype(V)))
    return D, V
end

function eig_vals!(A::Diagonal, D::AbstractVector, alg::DiagonalAlgorithm)
    check_input(eig_vals!, A, D, alg)
    Ad = diagview(A)
    D === Ad || copy!(D, Ad)
    sort!(D; by = eig_sortby)
    return D
end

# Deprecations
# ------------
for lapack_algtype in (:LAPACK_Simple, :LAPACK_Expert)
    @eval begin
        Base.@deprecate(
            eig_full!(A, DV, alg::$lapack_algtype),
            eig_full!(A, DV, QRIteration(; alg.kwargs...))
        )
        Base.@deprecate(
            eig_vals!(A, D, alg::$lapack_algtype),
            eig_vals!(A, D, QRIteration(; alg.kwargs...))
        )
    end
end
Base.@deprecate(
    eig_full!(A, DV, alg::CUSOLVER_Simple),
    eig_full!(A, DV, QRIteration(; driver = CUSOLVER(), alg.kwargs...))
)
Base.@deprecate(
    eig_vals!(A, D, alg::CUSOLVER_Simple),
    eig_vals!(A, D, QRIteration(; driver = CUSOLVER(), alg.kwargs...))
)
