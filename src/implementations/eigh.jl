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
    LinearAlgebra.checksquare(A)
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

# DefaultAlgorithm intercepts
# ---------------------------
for f! in (:eigh_full!, :eigh_vals!, :eigh_trunc!, :eigh_trunc_no_error!)
    @eval function $f!(A, alg::DefaultAlgorithm)
        return $f!(A, select_algorithm($f!, A, nothing; alg.kwargs...))
    end
    @eval function $f!(A, out, alg::DefaultAlgorithm)
        return $f!(A, out, select_algorithm($f!, A, nothing; alg.kwargs...))
    end
end

# ==========================
#      IMPLEMENTATIONS
# ==========================

for f! in (:heevr!, :heevd!, :heev!, :heevx!, :heevj!)
    @eval $f!(driver::Driver, args...) = throw(ArgumentError("$driver does not provide $f!"))
end

# LAPACK implementations
for f! in (:heevr!, :heevd!, :heev!, :heevx!)
    @eval $f!(::LAPACK, args...; kwargs...) = YALAPACK.$f!(args...; kwargs...)
end

supports_eigh(::Driver, ::Symbol) = false
supports_eigh(::LAPACK, f::Symbol) = f in (:mrrr, :divide_and_conquer, :qr_iteration, :bisection)
supports_eigh(::GLA, f::Symbol) = f === :qr_iteration
supports_eigh(::CUSOLVER, f::Symbol) = f in (:jacobi, :divide_and_conquer)
supports_eigh(::ROCSOLVER, f::Symbol) = f in (:jacobi, :divide_and_conquer, :qr_iteration, :bisection)

for (f, f_lapack!, Alg) in (
        (:mrrr, :heevr!, :MultipleRelativelyRobustRepresentations),
        (:divide_and_conquer, :heevd!, :DivideAndConquer),
        (:qr_iteration, :heev!, :QRIteration),
        (:bisection, :heevx!, :Bisection),
        (:jacobi, :heevj!, :Jacobi),
    )
    f_eigh_full! = Symbol(f, :_eigh_full!)
    f_eigh_vals! = Symbol(f, :_eigh_vals!)

    # MatrixAlgebraKit wrappers
    @eval begin
        function eigh_full!(A::AbstractMatrix, DV, alg::$Alg)
            check_input(eigh_full!, A, DV, alg)
            D, V = DV
            Dd, V = $f_eigh_full!(A, D.diag, V; alg.kwargs...)
            return D, V
        end
        function eigh_vals!(A::AbstractMatrix, D, alg::$Alg)
            check_input(eigh_vals!, A, D, alg)
            V = similar(A, (size(A, 1), 0))
            $f_eigh_vals!(A, D, V; alg.kwargs...)
            return D
        end
    end

    # driver dispatch
    @eval begin
        @inline $f_eigh_full!(A, Dd, V; driver::Driver = DefaultDriver(), kwargs...) =
            $f_eigh_full!(driver, A, Dd, V; kwargs...)
        @inline $f_eigh_vals!(A, D, V; driver::Driver = DefaultDriver(), kwargs...) =
            $f_eigh_vals!(driver, A, D, V; kwargs...)

        @inline $f_eigh_full!(::DefaultDriver, A, Dd, V; kwargs...) =
            $f_eigh_full!(default_driver($Alg, A), A, Dd, V; kwargs...)
        @inline $f_eigh_vals!(::DefaultDriver, A, D, V; kwargs...) =
            $f_eigh_vals!(default_driver($Alg, A), A, D, V; kwargs...)
    end

    # Implementation
    @eval begin
        function $f_eigh_full!(driver::Driver, A, Dd, V; fixgauge::Bool = default_fixgauge(), kwargs...)
            supports_eigh(driver, $(QuoteNode(f))) ||
                throw(ArgumentError(LazyString("driver ", driver, " does not provide `$($(QuoteNode(f_lapack!)))`")))
            $f_lapack!(driver, A, Dd, V; kwargs...)
            fixgauge && gaugefix!(eigh_full!, V)
            return Dd, V
        end
        function $f_eigh_vals!(driver::Driver, A, D, V; fixgauge::Bool = default_fixgauge(), kwargs...)
            supports_eigh(driver, $(QuoteNode(f))) ||
                throw(ArgumentError(LazyString("driver ", driver, " does not provide `$($(QuoteNode(f_lapack!)))`")))
            $f_lapack!(driver, A, D, V; kwargs...)
            return D
        end
    end
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

# Diagonal logic
# --------------
function eigh_full!(A::Diagonal, DV, alg::DiagonalAlgorithm)
    check_input(eigh_full!, A, DV, alg)
    D, V = DV
    diagA = diagview(A)
    I = sortperm(diagA; by = real)
    if D === A
        permute!(diagA, I)
    else
        diagview(D) .= real.(view(diagA, I))
    end
    zero!(V)
    n = size(A, 1)
    I .+= (0:(n - 1)) .* n
    V[I] .= Ref(one(eltype(V)))
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

# Deprecations
# ------------
for algtype in (:MultipleRelativelyRobustRepresentations, :DivideAndConquer, :QRIteration, :Bisection)
    lapack_algtype = Symbol(:LAPACK_, algtype)
    @eval begin
        Base.@deprecate(
            eigh_full!(A, DV, alg::$lapack_algtype),
            eigh_full!(A, DV, $algtype(; driver = LAPACK(), alg.kwargs...))
        )
        Base.@deprecate(
            eigh_vals!(A, D, alg::$lapack_algtype),
            eigh_vals!(A, D, $algtype(; driver = LAPACK(), alg.kwargs...))
        )
    end
end
for (algtype, newtype, drivertype) in (
        (:CUSOLVER_DivideAndConquer, :DivideAndConquer, :CUSOLVER),
        (:CUSOLVER_Jacobi, :Jacobi, :CUSOLVER),
        (:ROCSOLVER_DivideAndConquer, :DivideAndConquer, :ROCSOLVER),
        (:ROCSOLVER_QRIteration, :QRIteration, :ROCSOLVER),
        (:ROCSOLVER_Bisection, :Bisection, :ROCSOLVER),
        (:ROCSOLVER_Jacobi, :Jacobi, :ROCSOLVER),
    )
    @eval begin
        Base.@deprecate(
            eigh_full!(A, DV, alg::$algtype),
            eigh_full!(A, DV, $newtype(; driver = $drivertype(), alg.kwargs...))
        )
        Base.@deprecate(
            eigh_vals!(A, D, alg::$algtype),
            eigh_vals!(A, D, $newtype(; driver = $drivertype(), alg.kwargs...))
        )
    end
end
Base.@deprecate(
    eigh_full!(A, DV, alg::GLA_QRIteration),
    eigh_full!(A, DV, QRIteration(; driver = GLA(), alg.kwargs...))
)
Base.@deprecate(
    eigh_vals!(A, D, alg::GLA_QRIteration),
    eigh_vals!(A, D, QRIteration(; driver = GLA(), alg.kwargs...))
)
