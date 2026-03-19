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
for f! in (:geev!, :geevx!)
    @eval $f!(driver::Driver, args...) = throw(ArgumentError("$driver does not provide $f!"))
end

# LAPACK implementations
for f! in (:geev!, :geevx!)
    @eval $f!(::LAPACK, args...; kwargs...) = YALAPACK.$f!(args...; kwargs...)
end

supports_eig(::Driver, ::Symbol) = false
supports_eig(::LAPACK, f::Symbol) = f in (:simple, :expert)

for (f, f_lapack!, Alg) in (
        (:simple, :geev!, :Simple),
        (:expert, :geevx!, :Expert),
    )
    f_eig_full! = Symbol(f, :_eig_full!)
    f_eig_vals! = Symbol(f, :_eig_vals!)

    # MatrixAlgebraKit wrappers
    @eval begin
        function eig_full!(A::AbstractMatrix, DV, alg::$Alg)
            check_input(eig_full!, A, DV, alg)
            D, V = DV
            Dd, V = $f_eig_full!(A, D.diag, V; alg.kwargs...)
            return D, V
        end
        function eig_vals!(A::AbstractMatrix, D, alg::$Alg)
            check_input(eig_vals!, A, D, alg)
            V = similar(A, complex(eltype(A)), (size(A, 1), 0))
            $f_eig_vals!(A, D, V; alg.kwargs...)
            return D
        end
    end

    # driver dispatch
    @eval begin
        @inline $f_eig_full!(A, Dd, V; driver::Driver = DefaultDriver(), kwargs...) =
            $f_eig_full!(driver, A, Dd, V; kwargs...)
        @inline $f_eig_vals!(A, D, V; driver::Driver = DefaultDriver(), kwargs...) =
            $f_eig_vals!(driver, A, D, V; kwargs...)

        @inline $f_eig_full!(::DefaultDriver, A, Dd, V; kwargs...) =
            $f_eig_full!(default_driver($Alg, A), A, Dd, V; kwargs...)
        @inline $f_eig_vals!(::DefaultDriver, A, D, V; kwargs...) =
            $f_eig_vals!(default_driver($Alg, A), A, D, V; kwargs...)
    end

    # Implementation
    @eval begin
        function $f_eig_full!(driver::Driver, A, Dd, V; fixgauge::Bool = default_fixgauge(), kwargs...)
            supports_eig(driver, $(QuoteNode(f))) ||
                throw(ArgumentError(LazyString("driver ", driver, " does not provide `$($(QuoteNode(f_lapack!)))`")))
            $(
                if f == :simple
                    :(isempty(kwargs) || throw(ArgumentError(LazyString("invalid keyword arguments for ", driver, " simple eig"))))
                else
                    :nothing
                end
            )
            $f_lapack!(driver, A, Dd, V; kwargs...)
            fixgauge && gaugefix!(eig_full!, V)
            return Dd, V
        end
        function $f_eig_vals!(driver::Driver, A, D, V; fixgauge::Bool = default_fixgauge(), kwargs...)
            supports_eig(driver, $(QuoteNode(f))) ||
                throw(ArgumentError(LazyString("driver ", driver, " does not provide `$($(QuoteNode(f_lapack!)))`")))
            $(
                if f == :simple
                    :(isempty(kwargs) || throw(ArgumentError(LazyString("invalid keyword arguments for ", driver, " simple eig"))))
                else
                    :nothing
                end
            )
            $f_lapack!(driver, A, D, V; kwargs...)
            return D
        end
    end
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
for algtype in (:Simple, :Expert)
    lapack_algtype = Symbol(:LAPACK_, algtype)
    @eval begin
        Base.@deprecate(
            eig_full!(A, DV, alg::$lapack_algtype),
            eig_full!(A, DV, $algtype(; driver = LAPACK(), alg.kwargs...))
        )
        Base.@deprecate(
            eig_vals!(A, D, alg::$lapack_algtype),
            eig_vals!(A, D, $algtype(; driver = LAPACK(), alg.kwargs...))
        )
    end
end
Base.@deprecate(
    eig_full!(A, DV, alg::CUSOLVER_Simple),
    eig_full!(A, DV, Simple(; driver = CUSOLVER(), alg.kwargs...))
)
Base.@deprecate(
    eig_vals!(A, D, alg::CUSOLVER_Simple),
    eig_vals!(A, D, Simple(; driver = CUSOLVER(), alg.kwargs...))
)
