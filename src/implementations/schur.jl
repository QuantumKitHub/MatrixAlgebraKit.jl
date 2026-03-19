# Inputs
# ------
copy_input(::typeof(schur_full), A) = copy_input(eig_full, A)
copy_input(::typeof(schur_vals), A) = copy_input(eig_vals, A)

# check input
function check_input(::typeof(schur_full!), A::AbstractMatrix, TZv, ::AbstractAlgorithm)
    m = LinearAlgebra.checksquare(A)
    T, Z, vals = TZv
    @assert T isa AbstractMatrix && Z isa AbstractMatrix && vals isa AbstractVector
    @check_size(T, (m, m))
    @check_scalar(T, A)
    @check_size(Z, (m, m))
    @check_scalar(Z, A)
    @check_size(vals, (m,))
    @check_scalar(vals, A, complex)
    return nothing
end
function check_input(::typeof(schur_vals!), A::AbstractMatrix, vals, ::AbstractAlgorithm)
    m = LinearAlgebra.checksquare(A)
    @assert vals isa AbstractVector
    @check_size(vals, (m,))
    @check_scalar(vals, A, complex)
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

# DefaultAlgorithm intercepts
# ---------------------------
for f! in (:schur_full!, :schur_vals!)
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

for f! in (:gees!, :geesx!)
    @eval $f!(driver::Driver, args...) = throw(ArgumentError("$driver does not provide $f!"))
end

# LAPACK implementations
for f! in (:gees!, :geesx!)
    @eval $f!(::LAPACK, args...; kwargs...) = YALAPACK.$f!(args...; kwargs...)
end

supports_schur(::Driver, ::Symbol) = false
supports_schur(::LAPACK, f::Symbol) = f in (:simple, :expert)

for (f, f_lapack!, Alg) in (
        (:simple, :gees!, :Simple),
        (:expert, :geesx!, :Expert),
    )
    f_schur_full! = Symbol(f, :_schur_full!)
    f_schur_vals! = Symbol(f, :_schur_vals!)

    # MatrixAlgebraKit wrappers
    @eval begin
        function schur_full!(A::AbstractMatrix, TZv, alg::$Alg)
            check_input(schur_full!, A, TZv, alg)
            T, Z, vals = TZv
            $f_schur_full!(A, T, Z, vals; alg.kwargs...)
            return T, Z, vals
        end
        function schur_vals!(A::AbstractMatrix, vals, alg::$Alg)
            check_input(schur_vals!, A, vals, alg)
            Z = similar(A, eltype(A), (size(A, 1), 0))
            $f_schur_vals!(A, Z, vals; alg.kwargs...)
            return vals
        end
    end

    # driver dispatch
    @eval begin
        @inline $f_schur_full!(A, T, Z, vals; driver::Driver = DefaultDriver(), kwargs...) =
            $f_schur_full!(driver, A, T, Z, vals; kwargs...)
        @inline $f_schur_vals!(A, Z, vals; driver::Driver = DefaultDriver(), kwargs...) =
            $f_schur_vals!(driver, A, Z, vals; kwargs...)

        @inline $f_schur_full!(::DefaultDriver, A, T, Z, vals; kwargs...) =
            $f_schur_full!(default_driver($Alg, A), A, T, Z, vals; kwargs...)
        @inline $f_schur_vals!(::DefaultDriver, A, Z, vals; kwargs...) =
            $f_schur_vals!(default_driver($Alg, A), A, Z, vals; kwargs...)
    end

    # Implementation
    @eval begin
        function $f_schur_full!(driver::Driver, A, T, Z, vals; kwargs...)
            supports_schur(driver, $(QuoteNode(f))) ||
                throw(ArgumentError(LazyString("driver ", driver, " does not provide `$($(QuoteNode(f_lapack!)))`")))
            isempty(kwargs) ||
                throw(ArgumentError(LazyString("invalid keyword arguments for ", driver, " schur")))
            $f_lapack!(driver, A, Z, vals)
            T === A || copy!(T, A)
            return T, Z, vals
        end
        function $f_schur_vals!(driver::Driver, A, Z, vals; kwargs...)
            supports_schur(driver, $(QuoteNode(f))) ||
                throw(ArgumentError(LazyString("driver ", driver, " does not provide `$($(QuoteNode(f_lapack!)))`")))
            isempty(kwargs) ||
                throw(ArgumentError(LazyString("invalid keyword arguments for ", driver, " schur")))
            $f_lapack!(driver, A, Z, vals)
            return vals
        end
    end
end

# Deprecations
# ------------
for algtype in (:Simple, :Expert)
    lapack_algtype = Symbol(:LAPACK_, algtype)
    @eval begin
        Base.@deprecate(
            schur_full!(A, TZv, alg::$lapack_algtype),
            schur_full!(A, TZv, $algtype(; driver = LAPACK(), alg.kwargs...))
        )
        Base.@deprecate(
            schur_vals!(A, vals, alg::$lapack_algtype),
            schur_vals!(A, vals, $algtype(; driver = LAPACK(), alg.kwargs...))
        )
    end
end
