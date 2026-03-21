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

gees!(driver::Driver, args...) = throw(ArgumentError("$driver does not provide `gees!`"))
function geesx!(driver::Driver, A, Dd, V; kwargs...)
    @warn "$driver does not provide `geesx!`, falling back to `gees!`" maxlog = 1
    return gees!(driver, A, Dd, V)
end

# LAPACK implementations
for f! in (:gees!, :geesx!)
    @eval $f!(::LAPACK, args...; kwargs...) = YALAPACK.$f!(args...; kwargs...)
end

# driver dispatch
@inline schur_full_qr_iteration!(A, TZv; driver::Driver = DefaultDriver(), kwargs...) =
    schur_full_qr_iteration!(driver, A, TZv; kwargs...)
@inline schur_vals_qr_iteration!(A, vals; driver::Driver = DefaultDriver(), kwargs...) =
    schur_vals_qr_iteration!(driver, A, vals; kwargs...)

@inline schur_full_qr_iteration!(::DefaultDriver, A, TZv; kwargs...) =
    schur_full_qr_iteration!(default_driver(QRIteration, A), A, TZv; kwargs...)
@inline schur_vals_qr_iteration!(::DefaultDriver, A, vals; kwargs...) =
    schur_vals_qr_iteration!(default_driver(QRIteration, A), A, vals; kwargs...)

# Implementation
function schur_full_qr_iteration!(driver::Driver, A, TZv; expert::Bool = false)
    T, Z, vals = TZv
    expert ? geesx!(driver, A, Z, vals) : gees!(driver, A, Z, vals)
    T === A || copy!(T, A)
    return TZv
end
function schur_vals_qr_iteration!(driver::Driver, A, vals; expert::Bool = false)
    Z = similar(A, eltype(A), (size(A, 1), 0))
    expert ? geesx!(driver, A, Z, vals) : gees!(driver, A, Z, vals)
    return vals
end

# Top-level QRIteration dispatch
function schur_full!(A::AbstractMatrix, TZv, alg::QRIteration)
    check_input(schur_full!, A, TZv, alg)
    schur_full_qr_iteration!(A, TZv; alg.kwargs...)
    return TZv
end
function schur_vals!(A::AbstractMatrix, vals, alg::QRIteration)
    check_input(schur_vals!, A, vals, alg)
    schur_vals_qr_iteration!(A, vals; alg.kwargs...)
    return vals
end

# Deprecations
# ------------
for (lapack_algtype, expert_val) in ((:LAPACK_Simple, false), (:LAPACK_Expert, true))
    @eval begin
        Base.@deprecate(
            schur_full!(A, TZv, alg::$lapack_algtype),
            schur_full!(A, TZv, QRIteration(; expert = $expert_val, alg.kwargs...))
        )
        Base.@deprecate(
            schur_vals!(A, vals, alg::$lapack_algtype),
            schur_vals!(A, vals, QRIteration(; expert = $expert_val, alg.kwargs...))
        )
    end
end
