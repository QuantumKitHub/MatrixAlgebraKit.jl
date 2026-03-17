# Input
# ------
copy_input(::typeof(svd_full), A::AbstractMatrix) = copy!(similar(A, float(eltype(A))), A)
copy_input(::typeof(svd_compact), A) = copy_input(svd_full, A)
copy_input(::typeof(svd_vals), A) = copy_input(svd_full, A)
copy_input(::Union{typeof(svd_trunc), typeof(svd_trunc_no_error)}, A) = copy_input(svd_compact, A)

copy_input(::typeof(svd_full), A::Diagonal) = copy(A)

# TODO: many of these checks are happening again in the LAPACK routines
function check_input(::typeof(svd_full!), A::AbstractMatrix, USVᴴ, ::AbstractAlgorithm)
    m, n = size(A)
    U, S, Vᴴ = USVᴴ
    @assert U isa AbstractMatrix && S isa AbstractMatrix && Vᴴ isa AbstractMatrix
    @check_size(U, (m, m))
    @check_scalar(U, A)
    @check_size(S, (m, n))
    @check_scalar(S, A, real)
    @check_size(Vᴴ, (n, n))
    @check_scalar(Vᴴ, A)
    return nothing
end
function check_input(::typeof(svd_compact!), A::AbstractMatrix, USVᴴ, ::AbstractAlgorithm)
    m, n = size(A)
    minmn = min(m, n)
    U, S, Vᴴ = USVᴴ
    @assert U isa AbstractMatrix && S isa Diagonal && Vᴴ isa AbstractMatrix
    @check_size(U, (m, minmn))
    @check_scalar(U, A)
    @check_size(S, (minmn, minmn))
    @check_scalar(S, A, real)
    @check_size(Vᴴ, (minmn, n))
    @check_scalar(Vᴴ, A)
    return nothing
end
function check_input(::typeof(svd_vals!), A::AbstractMatrix, S, ::AbstractAlgorithm)
    m, n = size(A)
    minmn = min(m, n)
    @assert S isa AbstractVector
    @check_size(S, (minmn,))
    @check_scalar(S, A, real)
    return nothing
end

function check_input(::typeof(svd_full!), A::AbstractMatrix, USVᴴ, ::DiagonalAlgorithm)
    m, n = size(A)
    @assert m == n && isdiag(A)
    U, S, Vᴴ = USVᴴ
    @assert U isa AbstractMatrix && S isa Diagonal && Vᴴ isa AbstractMatrix
    @check_size(U, (m, m))
    @check_scalar(U, A)
    @check_size(S, (m, n))
    @check_scalar(S, A, real)
    @check_size(Vᴴ, (n, n))
    @check_scalar(Vᴴ, A)
    return nothing
end
function check_input(
        ::typeof(svd_compact!), A::AbstractMatrix, USVᴴ, alg::DiagonalAlgorithm
    )
    return check_input(svd_full!, A, USVᴴ, alg)
end
function check_input(::typeof(svd_vals!), A::AbstractMatrix, S, ::DiagonalAlgorithm)
    m, n = size(A)
    @assert m == n && isdiag(A)
    @assert S isa AbstractVector
    @check_size(S, (m,))
    @check_scalar(S, A, real)
    return nothing
end

# Outputs
# -------
function initialize_output(::typeof(svd_full!), A::AbstractMatrix, ::AbstractAlgorithm)
    m, n = size(A)
    U = similar(A, (m, m))
    S = similar(A, real(eltype(A)), (m, n)) # TODO: Rectangular diagonal type?
    Vᴴ = similar(A, (n, n))
    return (U, S, Vᴴ)
end
function initialize_output(::typeof(svd_compact!), A::AbstractMatrix, ::AbstractAlgorithm)
    m, n = size(A)
    minmn = min(m, n)
    U = similar(A, (m, minmn))
    S = Diagonal(similar(A, real(eltype(A)), (minmn,)))
    Vᴴ = similar(A, (minmn, n))
    return (U, S, Vᴴ)
end
function initialize_output(::typeof(svd_vals!), A::AbstractMatrix, ::AbstractAlgorithm)
    return similar(A, real(eltype(A)), (min(size(A)...),))
end
function initialize_output(::Union{typeof(svd_trunc!), typeof(svd_trunc_no_error!)}, A, alg::TruncatedAlgorithm)
    return initialize_output(svd_compact!, A, alg.alg)
end

function initialize_output(::typeof(svd_full!), A::Diagonal, ::DiagonalAlgorithm)
    TA = eltype(A)
    TUV = Base.promote_op(sign_safe, TA)
    return similar(A, TUV, size(A)), similar(A, real(TA)), similar(A, TUV, size(A))
end
function initialize_output(::typeof(svd_compact!), A::Diagonal, alg::DiagonalAlgorithm)
    return initialize_output(svd_full!, A, alg)
end
function initialize_output(::typeof(svd_vals!), A::Diagonal, ::DiagonalAlgorithm)
    return eltype(A) <: Real ? diagview(A) : similar(A, real(eltype(A)), size(A, 1))
end

# ==========================
#      IMPLEMENTATIONS
# ==========================

for f! in (:gesdd!, :gesvd!, :gesvdj!, :gesvdp!, :gesvdx!, :gesvdr!, :gesdvd!)
    @eval $f!(driver::Driver, args...) = throw(ArgumentError("$driver does not provide $f!"))
end

"""
    svd_via_adjoint!(f!, driver, A, S, U, Vᴴ; kwargs...)

Compute the SVD of `A` (m × n, m < n) by computing the SVD of `adjoint(A)` using
the provided function `f!(driver, A, S, U, Vᴴ; kwargs...)`. Use this as a building
block for drivers whose SVD routines require m ≥ n.
"""
function svd_via_adjoint!(f!::F, driver::Driver, A, S, U, Vᴴ; kwargs...) where {F}
    Aᴴ = adjoint!(similar(A'), A)
    Uᴴ = similar(U')
    V = similar(Vᴴ')
    f!(driver, Aᴴ, S, V, Uᴴ; kwargs...)
    length(U) > 0 && adjoint!(U, Uᴴ)
    length(Vᴴ) > 0 && adjoint!(Vᴴ, V)
    return S, U, Vᴴ
end

# LAPACK
for f! in (:gesdd!, :gesvd!, :gesvdx!, :gesdvd!)
    @eval $f!(::LAPACK, args...; kwargs...) = YALAPACK.$f!(args...; kwargs...)
end

function gesvdj!(::LAPACK, A, S, U, Vᴴ; kwargs...)
    m, n = size(A)
    m >= n && return YALAPACK.gesvdj!(A, S, U, Vᴴ)
    return svd_via_adjoint!(gesvdj!, LAPACK(), A, S, U, Vᴴ; kwargs...)
end

for (f, f_lapack!, Alg) in (
        (:safe_divide_and_conquer, :gesdvd!, :SafeDivideAndConquer),
        (:divide_and_conquer, :gesdd!, :DivideAndConquer),
        (:qr_iteration, :gesvd!, :QRIteration),
        (:bisection, :gesvdx!, :Bisection),
        (:jacobi, :gesvdj!, :Jacobi),
        (:svd_polar, :gesvdp!, :SVDViaPolar),
    )
    f_svd! = Symbol(f, :_svd!)
    f_svd_full! = Symbol(f, :_svd_full!)
    f_svd_vals! = Symbol(f, :_svd_vals!)

    # MatrixAlgebraKit wrappers
    @eval begin
        function svd_compact!(A, USVᴴ, alg::$Alg)
            check_input(svd_compact!, A, USVᴴ, alg)
            return $f_svd!(A, USVᴴ...; alg.kwargs...)
        end
        function svd_full!(A, USVᴴ, alg::$Alg)
            check_input(svd_full!, A, USVᴴ, alg)
            return $f_svd_full!(A, USVᴴ...; alg.kwargs...)
        end
        function svd_vals!(A, S, alg::$Alg)
            check_input(svd_vals!, A, S, alg)
            return $f_svd_vals!(A, S; alg.kwargs...)
        end
    end

    # driver
    @eval begin
        @inline $f_svd!(A, U, S, Vᴴ; driver::Driver = DefaultDriver(), kwargs...) = $f_svd!(driver, A, U, S, Vᴴ; kwargs...)
        @inline $f_svd_full!(A, U, S, Vᴴ; driver::Driver = DefaultDriver(), kwargs...) = $f_svd_full!(driver, A, U, S, Vᴴ; kwargs...)
        @inline $f_svd_vals!(A, S; driver::Driver = DefaultDriver(), kwargs...) = $f_svd_vals!(driver, A, S; kwargs...)

        @inline $f_svd!(::DefaultDriver, A, U, S, Vᴴ; kwargs...) = $f_svd!(default_driver($Alg, A), A, U, S, Vᴴ; kwargs...)
        @inline $f_svd_full!(::DefaultDriver, A, U, S, Vᴴ; kwargs...) = $f_svd_full!(default_driver($Alg, A), A, U, S, Vᴴ; kwargs...)
        @inline $f_svd_vals!(::DefaultDriver, A, S; kwargs...) = $f_svd_vals!(default_driver($Alg, A), A, S; kwargs...)
    end

    # Implementation
    @eval begin
        function $f_svd!(driver::Driver, A, U, S, Vᴴ; fixgauge::Bool = true, kwargs...)
            supports_svd(driver, $(QuoteNode(f))) ||
                throw(ArgumentError(LazyString("driver ", driver, " does not provide `$($(QuoteNode(f_lapack!)))`")))
            isempty(A) && return one!(U), zero!(S), one!(Vᴴ)
            $f_lapack!(driver, A, diagview(S), U, Vᴴ; kwargs...)
            fixgauge && gaugefix!(svd_compact!, U, Vᴴ)
            return U, S, Vᴴ
        end
        function $f_svd_full!(driver::Driver, A, U, S, Vᴴ; fixgauge::Bool = true, kwargs...)
            supports_svd_full(driver, $(QuoteNode(f))) ||
                throw(ArgumentError(LazyString("driver ", driver, " does not provide `$($(QuoteNode(f_lapack!)))`")))
            isempty(A) && return one!(U), zero!(S), one!(Vᴴ)
            zero!(S)
            minmn = min(size(A)...)
            $f_lapack!(driver, A, view(S, 1:minmn, 1), U, Vᴴ; kwargs...)
            diagview(S) .= view(S, 1:minmn, 1)
            zero!(view(S, 2:minmn, 1))
            fixgauge && gaugefix!(svd_full!, U, Vᴴ)
            return U, S, Vᴴ
        end
        function $f_svd_vals!(driver::Driver, A, S; fixgauge::Bool = true, kwargs...)
            supports_svd(driver, $(QuoteNode(f))) ||
                throw(ArgumentError(LazyString("driver ", driver, " does not provide `$($(QuoteNode(f_lapack!)))`")))
            isempty(A) && return zero!(S)
            U, Vᴴ = similar(A, (0, 0)), similar(A, (0, 0))
            $f_lapack!(driver, A, S, U, Vᴴ; kwargs...)
            return S
        end
    end
end

supports_svd(::Driver, ::Symbol) = false
supports_svd(::LAPACK, f::Symbol) = f in (:safe_divide_and_conquer, :divide_and_conquer, :qr_iteration, :bisection, :jacobi)
supports_svd(::GLA, f::Symbol) = f === :qr_iteration
supports_svd(::CUSOLVER, f::Symbol) = f in (:qr_iteration, :jacobi, :svd_polar)
supports_svd(::ROCSOLVER, f::Symbol) = f in (:qr_iteration, :jacobi)
supports_svd_full(::Driver, ::Symbol) = false
supports_svd_full(::LAPACK, f::Symbol) = f in (:safe_divide_and_conquer, :divide_and_conquer, :qr_iteration)
supports_svd_full(::GLA, f::Symbol) = f === :qr_iteration
supports_svd_full(::CUSOLVER, f::Symbol) = f === :qr_iteration
supports_svd_full(::ROCSOLVER, f::Symbol) = f === :qr_iteration

function svd_trunc_no_error!(A, USVᴴ, alg::TruncatedAlgorithm)
    U, S, Vᴴ = svd_compact!(A, USVᴴ, alg.alg)
    USVᴴtrunc, ind = truncate(svd_trunc!, (U, S, Vᴴ), alg.trunc)
    return USVᴴtrunc
end

function svd_trunc!(A, USVᴴ, alg::TruncatedAlgorithm)
    U, S, Vᴴ = svd_compact!(A, USVᴴ, alg.alg)
    USVᴴtrunc, ind = truncate(svd_trunc!, (U, S, Vᴴ), alg.trunc)
    ϵ = truncation_error!(diagview(S), ind)
    return USVᴴtrunc..., ϵ
end

# Diagonal logic
# --------------
function svd_full!(A::AbstractMatrix, USVᴴ, alg::DiagonalAlgorithm)
    check_input(svd_full!, A, USVᴴ, alg)
    Ad = diagview(A)
    U, S, Vᴴ = USVᴴ
    if isempty(Ad)
        one!(U)
        one!(Vᴴ)
        return USVᴴ
    end
    p = sortperm(Ad; by = abs, rev = true)
    zero!(U)
    zero!(Vᴴ)
    n = size(A, 1)

    pV = (1:n) .+ (p .- 1) .* n
    Vᴴ[pV] .= sign_safe.(view(Ad, p))

    Sd = diagview(S)
    if Ad === Sd
        @. Sd = abs(Ad)
        permute!(Sd, p)
    else
        Sd .= abs.(view(Ad, p))
    end

    p .+= (0:(n - 1)) .* n
    U[p] .= Ref(one(eltype(U)))

    return U, S, Vᴴ
end
function svd_compact!(A, USVᴴ, alg::DiagonalAlgorithm)
    return svd_full!(A, USVᴴ, alg)
end
function svd_vals!(A::AbstractMatrix, S, alg::DiagonalAlgorithm)
    check_input(svd_vals!, A, S, alg)
    m, n = size(A)
    minmn = min(m, n)
    if minmn == 0
        zero!(S)
        return S
    end
    Ad = diagview(A)
    S .= abs.(Ad)
    sort!(S; rev = true)
    return S
end

# GPU logic (randomized SVD - CUSOLVER_Randomized has no CPU analog, kept as-is)
# ---------------------------------------------------------------------------------

function check_input(
        ::Union{typeof(svd_trunc!), typeof(svd_trunc_no_error!)}, A::AbstractMatrix, USVᴴ, alg::CUSOLVER_Randomized
    )
    m, n = size(A)
    minmn = min(m, n)
    U, S, Vᴴ = USVᴴ
    @assert U isa AbstractMatrix && S isa Diagonal && Vᴴ isa AbstractMatrix
    @check_size(U, (m, m))
    @check_scalar(U, A)
    @check_size(S, (minmn, minmn))
    @check_scalar(S, A, real)
    @check_size(Vᴴ, (n, n))
    @check_scalar(Vᴴ, A)
    return nothing
end

function initialize_output(
        ::Union{typeof(svd_trunc!), typeof(svd_trunc_no_error!)}, A::AbstractMatrix, alg::TruncatedAlgorithm{<:CUSOLVER_Randomized}
    )
    m, n = size(A)
    minmn = min(m, n)
    U = similar(A, (m, m))
    S = Diagonal(similar(A, real(eltype(A)), (minmn,)))
    Vᴴ = similar(A, (n, n))
    return (U, S, Vᴴ)
end

function _gpu_Xgesvdr!(
        A::AbstractMatrix, S::AbstractVector, U::AbstractMatrix, Vᴴ::AbstractMatrix; kwargs...
    )
    throw(MethodError(_gpu_Xgesvdr!, (A, S, U, Vᴴ)))
end

function svd_trunc_no_error!(A::AbstractMatrix, USVᴴ, alg::TruncatedAlgorithm{<:GPU_Randomized})
    U, S, Vᴴ = USVᴴ
    check_input(svd_trunc_no_error!, A, (U, S, Vᴴ), alg.alg)
    _gpu_Xgesvdr!(A, diagview(S), U, Vᴴ; alg.alg.kwargs...)

    # TODO: make sure that truncation is based on maxrank, otherwise this might be wrong
    (Utr, Str, Vᴴtr), _ = truncate(svd_trunc!, (U, S, Vᴴ), alg.trunc)

    do_gauge_fix = get(alg.alg.kwargs, :fixgauge, default_fixgauge())::Bool
    # the output matrices here are the same size as for svd_full!
    do_gauge_fix && gaugefix!(svd_trunc!, Utr, Vᴴtr)

    return Utr, Str, Vᴴtr
end

function svd_trunc!(A::AbstractMatrix, USVᴴ, alg::TruncatedAlgorithm{<:GPU_Randomized})
    Utr, Str, Vᴴtr = svd_trunc_no_error!(A, USVᴴ, alg)
    # normal `truncation_error!` does not work here since `S` is not the full singular value spectrum
    normS = norm(diagview(Str))
    normA = norm(A)
    # equivalent to sqrt(normA^2 - normS^2)
    # but may be more accurate
    ϵ = sqrt((normA + normS) * abs(normA - normS))
    return Utr, Str, Vᴴtr, ϵ
end

# Deprecations
# ------------
for algtype in (:SafeDivideAndConquer, :DivideAndConquer, :QRIteration, :Jacobi, :Bisection)
    lapack_algtype = Symbol(:LAPACK_, algtype)
    @eval begin
        Base.@deprecate(
            svd_compact!(A, USVᴴ, alg::$lapack_algtype),
            svd_compact!(A, USVᴴ, $algtype(; driver = LAPACK(), alg.kwargs...))
        )
        Base.@deprecate(
            svd_full!(A, USVᴴ, alg::$lapack_algtype),
            svd_full!(A, USVᴴ, $algtype(; driver = LAPACK(), alg.kwargs...))
        )
        Base.@deprecate(
            svd_vals!(A, S, alg::$lapack_algtype),
            svd_vals!(A, S, $algtype(; driver = LAPACK(), alg.kwargs...))
        )
    end
end

for (algtype, newtype, drivertype) in (
        (:CUSOLVER_QRIteration, :QRIteration, :CUSOLVER),
        (:CUSOLVER_Jacobi, :Jacobi, :CUSOLVER),
        (:CUSOLVER_SVDPolar, :SVDViaPolar, :CUSOLVER),
        (:ROCSOLVER_QRIteration, :QRIteration, :ROCSOLVER),
        (:ROCSOLVER_Jacobi, :Jacobi, :ROCSOLVER),
    )
    @eval begin
        Base.@deprecate(
            svd_compact!(A, USVᴴ, alg::$algtype),
            svd_compact!(A, USVᴴ, $newtype(; driver = $drivertype(), alg.kwargs...))
        )
        Base.@deprecate(
            svd_full!(A, USVᴴ, alg::$algtype),
            svd_full!(A, USVᴴ, $newtype(; driver = $drivertype(), alg.kwargs...))
        )
        Base.@deprecate(
            svd_vals!(A, S, alg::$algtype),
            svd_vals!(A, S, $newtype(; driver = $drivertype(), alg.kwargs...))
        )
    end
end

# GLA_QRIteration SVD deprecations (eigh methods remain in the GLA extension)
Base.@deprecate(
    svd_compact!(A, USVᴴ, alg::GLA_QRIteration),
    svd_compact!(A, USVᴴ, QRIteration(; driver = GLA(), alg.kwargs...))
)
Base.@deprecate(
    svd_full!(A, USVᴴ, alg::GLA_QRIteration),
    svd_full!(A, USVᴴ, QRIteration(; driver = GLA(), alg.kwargs...))
)
Base.@deprecate(
    svd_vals!(A, S, alg::GLA_QRIteration),
    svd_vals!(A, S, QRIteration(; driver = GLA(), alg.kwargs...))
)
