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

# DefaultAlgorithm intercepts
# ---------------------------
for f! in (:svd_full!, :svd_compact!, :svd_vals!, :svd_trunc!, :svd_trunc_no_error!)
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

for f! in (:gesdd!, :gesvd!, :gesvdj!, :gesvdp!, :gesvdx!, :gesvdr!, :gesdvd!)
    @eval $f!(driver::Driver, args...) = throw(ArgumentError("$driver does not provide $($(f!))"))
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
    svd_compact_f! = Symbol(:svd_compact_, f, :!)
    svd_full_f! = Symbol(:svd_full_, f, :!)
    svd_vals_f! = Symbol(:svd_vals_, f, :!)

    # MatrixAlgebraKit wrappers
    @eval begin
        function svd_compact!(A::AbstractMatrix, USVᴴ, alg::$Alg)
            check_input(svd_compact!, A, USVᴴ, alg)
            return $svd_compact_f!(A, USVᴴ...; alg.kwargs...)
        end
        function svd_full!(A::AbstractMatrix, USVᴴ, alg::$Alg)
            check_input(svd_full!, A, USVᴴ, alg)
            return $svd_full_f!(A, USVᴴ...; alg.kwargs...)
        end
        function svd_vals!(A::AbstractMatrix, S, alg::$Alg)
            check_input(svd_vals!, A, S, alg)
            return $svd_vals_f!(A, S; alg.kwargs...)
        end
    end

    # driver
    @eval begin
        @inline $svd_compact_f!(A, U, S, Vᴴ; driver::Driver = DefaultDriver(), kwargs...) = $svd_compact_f!(driver, A, U, S, Vᴴ; kwargs...)
        @inline $svd_full_f!(A, U, S, Vᴴ; driver::Driver = DefaultDriver(), kwargs...) = $svd_full_f!(driver, A, U, S, Vᴴ; kwargs...)
        @inline $svd_vals_f!(A, S; driver::Driver = DefaultDriver(), kwargs...) = $svd_vals_f!(driver, A, S; kwargs...)

        @inline $svd_compact_f!(::DefaultDriver, A, U, S, Vᴴ; kwargs...) = $svd_compact_f!(default_driver($Alg, A), A, U, S, Vᴴ; kwargs...)
        @inline $svd_full_f!(::DefaultDriver, A, U, S, Vᴴ; kwargs...) = $svd_full_f!(default_driver($Alg, A), A, U, S, Vᴴ; kwargs...)
        @inline $svd_vals_f!(::DefaultDriver, A, S; kwargs...) = $svd_vals_f!(default_driver($Alg, A), A, S; kwargs...)
    end

    # Implementation
    @eval begin
        function $svd_compact_f!(driver::Driver, A, U, S, Vᴴ; fixgauge::Bool = true, kwargs...)
            isempty(A) && return one!(U), zero!(S), one!(Vᴴ)
            $f_lapack!(driver, A, diagview(S), U, Vᴴ; kwargs...)
            fixgauge && gaugefix!(svd_compact!, U, Vᴴ)
            return U, S, Vᴴ
        end
        function $svd_full_f!(driver::Driver, A, U, S, Vᴴ; fixgauge::Bool = true, kwargs...)
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
        function $svd_vals_f!(driver::Driver, A, S; fixgauge::Bool = true, kwargs...)
            isempty(A) && return zero!(S)
            U, Vᴴ = similar(A, (0, 0)), similar(A, (0, 0))
            $f_lapack!(driver, A, S, U, Vᴴ; kwargs...)
            return S
        end
    end
end

supports_svd_full(::Driver, ::Symbol) = false
supports_svd_full(::LAPACK, f::Symbol) = f in (:safe_divide_and_conquer, :divide_and_conquer, :qr_iteration)

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

# Sketched Logic
# --------------
function initialize_output(::typeof(svd_trunc_no_error!), A::AbstractMatrix, alg::SketchedAlgorithm)
    U, Vᴴ = initialize_output(left_sketch!, A, alg.sketch)
    S = Diagonal(similar(U, real(eltype(U)), (size(U, 2),)))
    return U, S, Vᴴ
end
initialize_output(::typeof(svd_trunc!), A::AbstractMatrix, alg::SketchedAlgorithm) =
    initialize_output(svd_trunc_no_error!, A, alg)

function check_input(::typeof(svd_trunc_no_error!), A::AbstractMatrix, (U, S, Vᴴ), alg::SketchedAlgorithm)
    check_input(left_sketch!, A, (U, Vᴴ), alg.sketch)
    @assert U isa AbstractMatrix && S isa Diagonal && Vᴴ isa AbstractMatrix
    k = size(U, 2)
    @check_size(S, (k, k))
    @check_scalar(S, U, real)
    return nothing
end
check_input(::typeof(svd_trunc!), A::AbstractMatrix, USVᴴ, alg::SketchedAlgorithm) =
    check_input(svd_trunc_no_error!, A, USVᴴ, alg)

function svd_trunc_no_error!(A::AbstractMatrix, (U, S, Vᴴ), alg::SketchedAlgorithm)
    check_input(svd_trunc_no_error!, A, (U, S, Vᴴ), alg)
    return gesvdr!(alg.driver, A, S, U, Vᴴ; alg.sketch, alg.alg, alg.trunc)
end

# CUSOLVER's gesvdr kernel requires full U and Vᴴ
function initialize_output(
        ::typeof(svd_trunc_no_error!), A::AbstractMatrix,
        alg::SketchedAlgorithm{<:AbstractAlgorithm, <:SketchingStrategy, <:TruncationStrategy, CUSOLVER},
    )
    m, n = size(A)
    minmn = min(m, n)
    T = float(eltype(A))
    U = similar(A, T, (m, m))
    S = Diagonal(similar(A, real(T), (minmn,)))
    Vᴴ = similar(A, T, (n, n))
    return (U, S, Vᴴ)
end

function check_input(
        ::typeof(svd_trunc_no_error!), A::AbstractMatrix, (U, S, Vᴴ),
        alg::SketchedAlgorithm{<:AbstractAlgorithm, <:SketchingStrategy, <:TruncationStrategy, CUSOLVER},
    )
    m, n = size(A)
    minmn = min(m, n)
    @assert U isa AbstractMatrix && S isa Diagonal && Vᴴ isa AbstractMatrix
    @check_size(U, (m, m))
    @check_scalar(U, A)
    @check_size(S, (minmn, minmn))
    @check_scalar(S, A, real)
    @check_size(Vᴴ, (n, n))
    @check_scalar(Vᴴ, A)
    return nothing
end

function svd_trunc!(A::AbstractMatrix, USVᴴ, alg::SketchedAlgorithm)
    U, S, Vᴴ = svd_trunc_no_error!(A, USVᴴ, alg)
    Na = norm(A)
    Ns = norm(S)
    return U, S, Vᴴ, sqrt(max(zero(Na), (Na + Ns) * (Na - Ns)))
end

# gesvdr! drivers
# ---------------
default_driver(::Type{<:SketchedAlgorithm}, ::Type{<:AbstractArray}) = Native()

gesvdr!(::DefaultDriver, A, S, U, Vᴴ; kwargs...) =
    gesvdr!(default_driver(SketchedAlgorithm, A), A, S, U, Vᴴ; kwargs...)

function gesvdr!(
        ::Native, A::AbstractMatrix, S, U, Vᴴ;
        sketch::SketchingStrategy, alg::AbstractAlgorithm,
        trunc::TruncationStrategy
    )
    m, n = size(A)
    if m ≥ n
        Q, B = left_sketch!(A, (U, Vᴴ), sketch)
        k = size(B, 1)
        U′ = similar(B, (k, k))
        Vᴴ′ = similar(B)
        Uout′, Sout, Vᴴout, _ = svd_trunc!(B, (U′, S, Vᴴ′), TruncatedAlgorithm(alg, trunc))
        Uout = Q * Uout′
    else
        B, Pᴴ = right_sketch!(A, (U, Vᴴ), sketch)
        k = size(B, 2)
        U′ = similar(B)
        Vᴴ′ = similar(B, (k, k))
        Uout, Sout, Vᴴout′, _ = svd_trunc!(B, (U′, S, Vᴴ′), TruncatedAlgorithm(alg, trunc))
        Vᴴout = Vᴴout′ * Pᴴ
    end
    return Uout, Sout, Vᴴout
end

# Deprecations
# ------------
for algtype in (:SafeDivideAndConquer, :DivideAndConquer, :QRIteration, :Jacobi, :Bisection)
    lapack_algtype = Symbol(:LAPACK_, algtype)
    @eval begin
        Base.@deprecate(
            svd_compact!(A::AbstractMatrix, USVᴴ, alg::$lapack_algtype),
            svd_compact!(A, USVᴴ, $algtype(; driver = LAPACK(), alg.kwargs...))
        )
        Base.@deprecate(
            svd_full!(A::AbstractMatrix, USVᴴ, alg::$lapack_algtype),
            svd_full!(A, USVᴴ, $algtype(; driver = LAPACK(), alg.kwargs...))
        )
        Base.@deprecate(
            svd_vals!(A::AbstractMatrix, S, alg::$lapack_algtype),
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
            svd_compact!(A::AbstractMatrix, USVᴴ, alg::$algtype),
            svd_compact!(A, USVᴴ, $newtype(; driver = $drivertype(), alg.kwargs...))
        )
        Base.@deprecate(
            svd_full!(A::AbstractMatrix, USVᴴ, alg::$algtype),
            svd_full!(A, USVᴴ, $newtype(; driver = $drivertype(), alg.kwargs...))
        )
        Base.@deprecate(
            svd_vals!(A::AbstractMatrix, S, alg::$algtype),
            svd_vals!(A, S, $newtype(; driver = $drivertype(), alg.kwargs...))
        )
    end
end

# CUSOLVER_Randomized → SketchedAlgorithm with driver = CUSOLVER()
function _cusolver_randomized_to_sketched(alg::CUSOLVER_Randomized)
    k = alg.kwargs.k
    p = alg.kwargs.p
    niters = alg.kwargs.niters
    return SketchedAlgorithm(
        QRIteration(),
        GaussianSketching(k + p; numiter = niters),
        truncrank(k);
        driver = CUSOLVER(),
    )
end

for f! in (:svd_trunc!, :svd_trunc_no_error!)
    @eval Base.@deprecate(
        $f!(A::AbstractMatrix, USVᴴ, alg::CUSOLVER_Randomized),
        $f!(A, USVᴴ, _cusolver_randomized_to_sketched(alg))
    )
end

@inline function select_algorithm(::typeof(svd_trunc!), A, alg::CUSOLVER_Randomized; kwargs...)
    Base.depwarn(
        "`CUSOLVER_Randomized` is deprecated; use \
         `SketchedAlgorithm(QRIteration(), GaussianSketching(k+p; numiter=niters), truncrank(k); driver=CUSOLVER())` instead.",
        :select_algorithm,
    )
    isempty(kwargs) ||
        throw(ArgumentError("Additional keyword arguments are not allowed when algorithm parameters are specified."))
    return _cusolver_randomized_to_sketched(alg)
end
@inline function select_algorithm(::typeof(svd_trunc_no_error!), A, alg::CUSOLVER_Randomized; kwargs...)
    return select_algorithm(svd_trunc!, A, alg; kwargs...)
end

# GLA_QRIteration SVD deprecations (eigh methods remain in the GLA extension)
Base.@deprecate(
    svd_compact!(A::AbstractMatrix, USVᴴ, alg::GLA_QRIteration),
    svd_compact!(A, USVᴴ, QRIteration(; driver = GLA(), alg.kwargs...))
)
Base.@deprecate(
    svd_full!(A::AbstractMatrix, USVᴴ, alg::GLA_QRIteration),
    svd_full!(A, USVᴴ, QRIteration(; driver = GLA(), alg.kwargs...))
)
Base.@deprecate(
    svd_vals!(A::AbstractMatrix, S, alg::GLA_QRIteration),
    svd_vals!(A, S, QRIteration(; driver = GLA(), alg.kwargs...))
)
