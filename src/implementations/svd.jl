# Input
# ------
copy_input(::typeof(svd_full), A::AbstractMatrix) = copy!(similar(A, float(eltype(A))), A)
copy_input(::typeof(svd_compact), A) = copy_input(svd_full, A)
copy_input(::typeof(svd_vals), A) = copy_input(svd_full, A)
copy_input(::typeof(svd_trunc), A) = copy_input(svd_compact, A)

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
function initialize_output(::typeof(svd_trunc!), A, alg::TruncatedAlgorithm)
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

# Implementation
# --------------
function svd_full!(A::AbstractMatrix, USVᴴ, alg::LAPACK_SVDAlgorithm)
    check_input(svd_full!, A, USVᴴ, alg)
    U, S, Vᴴ = USVᴴ
    fill!(S, zero(eltype(S)))
    m, n = size(A)
    minmn = min(m, n)
    if minmn == 0
        one!(U)
        zero!(S)
        one!(Vᴴ)
        return USVᴴ
    end

    do_gauge_fix = get(alg.kwargs, :fixgauge, default_fixgauge())::Bool
    alg_kwargs = Base.structdiff(alg.kwargs, NamedTuple{(:fixgauge,)})

    if alg isa LAPACK_QRIteration
        isempty(alg_kwargs) ||
            throw(ArgumentError("invalid keyword arguments for LAPACK_QRIteration"))
        YALAPACK.gesvd!(A, view(S, 1:minmn, 1), U, Vᴴ)
    elseif alg isa LAPACK_DivideAndConquer
        isempty(alg_kwargs) ||
            throw(ArgumentError("invalid keyword arguments for LAPACK_DivideAndConquer"))
        YALAPACK.gesdd!(A, view(S, 1:minmn, 1), U, Vᴴ)
    elseif alg isa LAPACK_Bisection
        throw(ArgumentError("LAPACK_Bisection is not supported for full SVD"))
    elseif alg isa LAPACK_Jacobi
        throw(ArgumentError("LAPACK_Jacobi is not supported for full SVD"))
    else
        throw(ArgumentError("Unsupported SVD algorithm"))
    end

    for i in 2:minmn
        S[i, i] = S[i, 1]
        S[i, 1] = zero(eltype(S))
    end

    do_gauge_fix && gaugefix!(svd_full!, U, Vᴴ)

    return USVᴴ
end

function svd_compact!(A::AbstractMatrix, USVᴴ, alg::LAPACK_SVDAlgorithm)
    check_input(svd_compact!, A, USVᴴ, alg)
    U, S, Vᴴ = USVᴴ

    do_gauge_fix = get(alg.kwargs, :fixgauge, default_fixgauge())::Bool
    alg_kwargs = Base.structdiff(alg.kwargs, NamedTuple{(:fixgauge,)})

    if alg isa LAPACK_QRIteration
        isempty(alg_kwargs) ||
            throw(ArgumentError("invalid keyword arguments for LAPACK_QRIteration"))
        YALAPACK.gesvd!(A, S.diag, U, Vᴴ)
    elseif alg isa LAPACK_DivideAndConquer
        isempty(alg_kwargs) ||
            throw(ArgumentError("invalid keyword arguments for LAPACK_DivideAndConquer"))
        YALAPACK.gesdd!(A, S.diag, U, Vᴴ)
    elseif alg isa LAPACK_Bisection
        YALAPACK.gesvdx!(A, S.diag, U, Vᴴ; alg_kwargs...)
    elseif alg isa LAPACK_Jacobi
        isempty(alg_kwargs) ||
            throw(ArgumentError("invalid keyword arguments for LAPACK_Jacobi"))
        YALAPACK.gesvj!(A, S.diag, U, Vᴴ)
    else
        throw(ArgumentError("Unsupported SVD algorithm"))
    end

    do_gauge_fix && gaugefix!(svd_compact!, U, Vᴴ)

    return USVᴴ
end

function svd_vals!(A::AbstractMatrix, S, alg::LAPACK_SVDAlgorithm)
    check_input(svd_vals!, A, S, alg)
    U, Vᴴ = similar(A, (0, 0)), similar(A, (0, 0))

    alg_kwargs = Base.structdiff(alg.kwargs, NamedTuple{(:fixgauge,)})

    if alg isa LAPACK_QRIteration
        isempty(alg_kwargs) ||
            throw(ArgumentError("invalid keyword arguments for LAPACK_QRIteration"))
        YALAPACK.gesvd!(A, S, U, Vᴴ)
    elseif alg isa LAPACK_DivideAndConquer
        isempty(alg_kwargs) ||
            throw(ArgumentError("invalid keyword arguments for LAPACK_DivideAndConquer"))
        YALAPACK.gesdd!(A, S, U, Vᴴ)
    elseif alg isa LAPACK_Bisection
        YALAPACK.gesvdx!(A, S, U, Vᴴ; alg_kwargs...)
    elseif alg isa LAPACK_Jacobi
        isempty(alg_kwargs) ||
            throw(ArgumentError("invalid keyword arguments for LAPACK_Jacobi"))
        YALAPACK.gesvj!(A, S, U, Vᴴ)
    else
        throw(ArgumentError("Unsupported SVD algorithm"))
    end

    return S
end

# nothing case here to handle GenericLinearAlgebra
function svd_trunc!(A, USVᴴϵ::Tuple{TU, TS, TVᴴ, Tϵ}, alg::TruncatedAlgorithm) where {TU, TS, TVᴴ, Tϵ}
    U, S, Vᴴ, ϵ = USVᴴϵ
    U, S, Vᴴ = svd_compact!(A, (U, S, Vᴴ), alg.alg)
    USVᴴtrunc, ind = truncate(svd_trunc!, (U, S, Vᴴ), alg.trunc)
    if !isempty(ϵ)
        ϵ[1] = truncation_error!(diagview(S), ind)
    end
    return USVᴴtrunc..., ϵ
end
function svd_trunc!(A, USVᴴϵ::Tuple{Nothing, Tϵ}, alg::TruncatedAlgorithm) where {Tϵ}
    USVᴴ, ϵ = USVᴴϵ
    U, S, Vᴴ = svd_compact!(A, USVᴴ, alg.alg)
    USVᴴtrunc, ind = truncate(svd_trunc!, (U, S, Vᴴ), alg.trunc)
    if !isempty(ϵ)
        ϵ[1] = truncation_error!(diagview(S), ind)
    end
    return USVᴴtrunc..., ϵ
end

function svd_trunc!(A, USVᴴ::Tuple{TU, TS, TVᴴ}, alg::TruncatedAlgorithm; compute_error::Bool = true) where {TU, TS, TVᴴ}
    Tr = real(eltype(A))
    ϵ = compute_error ? zeros(Tr, 1) : zeros(Tr, 0)
    (U, S, Vᴴ, ϵ) = svd_trunc!(A, (USVᴴ..., ϵ), alg)
    return compute_error ? (U, S, Vᴴ, ϵ[1]) : (U, S, Vᴴ, Tr(NaN))
end
function svd_trunc!(A, USVᴴ::Nothing, alg::TruncatedAlgorithm; compute_error::Bool = true)
    ϵ = compute_error ? zeros(real(eltype(A)), 1) : zeros(real(eltype(A)), 0)
    U, S, Vᴴ, ϵ = svd_trunc!(A, (USVᴴ, ϵ), alg)
    return compute_error ? (U, S, Vᴴ, ϵ[1]) : (U, S, Vᴴ, Tr(NaN))
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
    Ad = diagview(A)
    S .= abs.(Ad)
    sort!(S; rev = true)
    return S
end

# GPU logic
# ---------
# placed here to avoid code duplication since much of the logic is replicable across
# CUDA and AMDGPU
###

function check_input(
        ::typeof(svd_trunc!), A::AbstractMatrix, USVᴴ, alg::CUSOLVER_Randomized
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
        ::typeof(svd_trunc!), A::AbstractMatrix, alg::TruncatedAlgorithm{<:CUSOLVER_Randomized}
    )
    m, n = size(A)
    minmn = min(m, n)
    U = similar(A, (m, m))
    S = Diagonal(similar(A, real(eltype(A)), (minmn,)))
    Vᴴ = similar(A, (n, n))
    return (U, S, Vᴴ)
end

function _gpu_gesvd!(
        A::AbstractMatrix, S::AbstractVector, U::AbstractMatrix, Vᴴ::AbstractMatrix
    )
    throw(MethodError(_gpu_gesvd!, (A, S, U, Vᴴ)))
end
function _gpu_Xgesvdp!(
        A::AbstractMatrix, S::AbstractVector, U::AbstractMatrix, Vᴴ::AbstractMatrix; kwargs...
    )
    throw(MethodError(_gpu_Xgesvdp!, (A, S, U, Vᴴ)))
end
function _gpu_Xgesvdr!(
        A::AbstractMatrix, S::AbstractVector, U::AbstractMatrix, Vᴴ::AbstractMatrix; kwargs...
    )
    throw(MethodError(_gpu_Xgesvdr!, (A, S, U, Vᴴ)))
end
function _gpu_gesvdj!(
        A::AbstractMatrix, S::AbstractVector, U::AbstractMatrix, Vᴴ::AbstractMatrix; kwargs...
    )
    throw(MethodError(_gpu_gesvdj!, (A, S, U, Vᴴ)))
end
function _gpu_gesvd_maybe_transpose!(A::AbstractMatrix, S::AbstractVector, U::AbstractMatrix, Vᴴ::AbstractMatrix)
    m, n = size(A)
    m ≥ n && return _gpu_gesvd!(A, S, U, Vᴴ)
    # both CUSOLVER and ROCSOLVER require m ≥ n for gesvd (QR_Iteration)
    # if this condition is not met, do the SVD via adjoint
    minmn = min(m, n)
    Aᴴ = min(m, n) > 0 ? adjoint!(similar(A'), A)::AbstractMatrix : similar(A')
    Uᴴ = similar(U')
    V = similar(Vᴴ')
    if size(U) == (m, m)
        _gpu_gesvd!(Aᴴ, view(S, 1:minmn, 1), V, Uᴴ)
    else
        _gpu_gesvd!(Aᴴ, S, V, Uᴴ)
    end
    length(U) > 0 && adjoint!(U, Uᴴ)
    length(Vᴴ) > 0 && adjoint!(Vᴴ, V)
    return U, S, Vᴴ
end

# GPU SVD implementation
function svd_full!(A::AbstractMatrix, USVᴴ, alg::GPU_SVDAlgorithm)
    check_input(svd_full!, A, USVᴴ, alg)
    U, S, Vᴴ = USVᴴ
    fill!(S, zero(eltype(S)))
    m, n = size(A)
    minmn = min(m, n)
    if minmn == 0
        one!(U)
        zero!(S)
        one!(Vᴴ)
        return USVᴴ
    end

    do_gauge_fix = get(alg.kwargs, :fixgauge, default_fixgauge())::Bool
    alg_kwargs = Base.structdiff(alg.kwargs, NamedTuple{(:fixgauge,)})

    if alg isa GPU_QRIteration
        isempty(alg_kwargs) || @warn "invalid keyword arguments for GPU_QRIteration"
        _gpu_gesvd_maybe_transpose!(A, view(S, 1:minmn, 1), U, Vᴴ)
    elseif alg isa GPU_SVDPolar
        _gpu_Xgesvdp!(A, view(S, 1:minmn, 1), U, Vᴴ; alg_kwargs...)
    elseif alg isa GPU_Jacobi
        _gpu_gesvdj!(A, view(S, 1:minmn, 1), U, Vᴴ; alg_kwargs...)
    else
        throw(ArgumentError("Unsupported SVD algorithm"))
    end
    diagview(S) .= view(S, 1:minmn, 1)
    view(S, 2:minmn, 1) .= zero(eltype(S))

    do_gauge_fix && gaugefix!(svd_full!, U, Vᴴ)

    return USVᴴ
end

function svd_trunc!(A::AbstractMatrix, USVᴴϵ::Tuple{TU, TS, TVᴴ, Tϵ}, alg::TruncatedAlgorithm{<:GPU_Randomized}) where {TU, TS, TVᴴ, Tϵ}
    U, S, Vᴴ, ϵ = USVᴴϵ
    check_input(svd_trunc!, A, (U, S, Vᴴ), alg.alg)
    _gpu_Xgesvdr!(A, S.diag, U, Vᴴ; alg.alg.kwargs...)

    # TODO: make sure that truncation is based on maxrank, otherwise this might be wrong
    (Utr, Str, Vᴴtr), _ = truncate(svd_trunc!, (U, S, Vᴴ), alg.trunc)

    if !isempty(ϵ)
        # normal `truncation_error!` does not work here since `S` is not the full singular value spectrum
        ϵ = sqrt(norm(A)^2 - norm(diagview(Str))^2) # is there a more accurate way to do this?
    end

    do_gauge_fix = get(alg.alg.kwargs, :fixgauge, default_fixgauge())::Bool
    do_gauge_fix && gaugefix!(svd_trunc!, Utr, Vᴴtr)

    return Utr, Str, Vᴴtr, ϵ
end

function svd_compact!(A::AbstractMatrix, USVᴴ, alg::GPU_SVDAlgorithm)
    check_input(svd_compact!, A, USVᴴ, alg)
    U, S, Vᴴ = USVᴴ

    do_gauge_fix = get(alg.kwargs, :fixgauge, default_fixgauge())::Bool
    alg_kwargs = Base.structdiff(alg.kwargs, NamedTuple{(:fixgauge,)})

    if alg isa GPU_QRIteration
        isempty(alg_kwargs) || @warn "invalid keyword arguments for GPU_QRIteration"
        _gpu_gesvd_maybe_transpose!(A, S.diag, U, Vᴴ)
    elseif alg isa GPU_SVDPolar
        _gpu_Xgesvdp!(A, S.diag, U, Vᴴ; alg_kwargs...)
    elseif alg isa GPU_Jacobi
        _gpu_gesvdj!(A, S.diag, U, Vᴴ; alg_kwargs...)
    else
        throw(ArgumentError("Unsupported SVD algorithm"))
    end

    do_gauge_fix && gaugefix!(svd_compact!, U, Vᴴ)

    return USVᴴ
end
_argmaxabs(x) = reduce(_largest, x; init = zero(eltype(x)))
_largest(x, y) = abs(x) < abs(y) ? y : x

function svd_vals!(A::AbstractMatrix, S, alg::GPU_SVDAlgorithm)
    check_input(svd_vals!, A, S, alg)
    U, Vᴴ = similar(A, (0, 0)), similar(A, (0, 0))

    alg_kwargs = Base.structdiff(alg.kwargs, NamedTuple{(:fixgauge,)})

    if alg isa GPU_QRIteration
        isempty(alg_kwargs) || @warn "invalid keyword arguments for GPU_QRIteration"
        _gpu_gesvd_maybe_transpose!(A, S, U, Vᴴ)
    elseif alg isa GPU_SVDPolar
        _gpu_Xgesvdp!(A, S, U, Vᴴ; alg_kwargs...)
    elseif alg isa GPU_Jacobi
        _gpu_gesvdj!(A, S, U, Vᴴ; alg_kwargs...)
    else
        throw(ArgumentError("Unsupported SVD algorithm"))
    end

    return S
end
