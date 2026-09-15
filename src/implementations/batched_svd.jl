# Inputs
# ------
copy_input(::typeof(batched_svd_full), As::AbstractVector{<:AbstractMatrix}) = map(A -> copy!(similar(A, float(eltype(A))), A), As)
copy_input(::typeof(batched_svd_full), A::AbstractArray{T, 3}) where {T} = copy!(similar(A, float(T)), A)
copy_input(::typeof(batched_svd_compact), A) = copy_input(batched_svd_full, A)
copy_input(::typeof(batched_svd_vals), A) = copy_input(batched_svd_full, A)

function check_input(::typeof(batched_svd_full!), A::AbstractVector{<:AbstractMatrix}, USVᴴ, ::AbstractAlgorithm)
    @assert all(==(size(first(A))), size.(A))
    m, n = size(first(A))
    batch_size = length(A)
    U, S, Vᴴ = USVᴴ
    @assert U isa AbstractArray && S isa AbstractArray && Vᴴ isa AbstractArray
    @check_size(U, (m, m, batch_size))
    @check_scalar(U, first(A))
    @check_size(S, (m, n, batch_size))
    @check_scalar(S, first(A), real)
    @check_size(Vᴴ, (n, n, batch_size))
    @check_scalar(Vᴴ, first(A))
    return nothing
end
function check_input(::typeof(batched_svd_compact!), A::AbstractVector{<:AbstractMatrix}, USVᴴ, ::AbstractAlgorithm)
    @assert all(==(size(first(A))), size.(A))
    m, n = size(first(A))
    batch_size = length(A)
    minmn = min(m, n)
    U, S, Vᴴ = USVᴴ
    @assert U isa AbstractArray && S isa AbstractArray && Vᴴ isa AbstractArray
    @check_size(U, (m, minmn, batch_size))
    @check_scalar(U, first(A))
    @check_size(S, (minmn, batch_size))
    @check_scalar(S, first(A), real)
    @check_size(Vᴴ, (minmn, n, batch_size))
    @check_scalar(Vᴴ, first(A))
    return nothing
end
function check_input(::typeof(batched_svd_vals!), A::AbstractVector{<:AbstractMatrix}, S, ::AbstractAlgorithm)
    @assert all(==(size(first(A))), size.(A))
    m, n = size(first(A))
    batch_size = length(A)
    minmn = min(m, n)
    @assert S isa AbstractMatrix
    @check_size(S, (minmn, batch_size))
    @check_scalar(S, first(A), real)
    return nothing
end
# ragged batches: matrices of different sizes, each with its own outputs
function check_input(
        ::typeof(batched_svd_compact!), A::AbstractVector{<:AbstractMatrix},
        USVᴴ::Tuple{AbstractVector{<:AbstractMatrix}, AbstractVector{<:AbstractVector}, AbstractVector{<:AbstractMatrix}},
        alg::AbstractAlgorithm
    )
    Us, Ss, Vᴴs = USVᴴ
    length(Us) == length(Ss) == length(Vᴴs) == length(A) ||
        throw(DimensionMismatch("expected $(length(A)) outputs for each of U, S and Vᴴ"))
    for (a, u, s, vᴴ) in zip(A, Us, Ss, Vᴴs)
        check_input(svd_compact!, a, (u, Diagonal(s), vᴴ), alg)
    end
    return nothing
end
function check_input(
        ::typeof(batched_svd_full!), A::AbstractVector{<:AbstractMatrix},
        USVᴴ::Tuple{AbstractVector{<:AbstractMatrix}, AbstractVector{<:AbstractMatrix}, AbstractVector{<:AbstractMatrix}},
        alg::AbstractAlgorithm
    )
    Us, Ss, Vᴴs = USVᴴ
    length(Us) == length(Ss) == length(Vᴴs) == length(A) ||
        throw(DimensionMismatch("expected $(length(A)) outputs for each of U, S and Vᴴ"))
    for (a, u, s, vᴴ) in zip(A, Us, Ss, Vᴴs)
        check_input(svd_full!, a, (u, s, vᴴ), alg)
    end
    return nothing
end
function check_input(
        ::typeof(batched_svd_vals!), A::AbstractVector{<:AbstractMatrix},
        S::AbstractVector{<:AbstractVector}, alg::AbstractAlgorithm
    )
    length(S) == length(A) || throw(DimensionMismatch("expected $(length(A)) outputs for S"))
    for (a, s) in zip(A, S)
        check_input(svd_vals!, a, s, alg)
    end
    return nothing
end
function check_input(::typeof(batched_svd_full!), A::AbstractArray{T, 3}, USVᴴ, ::AbstractAlgorithm) where {T}
    m, n, batch_size = size(A)
    U, S, Vᴴ = USVᴴ
    @assert U isa AbstractArray && S isa AbstractArray && Vᴴ isa AbstractArray
    @check_size(U, (m, m, batch_size))
    @check_scalar(U, A)
    @check_size(S, (m, n, batch_size))
    @check_scalar(S, A, real)
    @check_size(Vᴴ, (n, n, batch_size))
    @check_scalar(Vᴴ, A)
    return nothing
end
function check_input(::typeof(batched_svd_compact!), A::AbstractArray{T, 3}, USVᴴ, ::AbstractAlgorithm) where {T}
    m, n, batch_size = size(A)
    minmn = min(m, n)
    U, S, Vᴴ = USVᴴ
    @assert U isa AbstractArray && S isa AbstractArray && Vᴴ isa AbstractArray
    @check_size(U, (m, minmn, batch_size))
    @check_scalar(U, A)
    @check_size(S, (minmn, batch_size))
    @check_scalar(S, A, real)
    @check_size(Vᴴ, (minmn, n, batch_size))
    @check_scalar(Vᴴ, A)
    return nothing
end
function check_input(::typeof(batched_svd_vals!), A::AbstractArray{T, 3}, S, ::AbstractAlgorithm) where {T}
    m, n, batch_size = size(A)
    minmn = min(m, n)
    @assert S isa AbstractMatrix
    @check_size(S, (minmn, batch_size))
    @check_scalar(S, A, real)
    return nothing
end

# Outputs
# -------
# a vector of matrices, which may have different sizes, gets one output per matrix
function initialize_output(::typeof(batched_svd_full!), A::AbstractVector{<:AbstractMatrix}, ::AbstractAlgorithm)
    Us = [similar(a, (size(a, 1), size(a, 1))) for a in A]
    Ss = [similar(a, real(eltype(a)), size(a)) for a in A] # TODO: Rectangular diagonal type?
    Vᴴs = [similar(a, (size(a, 2), size(a, 2))) for a in A]
    return (Us, Ss, Vᴴs)
end
function initialize_output(::typeof(batched_svd_full!), A::AbstractArray{T, 3}, ::AbstractAlgorithm) where {T}
    m, n, batch_size = size(A)
    U = similar(A, (m, m, batch_size))
    S = similar(A, real(eltype(A)), (m, n, batch_size))
    Vᴴ = similar(A, (n, n, batch_size))
    return (U, S, Vᴴ)
end
function initialize_output(::typeof(batched_svd_compact!), A::AbstractVector{<:AbstractMatrix}, ::AbstractAlgorithm)
    Us = [similar(a, (size(a, 1), minimum(size(a)))) for a in A]
    Ss = [similar(a, real(eltype(a)), minimum(size(a))) for a in A]
    Vᴴs = [similar(a, (minimum(size(a)), size(a, 2))) for a in A]
    return (Us, Ss, Vᴴs)
end
function initialize_output(::typeof(batched_svd_compact!), A::AbstractArray{T, 3}, ::AbstractAlgorithm) where {T}
    m, n, batch_size = size(A)
    minmn = min(m, n)
    U = similar(A, (m, minmn, batch_size))
    S = similar(A, real(eltype(A)), (minmn, batch_size))
    Vᴴ = similar(A, (minmn, n, batch_size))
    return (U, S, Vᴴ)
end
function initialize_output(::typeof(batched_svd_vals!), A::AbstractVector{<:AbstractMatrix}, ::AbstractAlgorithm)
    return [similar(a, real(eltype(a)), minimum(size(a))) for a in A]
end
function initialize_output(::typeof(batched_svd_vals!), A::AbstractArray{T, 3}, ::AbstractAlgorithm) where {T}
    m, n, batch_size = size(A)
    return similar(A, real(eltype(A)), (min(m, n), batch_size))
end

for f! in (:gesdd_batched!, :gesvd_batched!, :gesvdj_batched!, :gesvdx_batched!)
    @eval $f!(driver::Driver, args...) = throw(ArgumentError("$driver does not provide $($(f!))"))
end

# Adjoint of every matrix in a batch, i.e. `dst[:, :, i] = adjoint(src[:, :, i])`.
function batched_adjoint!(dst::AbstractArray{<:Any, 3}, src::AbstractArray{<:Any, 3})
    isempty(dst) && return dst
    permutedims!(dst, src, (2, 1, 3))
    eltype(dst) <: Real || (dst .= conj.(dst))
    return dst
end
function batched_adjoint(A::AbstractArray{<:Any, 3})
    return batched_adjoint!(similar(A, (size(A, 2), size(A, 1), size(A, 3))), A)
end
batched_adjoint(A::AbstractVector{<:AbstractMatrix}) = map(a -> adjoint!(similar(a'), a), A)

"""
    batched_svd_via_adjoint!(f!, driver, A, S, U, Vᴴ; kwargs...)

Compute the SVD of every matrix in the batch `A` (m × n, m < n) by computing the SVD of their
adjoints using the provided function `f!(driver, A, S, U, Vᴴ; kwargs...)`. Use this as a
building block for drivers whose batched SVD routines require m ≥ n, mirroring
[`svd_via_adjoint!`](@ref).
"""
function batched_svd_via_adjoint!(f!::F, driver::Driver, A, S, U, Vᴴ; kwargs...) where {F}
    Aᴴ = batched_adjoint(A)
    V = similar(Vᴴ, (size(Vᴴ, 2), size(Vᴴ, 1), size(Vᴴ, 3)))
    Uᴴ = similar(U, (size(U, 2), size(U, 1), size(U, 3)))
    f!(driver, Aᴴ, S, V, Uᴴ; kwargs...)
    length(U) > 0 && batched_adjoint!(U, Uᴴ)
    length(Vᴴ) > 0 && batched_adjoint!(Vᴴ, V)
    return S, U, Vᴴ
end

for (f, f_lapack!, Alg) in (
        (:divide_and_conquer, :gesdd_batched!, :DivideAndConquer),
        (:qr_iteration, :gesvd_batched!, :QRIteration),
        (:bisection, :gesvdx_batched!, :Bisection),
        (:jacobi, :gesvdj_batched!, :Jacobi),
    )
    svd_compact_f! = Symbol(:batched_svd_compact_, f, :!)
    svd_full_f! = Symbol(:batched_svd_full_, f, :!)
    svd_vals_f! = Symbol(:batched_svd_vals_, f, :!)

    # MatrixAlgebraKit wrappers
    @eval begin
        function batched_svd_compact!(A::AbstractVector{<:AbstractMatrix}, USVᴴ, alg::$Alg)
            check_input(batched_svd_compact!, A, USVᴴ, alg)
            return $svd_compact_f!(A, USVᴴ...; alg.kwargs...)
        end
        function batched_svd_compact!(A::AbstractArray{T, 3}, USVᴴ, alg::$Alg) where {T}
            check_input(batched_svd_compact!, A, USVᴴ, alg)
            return $svd_compact_f!(A, USVᴴ...; alg.kwargs...)
        end
        function batched_svd_full!(A::AbstractVector{<:AbstractMatrix}, USVᴴ, alg::$Alg)
            check_input(batched_svd_full!, A, USVᴴ, alg)
            return $svd_full_f!(A, USVᴴ...; alg.kwargs...)
        end
        function batched_svd_full!(A::AbstractArray{T, 3}, USVᴴ, alg::$Alg) where {T}
            check_input(batched_svd_full!, A, USVᴴ, alg)
            return $svd_full_f!(A, USVᴴ...; alg.kwargs...)
        end
        function batched_svd_vals!(A::AbstractVector{<:AbstractMatrix}, S, alg::$Alg)
            check_input(batched_svd_vals!, A, S, alg)
            return $svd_vals_f!(A, S; alg.kwargs...)
        end
        function batched_svd_vals!(A::AbstractArray{T, 3}, S, alg::$Alg) where {T}
            check_input(batched_svd_vals!, A, S, alg)
            return $svd_vals_f!(A, S; alg.kwargs...)
        end

        # ragged batches: pack into 3D batches, see `_ragged_batches`
        function batched_svd_compact!(
                A::AbstractVector{<:AbstractMatrix},
                USVᴴ::Tuple{AbstractVector{<:AbstractMatrix}, AbstractVector{<:AbstractVector}, AbstractVector{<:AbstractMatrix}},
                alg::$Alg
            )
            check_input(batched_svd_compact!, A, USVᴴ, alg)
            Us, Ss, Vᴴs = USVᴴ
            batches, rest = _ragged_batches(A, alg)
            for (inds, (m, n)) in batches
                Ab = _ragged_pack(A, inds, m, n)
                Ub, Sb, Vᴴb = batched_svd_compact!(Ab, initialize_output(batched_svd_compact!, Ab, alg), alg)
                for (j, i) in enumerate(inds)
                    copyto!(Us[i], view(Ub, axes(Us[i])..., j))
                    copyto!(Ss[i], view(Sb, axes(Ss[i], 1), j))
                    copyto!(Vᴴs[i], view(Vᴴb, axes(Vᴴs[i])..., j))
                end
            end
            for i in rest
                svd_compact!(A[i], (Us[i], Diagonal(Ss[i]), Vᴴs[i]), alg)
            end
            return USVᴴ
        end
        function batched_svd_full!(
                A::AbstractVector{<:AbstractMatrix},
                USVᴴ::Tuple{AbstractVector{<:AbstractMatrix}, AbstractVector{<:AbstractMatrix}, AbstractVector{<:AbstractMatrix}},
                alg::$Alg
            )
            check_input(batched_svd_full!, A, USVᴴ, alg)
            Us, Ss, Vᴴs = USVᴴ
            # zero padding mixes the padded dimensions into the complements of the full
            # `U` and `Vᴴ`, so only matrices of equal size are batched
            batches, rest = _ragged_batches(A, alg; pad = false)
            for (inds, (m, n)) in batches
                Ab = _ragged_pack(A, inds, m, n)
                Ub, Sb, Vᴴb = batched_svd_full!(Ab, initialize_output(batched_svd_full!, Ab, alg), alg)
                for (j, i) in enumerate(inds)
                    copyto!(Us[i], view(Ub, :, :, j))
                    copyto!(Ss[i], view(Sb, :, :, j))
                    copyto!(Vᴴs[i], view(Vᴴb, :, :, j))
                end
            end
            for i in rest
                svd_full!(A[i], (Us[i], Ss[i], Vᴴs[i]), alg)
            end
            return USVᴴ
        end
        function batched_svd_vals!(
                A::AbstractVector{<:AbstractMatrix}, S::AbstractVector{<:AbstractVector}, alg::$Alg
            )
            check_input(batched_svd_vals!, A, S, alg)
            batches, rest = _ragged_batches(A, alg)
            for (inds, (m, n)) in batches
                Ab = _ragged_pack(A, inds, m, n)
                Sb = batched_svd_vals!(Ab, initialize_output(batched_svd_vals!, Ab, alg), alg)
                for (j, i) in enumerate(inds)
                    copyto!(S[i], view(Sb, axes(S[i], 1), j))
                end
            end
            for i in rest
                svd_vals!(A[i], S[i], alg)
            end
            return S
        end
    end

    # driver
    @eval begin
        @inline $svd_compact_f!(A, U, S, Vᴴ; driver::Driver = DefaultDriver(), kwargs...) = $svd_compact_f!(driver, A, U, S, Vᴴ; kwargs...)
        @inline $svd_full_f!(A, U, S, Vᴴ; driver::Driver = DefaultDriver(), kwargs...) = $svd_full_f!(driver, A, U, S, Vᴴ; kwargs...)
        @inline $svd_vals_f!(A, S; driver::Driver = DefaultDriver(), kwargs...) = $svd_vals_f!(driver, A, S; kwargs...)
        @inline $svd_compact_f!(::DefaultDriver, A::AbstractVector{<:AbstractMatrix}, U, S, Vᴴ; kwargs...) = $svd_compact_f!(default_driver($Alg, A), A, U, S, Vᴴ; kwargs...)
        @inline $svd_compact_f!(::DefaultDriver, A::AbstractArray{<:Any, 3}, U, S, Vᴴ; kwargs...) = $svd_compact_f!(default_driver($Alg, A), A, U, S, Vᴴ; kwargs...)
        @inline $svd_full_f!(::DefaultDriver, A::AbstractVector{<:AbstractMatrix}, U, S, Vᴴ; kwargs...) = $svd_full_f!(default_driver($Alg, A), A, U, S, Vᴴ; kwargs...)
        @inline $svd_full_f!(::DefaultDriver, A::AbstractArray{<:Any, 3}, U, S, Vᴴ; kwargs...) = $svd_full_f!(default_driver($Alg, A), A, U, S, Vᴴ; kwargs...)
        @inline $svd_vals_f!(::DefaultDriver, A::AbstractVector{<:AbstractMatrix}, S::AbstractMatrix; kwargs...) = $svd_vals_f!(default_driver($Alg, A), A, S; kwargs...)
        @inline $svd_vals_f!(::DefaultDriver, A::AbstractArray{<:Any, 3}, S::AbstractMatrix; kwargs...) = $svd_vals_f!(default_driver($Alg, A), A, S; kwargs...)
    end

    # Implementation
    @eval begin
        function $svd_compact_f!(driver::Driver, A, U, S, Vᴴ; fixgauge::Bool = true, kwargs...)
            isempty(A) && return one!(U), zero!(S), one!(Vᴴ)
            $f_lapack!(driver, A, S, U, Vᴴ; kwargs...)
            if fixgauge
                for (u, vᴴ) in zip(eachslice(U, dims = 3), eachslice(Vᴴ, dims = 3))
                    gaugefix!(svd_compact!, u, vᴴ)
                end
            end
            return U, S, Vᴴ
        end
        function $svd_full_f!(driver::Driver, A, U, S, Vᴴ; fixgauge::Bool = true, kwargs...)
            supports_svd_full(driver, $(QuoteNode(f))) ||
                throw(ArgumentError(LazyString("driver ", driver, " does not provide `$($(QuoteNode(f_lapack!)))`")))
            isempty(A) && return one!(U), zero!(S), one!(Vᴴ)
            zero!(S)
            m, n, batch_size = size(S)
            minmn = min(m, n)
            Sd = similar(S, (minmn, batch_size))
            $f_lapack!(driver, A, Sd, U, Vᴴ; kwargs...)
            for (s, sd) in zip(eachslice(S, dims = 3), eachslice(Sd, dims = 2))
                diagview(s) .= sd
            end
            if fixgauge
                for (u, vᴴ) in zip(eachslice(U, dims = 3), eachslice(Vᴴ, dims = 3))
                    gaugefix!(svd_full!, u, vᴴ)
                end
            end
            return U, S, Vᴴ
        end
        function $svd_vals_f!(driver::Driver, A::AbstractArray{T, 3}, S::AbstractMatrix; fixgauge::Bool = true, kwargs...) where {T}
            isempty(A) && return zero!(S)
            U, Vᴴ = similar(A, (0, 0, 0)), similar(A, (0, 0, 0))
            $f_lapack!(driver, A, S, U, Vᴴ; kwargs...)
            return S
        end
        function $svd_vals_f!(driver::Driver, A::AbstractVector{<:AbstractMatrix}, S::AbstractMatrix; fixgauge::Bool = true, kwargs...)
            isempty(A) && return zero!(S)
            U, Vᴴ = similar(first(A), (0, 0, 0)), similar(first(A), (0, 0, 0))
            $f_lapack!(driver, A, S, U, Vᴴ; kwargs...)
            return S
        end
    end
end

# Ragged batches
# --------------
"""
    max_batched_blocksize(alg, T::Type) -> Int

Largest matrix dimension that the batched driver for `alg` accepts for arrays of type `T`.
Larger matrices in a ragged batch are decomposed one at a time instead. Unlimited by default.
"""
max_batched_blocksize(::AbstractAlgorithm, ::Type) = typemax(Int)

# Fewest matrices in a ragged batch that are worth a batched call
# Should this be settable by the user?
const BATCHED_SVD_THRESHOLD::Int = 4

# Split a ragged batch into batches the driver can handle: matrices of equal size are
# batched together, and, if `pad`, whatever is left over is zero-padded into one more batch.
# Returns the batches as `(indices, (m, n))` pairs, and the indices of the matrices that have
# to be decomposed one at a time.
# TODO: should everything be padded into ONE batch?
function _ragged_batches(A::AbstractVector{<:AbstractMatrix}, alg::AbstractAlgorithm; pad::Bool = true)
    batches = Tuple{Vector{Int}, Tuple{Int, Int}}[]
    rest = Int[]
    isempty(A) && return batches, rest
    lim = max_batched_blocksize(alg, typeof(first(A)))
    needs_tall = requires_tall(alg)
    groups = Dict{Tuple{Int, Int}, Vector{Int}}()
    for i in eachindex(A)
        push!(get!(Vector{Int}, groups, size(A[i])), i)
    end
    for ((m, n), inds) in groups
        if length(inds) >= BATCHED_SVD_THRESHOLD && max(m, n) <= lim && (!needs_tall || m >= n)
            push!(batches, (inds, (m, n)))
        else
            append!(rest, inds)
        end
    end
    pad || return batches, rest
    # Zero padding leaves the leading `min(m, n)` singular values and vectors of every input
    # untouched. Pad to a square only when the algorithm requires `m ≥ n`
    # (currently only `QRIteration`).
    m = maximum(i -> size(A[i], 1), rest; init = 0)
    n = maximum(i -> size(A[i], 2), rest; init = 0)
    padded = needs_tall ? (max(m, n), max(m, n)) : (m, n)
    if length(rest) >= BATCHED_SVD_THRESHOLD && maximum(padded) <= lim
        push!(batches, (rest, padded))
        rest = Int[]
    end
    return batches, rest
end

# Copy `A[inds]` into a single `(m, n, length(inds))` batch, zero-padding where needed.
function _ragged_pack(A::AbstractVector{<:AbstractMatrix}, inds, m::Int, n::Int)
    uniform = all(i -> size(A[i]) == (m, n), inds)
    # `stack` can't zero-pad
    # On the GPU it falls back to scalar indexing for matrices that are views
    uniform && A isa AbstractVector{<:Array} && return stack(view(A, inds))
    Ab = similar(A[first(inds)], (m, n, length(inds)))
    uniform || zero!(Ab) # only need to zero if matrices are ragged
    for (j, i) in enumerate(inds)
        copyto!(view(Ab, axes(A[i])..., j), A[i])
    end
    return Ab
end
