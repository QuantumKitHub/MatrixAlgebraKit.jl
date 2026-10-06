# Inputs
# ------
copy_input(::typeof(batched_eigh_full), As::AbstractVector{<:AbstractMatrix}) = map(Base.Fix1(copy_input, eigh_full), As)
copy_input(::typeof(batched_eigh_full), A::AbstractArray{T, 3}) where {T} = copy!(similar(A, float(T)), A)
copy_input(::typeof(batched_eigh_vals), A) = copy_input(batched_eigh_full, A)

# ragged batches: matrices of different sizes, each with its own outputs
function check_input(
        ::typeof(batched_eigh_full!), A::AbstractVector{<:AbstractMatrix},
        DV::Tuple{AbstractVector{<:Diagonal}, AbstractVector{<:AbstractMatrix}},
        alg::AbstractAlgorithm
    )
    Ds, Vs = DV
    length(Ds) == length(Vs) == length(A) ||
        throw(DimensionMismatch("expected $(length(A)) outputs for each of D and V"))
    for (a, d, v) in zip(A, Ds, Vs)
        check_input(eigh_full!, a, (d, v), alg)
    end
    return nothing
end
function check_input(
        ::typeof(batched_eigh_vals!), A::AbstractVector{<:AbstractMatrix},
        D::AbstractVector{<:AbstractVector}, alg::AbstractAlgorithm
    )
    length(D) == length(A) || throw(DimensionMismatch("expected $(length(A)) outputs for D"))
    for (a, d) in zip(A, D)
        check_input(eigh_vals!, a, d, alg)
    end
    return nothing
end
function check_input(::typeof(batched_eigh_full!), A::AbstractVector{<:AbstractMatrix}, DV, ::AbstractAlgorithm)
    isempty(A) && return nothing
    m, n = size(first(A))
    @assert all(==((n, n)) ∘ size, A)
    batch_size = length(A)
    D, V = DV
    @assert D isa AbstractArray && V isa AbstractArray
    @check_size(D, (n, batch_size))
    @check_scalar(D, first(A), real)
    @check_size(V, (n, n, batch_size))
    @check_scalar(V, first(A))
    return nothing
end
function check_input(::typeof(batched_eigh_vals!), A::AbstractVector{<:AbstractMatrix}, D, ::AbstractAlgorithm)
    isempty(A) && return nothing
    m, n = size(first(A))
    @assert all(==((n, n)) ∘ size, A)
    batch_size = length(A)
    @assert D isa AbstractMatrix
    @check_size(D, (n, batch_size))
    @check_scalar(D, first(A), real)
    return nothing
end
function check_input(::typeof(batched_eigh_full!), A::AbstractArray{T, 3}, DV, ::AbstractAlgorithm) where {T}
    m, n, batch_size = size(A)
    D, V = DV
    @assert D isa AbstractArray && V isa AbstractArray
    @check_size(D, (n, batch_size))
    @check_scalar(D, A, real)
    @check_size(V, (n, n, batch_size))
    @check_scalar(V, A)
    return nothing
end
function check_input(::typeof(batched_eigh_vals!), A::AbstractArray{T, 3}, D, ::AbstractAlgorithm) where {T}
    m, n, batch_size = size(A)
    @assert m == n
    @assert D isa AbstractMatrix
    @check_size(D, (n, batch_size))
    @check_scalar(D, A, real)
    return nothing
end

# Outputs
# -------
# a vector of matrices, which may have different sizes, gets one output per matrix
function initialize_output(::typeof(batched_eigh_full!), A::AbstractVector{<:AbstractMatrix}, ::AbstractAlgorithm)
    Ds = [Diagonal(similar(a, real(eltype(a)), first(size(a)))) for a in A]
    Vs = [similar(a, (size(a, 1), size(a, 2))) for a in A]
    return (Ds, Vs)
end
function initialize_output(::typeof(batched_eigh_full!), A::AbstractArray{T, 3}, ::AbstractAlgorithm) where {T}
    m, n, batch_size = size(A)
    D = similar(A, real(eltype(A)), (n, batch_size))
    V = similar(A, (n, n, batch_size))
    return (D, V)
end
function initialize_output(::typeof(batched_eigh_vals!), A::AbstractVector{<:AbstractMatrix}, ::AbstractAlgorithm)
    return [similar(a, real(eltype(a)), first(size(a))) for a in A]
end
function initialize_output(::typeof(batched_eigh_vals!), A::AbstractArray{T, 3}, ::AbstractAlgorithm) where {T}
    m, n, batch_size = size(A)
    return similar(A, real(eltype(A)), (n, batch_size))
end

for f! in (:heevj_batched!, :heev_batched!)
    @eval $f!(driver::Driver, args...) = throw(ArgumentError("$driver does not provide $($(f!))"))
end

for (f, f_lapack!, Alg) in (
        (:qr_iteration, :heev_batched!, :QRIteration),
        (:jacobi, :heevj_batched!, :Jacobi),
    )
    eigh_full_f! = Symbol(:batched_eigh_full_, f, :!)
    eigh_vals_f! = Symbol(:batched_eigh_vals_, f, :!)

    # MatrixAlgebraKit wrappers
    @eval begin
        function batched_eigh_full!(A::AbstractVector{<:AbstractMatrix}, DV, alg::$Alg)
            check_input(batched_eigh_full!, A, DV, alg)
            driver = get(alg.kwargs, :driver, default_driver(alg, eltype(A)))
            supports_pointer_batch(alg, driver, eltype(A)) ||
                throw(ArgumentError(LazyString("driver ", driver, " does not suppport ragged (non-uniform) batches")))
            return $eigh_full_f!(A, DV...; alg.kwargs...)
        end
        function batched_eigh_full!(A::AbstractArray{T, 3}, DV, alg::$Alg) where {T}
            check_input(batched_eigh_full!, A, DV, alg)
            return $eigh_full_f!(A, DV...; alg.kwargs...)
        end
        function batched_eigh_vals!(A::AbstractVector{<:AbstractMatrix}, D, alg::$Alg)
            check_input(batched_eigh_vals!, A, D, alg)
            driver = get(alg.kwargs, :driver, default_driver(alg, eltype(A)))
            supports_pointer_batch(alg, driver, eltype(A)) ||
                throw(ArgumentError(LazyString("driver ", driver, " does not suppport ragged (non-uniform) batches")))
            return $eigh_vals_f!(A, D; alg.kwargs...)
        end
        function batched_eigh_vals!(A::AbstractArray{T, 3}, D, alg::$Alg) where {T}
            check_input(batched_eigh_vals!, A, D, alg)
            return $eigh_vals_f!(A, D; alg.kwargs...)
        end

        # ragged batches: pack into 3D batches, see `_ragged_batches`
        function batched_eigh_full!(
                A::AbstractVector{<:AbstractMatrix},
                DV::Tuple{AbstractVector{<:Diagonal}, AbstractVector{<:AbstractMatrix}},
                alg::$Alg
            )
            check_input(batched_eigh_full!, A, DV, alg)
            Ds, Vs = DV
            # zero padding mixes the padded dimensions into the complements of the full
            # `V`, so only matrices of equal size are batched
            batches, rest = _ragged_batches(A, alg; pad = false)
            for (inds, (m, n)) in batches
                Ab = _ragged_pack(A, inds, m, n, alg)
                Db, Vb = batched_eigh_full!(Ab, _packed_output(batched_eigh_full!, Ab, alg), alg)
                for (j, i) in enumerate(inds)
                    copyto!(Ds[i], view(Db, :, :, j))
                    copyto!(Vs[i], view(Vb, :, :, j))
                end
            end
            for i in rest
                eigh_full!(A[i], (Ds[i], Vs[i]), alg)
            end
            return DV
        end
        function batched_eigh_vals!(
                A::AbstractVector{<:AbstractMatrix}, D::AbstractVector{<:AbstractVector}, alg::$Alg
            )
            check_input(batched_eigh_vals!, A, D, alg)
            batches, rest = _ragged_batches(A, alg)
            for (inds, (m, n)) in batches
                Ab = _ragged_pack(A, inds, m, n, alg)
                Db = batched_eigh_vals!(Ab, _packed_output(batched_eigh_vals!, Ab, alg), alg)
                for (j, i) in enumerate(inds)
                    copyto!(D[i], view(Db, axes(D[i], 1), j))
                end
            end
            for i in rest
                eigh_vals!(A[i], D[i], alg)
            end
            return D
        end
    end

    # driver
    @eval begin
        @inline $eigh_full_f!(A, D, V; driver::Driver = DefaultDriver(), kwargs...) = $eigh_full_f!(driver, A, D, V; kwargs...)
        @inline $eigh_vals_f!(A, D; driver::Driver = DefaultDriver(), kwargs...) = $eigh_vals_f!(driver, A, D; kwargs...)
        @inline $eigh_full_f!(::DefaultDriver, A::AbstractVector{<:AbstractMatrix}, D, V; kwargs...) = $eigh_full_f!(default_driver($Alg, A), A, D, V; kwargs...)
        @inline $eigh_full_f!(::DefaultDriver, A::AbstractArray{<:Any, 3}, D, V; kwargs...) = $eigh_full_f!(default_driver($Alg, A), A, D, V; kwargs...)
        @inline $eigh_vals_f!(::DefaultDriver, A::AbstractVector{<:AbstractMatrix}, D::AbstractMatrix; kwargs...) = $eigh_vals_f!(default_driver($Alg, A), A, D; kwargs...)
        @inline $eigh_vals_f!(::DefaultDriver, A::AbstractArray{<:Any, 3}, D::AbstractMatrix; kwargs...) = $eigh_vals_f!(default_driver($Alg, A), A, D; kwargs...)
    end

    # Implementation
    @eval begin
        function $eigh_full_f!(driver::Driver, A, D, V; fixgauge::Bool = true, kwargs...)
            if _isempty_batch(A)
                zero!(D)
                foreach(one!, eachslice(V, dims = 3))
                return D, V
            end
            zero!(D)
            $f_lapack!(driver, A, D, V; kwargs...)
            fixgauge && gaugefix!(batched_eigh_full!, V)
            return D, V
        end
        function $eigh_vals_f!(driver::Driver, A::AbstractArray{T, 3}, D::AbstractMatrix; fixgauge::Bool = true, kwargs...) where {T}
            _isempty_batch(A) && return zero!(D)
            V = similar(A, (0, 0, size(A, 3)))
            $f_lapack!(driver, A, D, V; kwargs...)
            return D
        end
        function $eigh_vals_f!(driver::Driver, A::AbstractVector{<:AbstractMatrix}, D::AbstractMatrix; fixgauge::Bool = true, kwargs...)
            _isempty_batch(A) && return zero!(D)
            V = similar(first(A), (0, 0, length(A)))
            $f_lapack!(driver, A, D, V; kwargs...)
            return D
        end
    end
end

# Fewest matrices in a ragged batch that are worth a batched call
# Should this be settable by the user?
const BATCHED_EIGH_THRESHOLD::Int = 4

function _packed_output(::typeof(batched_eigh_full!), A::AbstractVector{<:AbstractMatrix}, ::AbstractAlgorithm)
    a = first(A)
    m, n = size(a)
    b = length(A)
    return (similar(a, real(eltype(a)), (n, b)), similar(a, (n, n, b)))
end
function _packed_output(::typeof(batched_eigh_vals!), A::AbstractVector{<:AbstractMatrix}, ::AbstractAlgorithm)
    a = first(A)
    return similar(a, real(eltype(a)), (first(size(a)), length(A)))
end
