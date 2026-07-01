# Inputs
# ------
function copy_input(::typeof(exponential), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end

copy_input(::typeof(exponential), A::Diagonal) = copy(A)
copy_input(::typeof(exponential), (τ, A)::Tuple{Number, AbstractMatrix}) = (τ, copy!(similar(A, float(eltype(A))), A))
copy_input(::typeof(exponential), (τ, A)::Tuple{Number, Diagonal}) = τ, copy(A)

function check_input(::typeof(exponential!), A::AbstractMatrix, expA, alg::AbstractAlgorithm)
    m = LinearAlgebra.checksquare(A)
    @check_size(expA, (m, m))
    @check_scalar(expA, A)
    return nothing
end

function check_input(::typeof(exponential!), A::AbstractMatrix, expA, ::DiagonalAlgorithm)
    m = LinearAlgebra.checksquare(A)
    @assert isdiag(A)
    @assert expA isa Diagonal
    @check_size(expA, (m, m))
    @check_scalar(expA, A)
    return nothing
end

function check_input(::typeof(exponential!), (τ, A)::Tuple{Number, AbstractMatrix}, expA, alg::AbstractAlgorithm)
    m = LinearAlgebra.checksquare(A)
    @check_size(expA, (m, m))
    @check_scalar(expA, A, (τ isa Real) ? identity : complex)
    return nothing
end

function check_input(::typeof(exponential!), (τ, A)::Tuple{Number, AbstractMatrix}, expA, ::DiagonalAlgorithm)
    m = LinearAlgebra.checksquare(A)
    @assert isdiag(A)
    @assert expA isa Diagonal
    @check_size(expA, (m, m))
    @check_scalar(expA, A, (τ isa Real) ? identity : complex)
    return nothing
end

# Algorithm selection
# ---------------------------
exponential!(A::AbstractMatrix, alg::DefaultAlgorithm) = exponential!(A, select_algorithm(exponential!, A, nothing; alg.kwargs...))
exponential!(A::AbstractMatrix, out, alg::DefaultAlgorithm) = exponential!(A, out, select_algorithm(exponential!, A, nothing; alg.kwargs...))
exponential!(τA::Tuple{Number, AbstractMatrix}, alg::DefaultAlgorithm) = exponential!(τA, select_algorithm(exponential!, τA, nothing; alg.kwargs...))
exponential!(τA::Tuple{Number, AbstractMatrix}, out, alg::DefaultAlgorithm) = exponential!(τA, out, select_algorithm(exponential!, τA, nothing; alg.kwargs...))

# Outputs
# -------
initialize_output(::typeof(exponential!), A::AbstractMatrix, ::AbstractAlgorithm) = A
initialize_output(::typeof(exponential!), (τ, A)::Tuple{T, AbstractMatrix}, ::AbstractAlgorithm) where {T <: Real} = A
initialize_output(::typeof(exponential!), (τ, A)::Tuple{Number, AbstractMatrix}, ::AbstractAlgorithm) = complex(A)

# Implementation
# --------------
function exponential!(A::AbstractMatrix, expA, alg::MatrixFunctionViaLA)
    check_input(exponential!, A, expA, alg)
    A = LinearAlgebra.exp!(A)
    A === expA || copy!(expA, A)
    return expA
end

exponential!(A::AbstractMatrix, expA, alg::MatrixFunctionViaEigh) = exponential!((one(eltype(A)), A), expA, alg)
exponential!(A::AbstractMatrix, expA, alg::MatrixFunctionViaEig) = exponential!((one(eltype(A)), A), expA, alg)

function exponential!((τ, A)::Tuple{Number, AbstractMatrix}, expA, alg::AbstractAlgorithm)
    expA .= A .* τ
    return exponential!(expA, expA, alg)
end

function exponential!((τ, A)::Tuple{Number, AbstractMatrix}, expA, alg::MatrixFunctionViaEigh)
    check_input(exponential!, (τ, A), expA, alg)
    D, V = eigh_full!(A, alg.eigh_alg)
    if eltype(A) <: Real
        if eltype(τ) <: Real
            VexpD = rmul!(V, exponential!((τ / 2, D), D))
        else
            VexpD = V * exponential((τ / 2, D))
        end
        return mul!(expA, VexpD, transpose(VexpD))
    else
        if eltype(τ) <: Real
            VexpD = V * exponential!((τ, D), D)
        else
            VexpD = V * exponential((τ, D))
        end
        return mul!(expA, VexpD, V')
    end
end

function exponential!((τ, A)::Tuple{Number, AbstractMatrix}, expA, alg::MatrixFunctionViaEig)
    check_input(exponential!, (τ, A), expA, alg)
    D, V = eig_full!(A, alg.eig_alg)
    if eltype(A) <: Real && eltype(τ) <: Real
        VexpD = V * exponential!((τ, D), D)
        expAc = rdiv!(VexpD, LinearAlgebra.lu!(V))
        return expA .= real.(expAc)
    else
        expA .= V .* transpose(diagview(exponential!((τ, D), D)))
        return rdiv!(expA, LinearAlgebra.lu!(V))
    end
end

# Diagonal logic
# --------------
function exponential!(A::AbstractMatrix, expA, alg::DiagonalAlgorithm)
    check_input(exponential!, A, expA, alg)
    return map_diagonal!(exp, expA, A)
end

function exponential!((τ, A)::Tuple{Number, AbstractMatrix}, expA, alg::DiagonalAlgorithm)
    check_input(exponential!, (τ, A), expA, alg)
    return map_diagonal!(x -> exp(x * τ), expA, A)
end

# Taylor logic
# ------------
function exponential!(A::AbstractMatrix, expA::AbstractMatrix, alg::MatrixFunctionViaTaylor)
    check_input(exponential!, A, expA, alg)
    T = eltype(A)
    tol = convert(real(T), get(alg.kwargs, :tol, eps(real(T))))
    scale = get(alg.kwargs, :balance, true) ? balance!(A)[2] : ones(real(T), size(A, 1))

    θ = LinearAlgebra.opnorm(A, 1)
    if iszero(θ)
        fill!(expA, zero(T))
        diagview(expA) .= one(T)
        return expA
    end

    order, squarings = taylor_order_and_squarings(θ, tol)
    squarings > 0 && rmul!(A, inv(convert(T, 2)^squarings))

    X = taylor_polynomial(A, order)
    if squarings > 0
        Y = similar(X)
        for _ in 1:squarings
            mul!(Y, X, X)
            X, Y = Y, X
        end
    end

    expA .= scale .* X ./ transpose(scale)
    return expA
end

# Truncation order `m` and number of squarings `s` minimizing the Paterson–Stockmeyer
# matrix-multiplication count subject to the Taylor remainder bound `(θ/2ˢ)ᵐ⁺¹/(m+1)! ≤ tol`.
function taylor_order_and_squarings(θ::Real, tol::Real)
    log2θ = log2(Float64(θ))
    log2tol = log2(Float64(tol))
    log2factorial = 0.0
    best_order = 1
    best_squarings = 0
    best_cost = typemax(Int)
    for order in 1:100
        log2factorial += log2(order + 1)
        squarings = max(0, ceil(Int, log2θ - (log2tol + log2factorial) / (order + 1)))
        blocksize = ceil(Int, sqrt(order + 1))
        cost = (blocksize - 1) + (cld(order + 1, blocksize) - 1) + squarings
        if cost < best_cost
            best_cost = cost
            best_order = order
            best_squarings = squarings
        end
    end
    return best_order, best_squarings
end

# Evaluate ∑ₖ₌₀ᵐ Aᵏ/k! via the Paterson–Stockmeyer scheme, returning a freshly allocated matrix.
function taylor_polynomial(A::AbstractMatrix, order::Integer)
    T = eltype(A)
    blocksize = ceil(Int, sqrt(order + 1))

    invfactorial = Vector{T}(undef, order + 1)
    invfactorial[1] = one(T)
    for k in 1:order
        invfactorial[k + 1] = invfactorial[k] / k
    end

    powers = Vector{typeof(A)}(undef, blocksize)
    powers[1] = A
    for p in 2:blocksize
        powers[p] = powers[p - 1] * A
    end

    result = similar(A)
    block = similar(A)
    numblocks = fld(order, blocksize)
    taylor_block!(result, powers, invfactorial, numblocks, blocksize, order)
    for j in (numblocks - 1):-1:0
        mul!(block, result, powers[blocksize])
        taylor_block!(result, powers, invfactorial, j, blocksize, order)
        result .+= block
    end
    return result
end

# Sub-polynomial ∑ᵢ₌₀ᵇ⁻¹ c_{jb+i} Aⁱ of degree `blocksize - 1`, written into `out`.
function taylor_block!(out::AbstractMatrix, powers, invfactorial, j::Integer, blocksize::Integer, order::Integer)
    base = j * blocksize
    fill!(out, zero(eltype(out)))
    diagview(out) .= invfactorial[base + 1]
    for i in 1:(blocksize - 1)
        k = base + i
        k > order && break
        out .+= invfactorial[k + 1] .* powers[i]
    end
    return out
end
