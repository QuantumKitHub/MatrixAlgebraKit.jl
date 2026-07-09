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
function exponential!(A::AbstractMatrix, expA, alg::MatrixFunctionViaTaylor)
    check_input(exponential!, A, expA, alg)
    m = LinearAlgebra.checksquare(A)
    T = eltype(A)
    R = real(T)
    tol = convert(R, get(alg.kwargs, :tol, eps(R)))

    dobalance = get(alg.kwargs, :balance, false)
    scale = dobalance ? balance!(A)[2] : nothing

    # Form a minimal set of powers of A up front and use them to sharpen the norm estimate
    # through the Al-Mohy–Higham quantities dₚ = ‖Aᵖ‖^(1/p).
    p₀ = min(convert(Int, get(alg.kwargs, :estimate_order, 4)), m)
    powers = Vector{typeof(A)}(undef, p₀)
    powers[1] = A
    d = Vector{R}(undef, p₀)
    d[1] = LinearAlgebra.opnorm(A, 1)
    iszero(d[1]) && return one!(expA)
    for p in 2:p₀
        powers[p] = powers[p - 1] * A
        d[p] = LinearAlgebra.opnorm(powers[p], 1)^(1 / p)
    end

    order, blocksize, squarings = taylor_order_and_squarings(d, tol)

    # Rescale the (≤ p₀) powers we hold into powers of A/2ˢ
    resize!(powers, min(p₀, blocksize))
    if squarings > 0
        f = inv(convert(T, 2)^squarings)
        fk = one(T)
        for k in 1:length(powers)
            fk *= f
            rmul!(powers[k], fk) # powers[k] ← (A/2ˢ)ᵏ
        end
    end
    # Extend to `blocksize` powers
    for p in (length(powers) + 1):blocksize
        push!(powers, powers[p - 1] * powers[1])
    end


    # Paterson–Stockmeyer evaluation, followed by squaring
    X = taylor_polynomial(powers, order)
    Y = expA # can reuse this memory
    for _ in 1:squarings
        mul!(Y, X, X)
        X, Y = Y, X
    end

    if dobalance
        expA .= scale .* X ./ transpose(scale)
    else
        X === expA || copyto!(expA, X)
    end
    return expA
end

# Truncation order `m`, blocksize `b` and number of squarings `s` minimizing the total
# matrix-multiplication count subject to the Taylor remainder bound `(θ/2ˢ)ᵐ⁺¹/(m+1)! ≤ tol`.
#
# The Paterson–Stockmeyer scheme (see `taylor_polynomial`) splits the degree-`m` polynomial
# into `K = cld(m, b)` blocks spanning powers A¹..Aᵇ plus a separate constant term, costing
# `max(0, b - p₀)` multiplications to extend the `p₀` powers already formed (those are free)
# and `K - 1` Horner multiplications. Since the polynomial cost is a step function of the
# order while squarings only decrease with it, only the highest order per multiplication
# budget `μ` needs considering: with `max(0, b - p₀) + (K - 1) = μ` the order `K⋅b` equals
# `(μ + p₀ + 1 - b)⋅b`, a downward parabola in `b ≥ p₀` maximized near `(μ + p₀ + 1)/2`.
# For `p₀ = 1` this reproduces the tabulated optimal Paterson–Stockmeyer orders
# 1, 2, 4, 6, 9, 12, 16, 20, 25, ... (Sastre, Linear Algebra Appl. 574 (2019),
# https://www.sciencedirect.com/science/article/pii/S0024379519301454).
#
# The effective norm `θ` per candidate order is the sharpest Al-Mohy–Higham quantity
# `αₚ = max(dₚ, dₚ₊₁)` (with `dₚ = ‖Aᵖ‖^(1/p)`) that is valid for that order: since every
# `k ≥ p(p-1)` is a nonnegative combination of `p` and `p+1`, `‖Aᵏ‖ ≤ αₚᵏ` holds for `k ≥ p(p-1)`,
# so `αₚ` bounds the degree-`m` remainder whenever `m+1 ≥ p(p-1)`.
function taylor_order_and_squarings(d::AbstractVector{<:Real}, tol::Real)
    p₀ = length(d)
    log2tol = Float64(log2(tol)) # log2 before conversion: high-precision `tol` may underflow Float64
    log2factorial = 0.0 # log2((order + 1)!), accumulated incrementally across increasing orders
    prev_order = 0
    best_order = best_blocksize = best_squarings = 0
    best_cost = typemax(Int)
    μ = 0
    while μ < best_cost # cost(μ) ≥ μ, so larger budgets cannot improve
        # highest order reachable with μ multiplications
        b_star = (μ + p₀ + 1) / 2
        order = blocksize = 0
        for bc in (floor(Int, b_star), ceil(Int, b_star))
            b = clamp(bc, p₀, μ + p₀)
            numblocks = μ + p₀ + 1 - b
            if numblocks * b > order
                order = numblocks * b
                blocksize = b
            end
        end
        for k in (prev_order + 2):(order + 1)
            log2factorial += log2(k)
        end
        prev_order = order
        # sharpest valid αₚ = min over p with p(p-1) ≤ order+1 and p+1 ≤ p₀
        θ = d[1]
        for p in 1:(p₀ - 1)
            p * (p - 1) ≤ order + 1 || break
            θ = min(θ, max(d[p], d[p + 1]))
        end
        # `θ == 0` (a nilpotent Aᵖ) makes the remainder vanish exactly ⇒ no squarings needed;
        # guard the `log2(0) = -Inf` before it reaches `ceil(Int, ⋅)`.
        excess = Float64(log2(θ)) - (log2tol + log2factorial) / (order + 1)
        squarings = excess > 0 ? ceil(Int, excess) : 0
        cost = μ + squarings
        if cost ≤ best_cost # ties → larger budget → fewer squarings (avoid overscaling)
            best_cost = cost
            best_order = order
            best_blocksize = blocksize
            best_squarings = squarings
        end
        squarings == 0 && break # cost can only grow beyond this point
        μ += 1
    end
    return best_order, best_blocksize, best_squarings
end

# Evaluate ∑ₖ₌₀ᵐ Aᵏ/k! via the Paterson–Stockmeyer scheme, returning a freshly allocated matrix.
# `powers` holds A, A², …, A^blocksize (already scaled). The polynomial is split as
# c₀ I + ∑ⱼ₌₁ᴷ A^((j-1)b) Bⱼ with blocks Bⱼ = ∑ᵢ₌₁ᵇ c_{(j-1)b+i} Aⁱ and K = cld(order, blocksize),
# evaluated by Horner over A^b using K - 1 multiplications; the constant term is added to the
# diagonal at the end.
function taylor_polynomial(powers::AbstractVector{<:AbstractMatrix}, order::Integer)
    A = powers[1]
    T = eltype(A)
    blocksize = length(powers)

    invfactorial = Vector{T}(undef, order + 1)
    invfactorial[1] = one(T)
    for k in 1:order
        invfactorial[k + 1] = invfactorial[k] / k
    end

    result = similar(A)
    block = similar(A)
    numblocks = cld(order, blocksize)
    taylor_block!(result, powers, invfactorial, numblocks, blocksize, order)
    for blockindex in (numblocks - 1):-1:1
        mul!(block, result, powers[blocksize])
        taylor_block!(result, powers, invfactorial, blockindex, blocksize, order)
        result .+= block
    end
    diagview(result) .+= invfactorial[1]
    return result
end

# Block Bⱼ = ∑ᵢ₌₁ᵇ c_{offset+i} Aⁱ with offset = (blockindex - 1) * blocksize, truncated at
# total degree `order`, written into `out`.
function taylor_block!(out::AbstractMatrix, powers, invfactorial, blockindex::Integer, blocksize::Integer, order::Integer)
    offset = (blockindex - 1) * blocksize
    fill!(out, zero(eltype(out)))
    for i in 1:blocksize
        k = offset + i
        k > order && break
        out .+= invfactorial[k + 1] .* powers[i]
    end
    return out
end
