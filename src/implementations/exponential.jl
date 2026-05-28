# Inputs
# ------
function copy_input(::typeof(exponential), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end

copy_input(::typeof(exponential), A::Diagonal) = copy(A)

function check_input(::typeof(exponential!), A::AbstractMatrix, expA::AbstractMatrix, alg::AbstractAlgorithm)
    m = LinearAlgebra.checksquare(A)
    @check_size(expA, (m, m))
    return @check_scalar(expA, A)
end

function check_input(::typeof(exponential!), A::AbstractMatrix, expA::AbstractMatrix, ::DiagonalAlgorithm)
    m = LinearAlgebra.checksquare(A)
    isdiag(A) || throw(DimensionMismatch("diagonal input matrix expected"))
    @assert expA isa Diagonal
    @check_size(expA, (m, m))
    @check_scalar(expA, A)
    return nothing
end

# Outputs
# -------
initialize_output(::typeof(exponential!), A::AbstractMatrix, ::AbstractAlgorithm) = A

# Implementation
# --------------
function exponential!(A, expA, alg::MatrixFunctionViaTaylor)
    check_input(exponential!, A, expA, alg)
    return exponential_via_taylor!(A, expA)
end


module TaylorExponential

"""
    exponential_via_taylor!(A::Matrix{T}) where T <: AbstractFloat

An implementation of the Fasi & Higham (2018) Taylor-based scaling 
and squaring for arbitrary precision.
"""
function exponential_via_taylor!(A::AbstractMatrix{T}, expA::AbstractMatrix{T}) where T
    n = size(A, 1)
    ϵ = eps(T)
    Apowers = fill(A, 8) # Preallocate for powers up to A^8, will be resized if needed
    for k = 2:length(Apowers)
        Apowers[k] = Apowers[k-1] * A
    end
    ρA = opnorm(Apowers[end], 1)^(1/length(Apowers)) # estimate of the spectral radius using Gelfand's formula

    # Find m and s such that (ρA/2^s)^(m+1) / (m+1)! < ϵ
    m, s = optimal_taylor_order(ρA, ϵ)
    
    # Scale A down by 2^s
    A ./= T(2)^s
    
    # Evaluate Taylor via Paterson-Stockmeyer approach
    X = evaluate_taylor_ps(As, m)
    
    # Squaring to undo the scaling
    for _ in 1:s
        X = X * X
    end
    
    return X
end

"""
Evaluates Taylor series using Paterson-Stockmeyer logic.
"""
function evaluate_taylor_ps(A, m)
    T = eltype(A)
    k = Int(floor(sqrt(m))) # Chunk size for Paterson-Stockmeyer
    
    # Precompute powers A^2, ..., A^k
    powers = Vector{typeof(A)}(undef, k)
    powers[1] = A
    for i in 2:k
        powers[i] = powers[i-1] * A
    end
    
    # Horner-like evaluation of the outer polynomial
    # e^A ≈ ∑ (A^k)^j * P_j(A)
    res = zeros(T, size(A))
    num_chunks = Int(ceil((m+1)/k))
    
    for j in (num_chunks-1):-1:0
        # Evaluate sub-polynomial P_j(A) of degree k-1
        poly_chunk = zeros(T, size(A))
        for i in 0:k-1
            idx = j*k + i
            if idx > m; continue; end
            coeff = 1 / factorial(big(idx))
            if i == 0
                poly_chunk += T(coeff) * I
            else
                poly_chunk += T(coeff) * powers[i]
            end
        end
        
        if j == num_chunks - 1
            res = poly_chunk
        else
            res = res * powers[k] + poly_chunk
        end
    end
    return res
end

function get_ps_costs(max_m) # number of matrix multiplications for an order m polynomial
    costs = Dict{Int, Int}()
    for m in 1:max_m
        # Find k in 1:m that minimizes (k-1) + ceil(m/k) - 1
        best_c = m # Default naive cost
        for k in 1:m
            c = (k - 1) + div(m-1, k) # == (k - 1) + ceil(Int, m/k) - 1
            if c < best_c
                best_c = c
            end
        end
        costs[m] = best_c
    end
    return costs
end

# To filter only for the "efficient" m (where degree increases for the same cost)
function get_efficient_m(max_m::Int)
    costs = get_ps_costs(max_m)
    efficient = Dict{Int, Int}()
    next_cost = costs[1]
    for m in 1:max_m
        cost = next_cost
        next_cost = m < max_m ? costs[m+1] : cost + 1
        if cost < next_cost
            efficient[m] = cost
        end
    end
    return efficient
end

PS_COSTS = get_efficient_m(100)

"""
Optimizes m and s to minimize cost ≈ m_mults + s.
"""
function optimal_taylor_order(ρA, ϵ)
    # In a full Fasi-Higham implementation, this would use a small 
    # search loop or a cost-model lookup. 
    # Here is a simplified version for BigFloat:
    best_m, best_s = 0, 0
    min_cost = typemax(Int)
    
    # Search over efficient Paterson-Stockmeyer degrees
    for (m, c) in PS_COSTS
        # Calculate s needed for this m: (normA/2^s)^(m+1) / (m+1)! < ϵ
        # log2(normA/2^s) * (m+1) - log2((m+1)!) < log2(ϵ)
        # log2(normA) - s - [log2((m+1)!) / (m+1)] < log2(ϵ) / (m+1)
        term = (log2(ρA) - (log2(factorial(big(m+1))) / (m+1)) - (log2(ϵ) / (m+1)))
        s = max(0, ceil(Int, term))
        
        cost = c + s
        if cost < min_cost || (cost == min_cost && m < best_m)
            min_cost = cost
            best_m, best_s = m, s
        end
    end
    return best_m, best_s
end

end