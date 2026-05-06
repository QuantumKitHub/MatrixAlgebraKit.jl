# Inputs / defaults / outputs
# ---------------------------
copy_input(::typeof(left_sketch), A) = A
copy_input(::typeof(right_sketch), A) = A

function initialize_output(::typeof(left_sketch!), A::AbstractMatrix, alg::GaussianSketching)
    m, n = size(A)
    k = min(alg.howmany, m, n)
    T = float(eltype(A))
    Q = similar(A, T, (m, k))
    B = similar(A, T, (k, n))
    return Q, B
end
function initialize_output(::typeof(right_sketch!), A::AbstractMatrix, alg::GaussianSketching)
    return initialize_output(left_sketch!, A, alg)
end

function check_input(::typeof(left_sketch!), A::AbstractMatrix, (Q, B), alg::GaussianSketching)
    m, n = size(A)
    k = min(alg.howmany, m, n)
    @assert Q isa AbstractMatrix
    @check_size(Q, (m, k))
    @check_scalar(Q, A, float)
    @assert B isa AbstractMatrix
    @check_size(B, (k, n))
    @check_scalar(B, A, float)
    return nothing
end
function check_input(::typeof(right_sketch!), A::AbstractMatrix, BPᴴ, alg::GaussianSketching)
    check_input(left_sketch!, A, BPᴴ, alg)
    return nothing
end

# Gaussian sketching, native implementation
# -----------------------------------------
function left_sketch!(A::AbstractMatrix, QB, alg::GaussianSketching)
    check_input(left_sketch!, A, QB, alg)
    Q, B = QB
    Ω = Random.randn!(alg.rng, similar(Q, (size(A, 2), size(Q, 2))))
    Y = A * Ω
    R = similar(Y, (0, 0))
    Q, _ = left_orth!(Y, (Q, R))
    for _ in 2:alg.numiter
        mul!(Ω, A', Q)
        mul!(Y, A, Ω)
        Q, _ = left_orth!(Y, (Q, R))
    end
    return Q, mul!(B, Q', A)
end

function right_sketch!(A::AbstractMatrix, BPᴴ, alg::GaussianSketching)
    check_input(right_sketch!, A, BPᴴ, alg)
    B, Pᴴ = BPᴴ
    Ω = Random.randn!(alg.rng, similar(Pᴴ, (size(Pᴴ, 1), size(A, 1))))
    Y = Ω * A
    L = similar(Y, (0, 0))
    _, Pᴴ = right_orth!(Y, (L, Pᴴ))
    for _ in 2:alg.numiter
        mul!(Ω, Pᴴ, A')
        mul!(Y, Ω, A)
        _, Pᴴ = right_orth!(Y, (L, Pᴴ))
    end
    return mul!(B, A, Pᴴ'), Pᴴ
end
