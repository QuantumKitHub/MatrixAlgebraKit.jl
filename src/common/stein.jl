# Solvers for the Stein equation X - G X Diagonal(w) = B, i.e. (1 - wᵢ G) xᵢ = bᵢ per column.
# Used in the pullback of truncated decompositions, with G = P Pᴴ or Pᴴ P, wᵢ = 1/σᵢ² for the case of SVD,
# and G = P, wᵢ = 1/λᵢ for the case of EIG(H). Here, P the part of A outside the kept vectors.
# Naive iteration requires γᵢ, the spectral radius of wᵢ G, to be smaller than 1.

"""
    accelerative_smith_iteration!(X, Xₙ, G, w, atol, maxiter)

Solve `X = B + G * X * Diagonal(w)` by summing the Neumann series
`X = Σₖ Gᵏ * B * Diagonal(w)ᵏ` by doubling (Smith's method), i.e. by repeatedly adding
`G^(2ʲ) * X * Diagonal(w)^(2ʲ)` to `X` until the norm of that increment drops below `atol`,
for at most `maxiter` steps.

On entry, `X` contains `B`, and it is overwritten with the result. `Xₙ` is used as a buffer,
and `G` and `w` are overwritten. `w` is normalized such that `maximum(abs, w) == 1`, so that
squaring it can only shrink it; `G` is scaled by the inverse factor to compensate.

Reference: https://doi.org/10.1016/j.aml.2009.01.012.
"""
function accelerative_smith_iteration!(X, Xₙ, G, w, atol, maxiter)
    Gₙ = similar(G)
    wmax = maximum(abs, w)
    w ./= wmax
    G .*= wmax
    for k in 1:maxiter
        Xₙ = rmul!(mul!(Xₙ, G, X), Diagonal(w))
        if maximum(abs, Xₙ) < atol
            break
        end
        X .+= Xₙ
        if k == maxiter
            @warn "Sylvester iteration did not converge after $k iterations, final norm of X: $(maximum(abs, X))"
            break
        end
        w .= w .^ 2
        Gₙ = mul!(Gₙ, G, G)
        G, Gₙ = Gₙ, G
    end
    return X
end

# smallest Ritz value: smallest eigenvalue of the Lanczos tridiagonal from the CG coefficients `α`, `β`
function _cg_ritz_min(α, β)
    l = length(α)
    d = similar(α)
    d[1] = 1 / α[1]
    for j in 2:l
        d[j] = 1 / α[j] + β[j - 1] / α[j - 1]
    end
    e = sqrt.(β[1:(l - 1)]) ./ α[1:(l - 1)]
    return LinearAlgebra.eigmin(LinearAlgebra.SymTridiagonal(d, e))
end

# (1 - wᵢ G) applied to the columns of Z
_stein_op(G, w, Z) = Z .- (G * Z) .* transpose(w)

# iterations until the squared residual norm `r` drops below `tol²`, at the faster of its rates over
# the last `l` iterations (from `r₀`) and all `m` (from `rᵢ`)
function _cg_remaining(r, r₀, rᵢ, tol, l, m)
    q = min(log(r / r₀) / l, log(r / rᵢ) / m) # logarithm of the reduction per iteration
    return q < 0 ? log(tol^2 / r) / q : oftype(float(r), Inf)
end

# real parts of the column-wise inner products of A and B, on the device of A and B. CPU arrays use
# `dot` per column, which avoids the temporary. This is restricted to `Matrix` rather than
# `StridedMatrix`, which GPU arrays also are: there it would return a CPU vector, one `dot` call
# each, which the GPU broadcasts in `hermitian_stein_cg!` cannot mix with the device arrays.
_coldots(A, B) = vec(real(sum(conj.(A) .* B; dims = 1)))
const _CPUMatrix = Union{Matrix, SubArray{<:Any, 2, <:Matrix}}
_coldots(A::_CPUMatrix, B::_CPUMatrix) = [real(LinearAlgebra.dot(view(A, :, j), view(B, :, j))) for j in axes(A, 2)]

"""
    hermitian_stein_cg!(X, applyG!, formG, w, atol, maxiter; cost_apply, cost_apply_formed, cost_form, cost_square = nothing)

Solve `X - G * X * Diagonal(w) = B` for Hermitian `G` by conjugate gradients on all columns at
once, in at most `maxiter` iterations: O(n² k) per iteration for k columns, against O(n³) per
doubling step. Only the products wᵢ G enter, so neither needs normalizing. `X` contains `B` on
entry and is overwritten with the result.

`applyG!(Y, Z)` sets `Y = G * Z` without forming `G`, and `formG()` returns `G`. `G` is formed
once the predicted remaining applications save more than `cost_form`, given the costs per column
`cost_apply` (by `applyG!`) and `cost_apply_formed` (by `G`); `cost_form = 0` forms it at once.
With `cost_square`, for indefinite `G` (eigh), the solver restarts on the residual equation
multiplied by 1 + wᵢ G, (1 - wᵢ² G²) xᵢ = (1 + wᵢ G) bᵢ, once half the predicted remaining cost
exceeds `cost_square` (forming G² and the new right-hand side): its spectrum [1 - γᵢ², 1] needs
about half the iterations. A column stops once its residual is below `atol` times the smallest
Ritz value of the slowest column, so that its error is below about `atol`.
"""
function hermitian_stein_cg!(
        X, applyG!, formG, w, atol, maxiter;
        cost_apply, cost_apply_formed = cost_apply, cost_form, cost_square = nothing, nprobe::Int = 5
    )
    RT = real(eltype(X))
    G = iszero(cost_form) ? formG() : nothing
    tol = RT(atol) # stopping residual 2-norm, refined by the Ritz values
    ρ = _coldots(X, X)
    cols = findall(ρ .> tol^2) # active columns, kept contiguous at the front
    nact = length(cols)
    iszero(nact) && return fill!(X, zero(eltype(X)))
    R = X[:, cols] # X keeps B until the end
    P = zero(R)
    Q = similar(R)
    Xc = zero(R)
    wc = w[cols]
    ρ = ρ[cols]
    β = zero(ρ)
    ρ₀ = copy(ρ) # squared residual norms at the previous check
    ρᵢ = copy(ρ) # and at the start
    αs = [RT[] for _ in 1:nact]
    βs = [RT[] for _ in 1:nact]
    for numiter in 1:maxiter
        if numiter > 1 && (numiter - 1) % nprobe == 0
            c = argmax(abs.(view(wc, 1:nact))) # the slowest column
            tol = atol * _cg_ritz_min(αs[c], βs[c])
            nact = _cg_compact!(
                view(ρ, 1:nact), tol, view(R, :, 1:nact), view(P, :, 1:nact), view(Xc, :, 1:nact), view(wc, 1:nact),
                view(β, 1:nact), view(ρ₀, 1:nact), view(ρᵢ, 1:nact), view(cols, 1:nact), view(αs, 1:nact), view(βs, 1:nact)
            )
            iszero(nact) && break
            # predicted column applications
            napply = sum(_cg_remaining.(view(ρ, 1:nact), view(ρ₀, 1:nact), view(ρᵢ, 1:nact), tol, nprobe, numiter - 1))
            if isnothing(G) && napply * (cost_apply - cost_apply_formed) > cost_form
                G = formG()
            end
            if !isnothing(cost_square) && napply * cost_apply / 2 > cost_square # restart on the squared equation
                isnothing(G) && (G = formG())
                X₁ = zero(X) # the current solution
                X₁[:, cols] .= Xc
                R = X .- _stein_op(G, w, X₁)
                X .= R .+ (G * R) .* transpose(w)
                G² = G * G
                hermitian_stein_cg!(X, nothing, () -> G², w .^ 2, atol, maxiter; cost_apply, cost_form = 0, nprobe)
                return X .+= X₁
            end
            ρ₀ .= ρ
        end
        Pₐ, Rₐ, Qₐ, wₐ = view(P, :, 1:nact), view(R, :, 1:nact), view(Q, :, 1:nact), view(wc, 1:nact)
        ρₐ, βₐ = view(ρ, 1:nact), view(β, 1:nact)
        Pₐ .= Rₐ .+ Pₐ .* transpose(βₐ)
        isnothing(G) ? applyG!(Qₐ, Pₐ) : mul!(Qₐ, G, Pₐ)
        Qₐ .= Pₐ .- Qₐ .* transpose(wₐ) # q = (1 - wᵢ G) p
        α = ρₐ ./ _coldots(Pₐ, Qₐ)
        view(Xc, :, 1:nact) .+= Pₐ .* transpose(α)
        Rₐ .-= Qₐ .* transpose(α)
        βₐ .= ρₐ # ρold
        ρₐ .= _coldots(Rₐ, Rₐ)
        βₐ .= ρₐ ./ βₐ
        for (j, a, b) in zip(1:nact, Array(α), Array(βₐ))
            push!(αs[j], a)
            push!(βs[j], b)
        end
        nact = _cg_compact!(
            ρₐ, tol, Rₐ, Pₐ, view(Xc, :, 1:nact), wₐ,
            βₐ, view(ρ₀, 1:nact), view(ρᵢ, 1:nact), view(cols, 1:nact), view(αs, 1:nact), view(βs, 1:nact)
        )
        iszero(nact) && break
    end
    iszero(nact) || @warn "conjugate gradients did not converge in $maxiter iterations, largest residual norm: $(sqrt(maximum(view(ρ, 1:nact))))"
    fill!(X, zero(eltype(X)))
    X[:, cols] .= Xc
    return X
end

# move the columns with `ρ > tol²` to the front of all arrays (in order) and return their number
function _cg_compact!(ρ, tol, arrays...)
    keep = Array(ρ) .> tol^2
    all(keep) && return length(keep)
    perm = vcat(findall(keep), findall(.!keep))
    for a in (ρ, arrays...)
        if a isa AbstractMatrix
            a .= a[:, perm]
        else
            a .= a[perm]
        end
    end
    return count(keep)
end
