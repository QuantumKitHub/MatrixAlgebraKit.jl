function remove_svdgauge_dependence!(
        ΔU, ΔVᴴ, U, S, Vᴴ;
        degeneracy_atol = MatrixAlgebraKit.default_pullback_gaugetol(S)
    )
    gaugepart = mul!(U' * ΔU, Vᴴ, ΔVᴴ', true, true)
    gaugepart = project_antihermitian!(gaugepart)
    gaugepart[abs.(transpose(diagview(S)) .- diagview(S)) .>= degeneracy_atol] .= 0
    mul!(ΔU, U, gaugepart, -1, 1)
    return ΔU, ΔVᴴ
end
function remove_eiggauge_dependence!(
        ΔV, D, V;
        degeneracy_atol = MatrixAlgebraKit.default_pullback_gaugetol(S)
    )
    gaugepart = V' * ΔV
    gaugepart[abs.(transpose(diagview(D)) .- diagview(D)) .>= degeneracy_atol] .= 0
    mul!(ΔV, V / (V' * V), gaugepart, -1, 1)
    return ΔV
end
function remove_eighgauge_dependence!(
        ΔV, D, V;
        degeneracy_atol = MatrixAlgebraKit.default_pullback_gaugetol(S)
    )
    gaugepart = V' * ΔV
    gaugepart = project_antihermitian!(gaugepart)
    gaugepart[abs.(transpose(diagview(D)) .- diagview(D)) .>= degeneracy_atol] .= 0
    mul!(ΔV, V, gaugepart, -1, 1)
    return ΔV
end
function stabilize_eigvals!(D::AbstractVector)
    absD = abs.(D)
    p = invperm(sortperm(absD)) # rank of abs(D)
    # account for exact degeneracies in absolute value when having complex conjugate pairs
    for i in 1:(length(D) - 1)
        if absD[i] == absD[i + 1] # conjugate pairs will appear sequentially
            p[p .>= p[i + 1]] .-= 1 # lower the rank of all higher ones
        end
    end
    n = maximum(p)
    # rescale eigenvalues so that they lie on distinct radii in the complex plane
    # that are chosen randomly in non-overlapping intervals [k/n, (k+0.5)/n)] for k=1,...,n
    radii = ((1:n) .+ rand(real(eltype(D)), n) ./ 2) ./ n
    for i in 1:length(D)
        D[i] = sign(D[i]) * radii[p[i]]
    end
    return D
end
function make_eig_matrix(rng, T, n)
    A = randn(rng, T, n, n)
    D, V = eig_full(A)
    stabilize_eigvals!(diagview(D))
    Ac = V * D * inv(V)
    return (T <: Real) ? real(Ac) : Ac
end
function make_eigh_matrix(rng, T, n)
    A = project_hermitian!(randn(rng, T, n, n))
    D, V = eigh_full(A)
    stabilize_eigvals!(diagview(D))
    return project_hermitian!(V * D * V')
end

precision(::Type{T}) where {T <: Number} = sqrt(eps(real(T)))
