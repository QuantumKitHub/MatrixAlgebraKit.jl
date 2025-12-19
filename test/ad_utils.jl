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
function stabilize_eigvals!(D)
    absD = abs.(D)
    p = invperm(sortperm(absD)) # rank of abs(D)
    for i in 1:(length(D) - 1)
        if absD[i] == absD[i + 1]
            p[p .>= p[i + 1]] .-= 1
        end
    end
    n = maximum(p)
    radii = 1 / n * ((1:n) + rand(real(eltype(D)), n) / 2)
    for i in 1:length(D)
        D[i] = sign(D[i]) * radii[p[i]]
    end
    return D
end
function make_eig_matrix(rng, T, n)
    A = randn(rng, T, n, n)
    D, V = eig_full(A)
    Ddiag = diagview(D)
    stabilize_eigvals!(Ddiag)
    if T <: Real
        A = real(V * D * inv(V))
    else
        A = V * D * inv(V)
    end
    return A
end
function make_eigh_matrix(rng, T, n)
    A = project_hermitian!(randn(rng, T, n, n))
    D, V = eigh_full(A)
    Ddiag = diagview(D)
    stabilize_eigvals!(Ddiag)
    A = project_hermitian!(V * D * V')
    return A
end

precision(::Type{T}) where {T <: Number} = sqrt(eps(real(T)))
