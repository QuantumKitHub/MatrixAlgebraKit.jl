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

precision(::Type{T}) where {T <: Number} = sqrt(eps(real(T)))
