function eig_pushforward!(
        ΔA, A, DV, ΔDV;
        degeneracy_atol::Real = default_pullback_rank_atol(DV[1])
    )
    D, V = DV
    ΔD, ΔV = ΔDV
    ΔAV = isnothing(ΔV) ? ΔA * V : mul!(ΔV, ΔA, V) # reusing ΔV memory if possible
    ∂K = V \ ΔAV
    if !iszerotangent(ΔD)
        diagview(ΔD) .= diagview(∂K)
    end
    if !iszerotangent(ΔV)
        ∂K .*= inv_safe.(transpose(diagview(D)) .- diagview(D), degeneracy_atol)
        mul!(ΔV, V, ∂K)
        if eltype(V) <: Real # fix norm conservation
            ΔV .-= V .* real.(sum(conj.(V) .* ΔV; dims = 1))
        else # also fix gauge for `gaugefix!` compatibility
            _, I = findmax(abs, V; dims = 1)
            ΔV .-= V .* (real.(sum(conj.(V) .* ΔV; dims = 1)) .+ im .* imag.(ΔV[I] ./ V[I]))
        end
    end
    return ΔDV
end

function eig_vals_pushforward!(ΔA, A, DV, ΔD; kwargs...)
    return eig_pushforward!(ΔA, A, DV, (diagonal(ΔD), nothing); kwargs...)
end
