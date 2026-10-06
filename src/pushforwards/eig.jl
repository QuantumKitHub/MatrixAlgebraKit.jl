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
        # The diagonal corrections depend on the unnormalized eigenvector tangent.
        mul!(ΔV, V, ∂K)
        if eltype(V) <: Real # fix norm conservation
            diagview(∂K) .-= vec(real.(sum(conj.(V) .* ΔV; dims = 1)))
        else # also fix gauge for `gaugefix!` compatibility
            _, I = findmax(abs, V; dims = 1)
            diagview(∂K) .-= vec(real.(sum(conj.(V) .* ΔV; dims = 1)) .+ im .* imag.(ΔV[I] ./ V[I]))
        end
        # Only the diagonal changed, so add its contribution to the existing tangent.
        mul!(ΔV, V, Diagonal(diagview(∂K)), 1, 1)
    end
    return ΔDV
end

function eig_vals_pushforward!(ΔA, A, DV, ΔD; kwargs...)
    return eig_pushforward!(ΔA, A, DV, (diagonal(ΔD), nothing); kwargs...)
end
