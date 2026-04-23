function eig_pushforward!(ΔA, A, DV, ΔDV; kwargs...)
    D, V = DV
    ΔD, ΔV = ΔDV
    iVΔAV = inv(V) * ΔA * V
    diagview(ΔD) .= diagview(iVΔAV)
    if !iszerotangent(ΔV)
        F = 1 ./ (transpose(diagview(D)) .- diagview(D))
        fill!(diagview(F), zero(eltype(F)))
        K̇ = F .* iVΔAV
        mul!(ΔV, V, K̇, 1, 0)
    end
    return ΔDV
end

function eig_trunc_pushforward!(ΔA, A, DV, ΔDV; kwargs...) end

function eig_vals_pushforward!(ΔA, A, DV, ΔD; kwargs...)
    return eig_pushforward!(ΔA, A, DV, ΔD; kwargs...)
end
