function eigh_full_pullback!(ΔA::AbstractMatrix, DV, ΔDV;
                             tol::Real=default_pullback_gaugetol(DV[1]),
                             degeneracy_atol::Real=tol,
                             gauge_atol::Real=tol)

    # Basic size checks and determination
    Dmat, V = DV
    D = diagview(Dmat)
    ΔDmat, ΔV = ΔDV
    n = LinearAlgebra.checksquare(V)
    n == length(D) || throw(DimensionMismatch())

    if !iszerotangent(ΔV)
        VdΔV = V' * ΔV
        aVdΔV = rmul!(VdΔV - VdΔV', 1 / 2)

        mask = abs.(D' .- D) .< degeneracy_atol
        Δgauge = norm(view(aVdΔV, mask))
        Δgauge < gauge_atol ||
            @warn "`eigh` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"

        aVdΔV .*= inv_safe.(D' .- D, tol)

        if !iszerotangent(ΔDmat)
            diagview(aVdΔV) .+= real.(diagview(ΔDmat))
        end
        # recylce VdΔV space
        ΔA = mul!(ΔA, mul!(VdΔV, V, aVdΔV), V', 1, 1)
    elseif !iszerotangent(ΔDmat)
        ΔA = mul!(ΔA, V * Diagonal(real(diagview(ΔDmat))), V', 1, 1)
    end
    return ΔA
end
