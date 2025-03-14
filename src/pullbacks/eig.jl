function eig_full_pullback!(ΔA::AbstractMatrix, DV, ΔDV;
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

        mask = abs.(transpose(D) .- D) .< degeneracy_atol
        Δgauge = norm(view(VdΔV, mask), Inf)
        Δgauge < gauge_atol ||
            @warn "`eig` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"

        VdΔV .*= conj.(inv_safe.(transpose(D) .- D, degeneracy_atol))

        if !iszerotangent(ΔDmat)
            diagview(VdΔV) .+= diagview(ΔDmat)
        end
        PΔV = V' \ VdΔV
        if eltype(ΔA) <: Real
            ΔAc = mul!(VdΔV, PΔV, V') # recycle VdΔV memory
            ΔA .+= real.(ΔAc)
        else
            ΔA = mul!(ΔA, PΔV, V', 1, 1)
        end
    elseif !iszerotangent(ΔDmat)
        PΔV = V' \ Diagonal(diagview(ΔDmat))
        if eltype(ΔA) <: Real
            ΔAc = PΔV * V'
            ΔA .+= real.(ΔAc)
        else
            ΔA = mul!(ΔA, PΔV, V', 1, 1)
        end
    end
    return ΔA
end
