"""
    left_polar_pullback!(ΔA, (W, P), (ΔW, ΔP))

Adds the pullback from the left polar decomposition of `A` to `ΔA` given the output `(W, P)` and
cotangent `(ΔW, ΔP)` of `left_polar(A)`.                        
"""
function left_polar_pullback!(ΔA::AbstractMatrix, WP, ΔWP)

    # Extract the Polar components
    W, P = WP

    # Extract and check the cotangents
    ΔW, ΔP = ΔWP
    if !iszerotangent(ΔP)
        ΔP = (ΔP + ΔP') / 2
    end
    M = zero(P)
    !iszerotangent(ΔW) && mul!(M, W', ΔW, 1, 1)
    !iszerotangent(ΔP) && mul!(M, ΔP, P, -1, 1)
    C = sylvester(P, P, M' - M)
    C .+= ΔP
    ΔA = mul!(ΔA, W, C, 1, 1)
    if !iszerotangent(ΔW)
        ΔWP = ΔW / P
        WdΔWP = W' * ΔWP
        ΔWP = mul!(ΔWP, W, WdΔWP, -1, 1)
        ΔA .+= ΔWP
    end
    return ΔA
end

"""
    right_polar_pullback!(ΔA, (P, Wᴴ), (ΔP, ΔWᴴ))

Adds the pullback from the left polar decomposition of `A` to `ΔA` given the output `(P, Wᴴ)` and
cotangent `(ΔP, ΔWᴴ)` of `right_polar(A)`.                        
"""
function right_polar_pullback!(ΔA::AbstractMatrix, PWᴴ, ΔPWᴴ)

    # Extract the Polar components
    P, Wᴴ = PWᴴ

    # Extract and check the cotangents
    ΔP, ΔWᴴ = ΔPWᴴ
    if !iszerotangent(ΔP)
        ΔP = (ΔP + ΔP') / 2
    end
    M = zero(P)
    !iszerotangent(ΔWᴴ) && mul!(M, ΔWᴴ, Wᴴ', 1, 1)
    !iszerotangent(ΔP) && mul!(M, P, ΔP, -1, 1)
    C = sylvester(P, P, M' - M)
    C .+= ΔP
    ΔA = mul!(ΔA, C, Wᴴ, 1, 1)
    if !iszerotangent(ΔWᴴ)
        PΔWᴴ = P \ ΔWᴴ
        PΔWᴴW = PΔWᴴ * Wᴴ'
        PΔWᴴ = mul!(PΔWᴴ, PΔWᴴW, Wᴴ, -1, 1)
        ΔA .+= PΔWᴴ
    end
    return ΔA
end