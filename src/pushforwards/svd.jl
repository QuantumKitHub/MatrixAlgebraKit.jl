function svd_pushforward!(
        ΔA, A, USVᴴ, ΔUSVᴴ, ind = Colon();
        rank_atol = default_pullback_rank_atol(USVᴴ[2]),
        degeneracy_atol = default_pullback_rank_atol(USVᴴ[2])
    )
    U, Smat, Vᴴ = USVᴴ
    m, n = size(U, 1), size(Vᴴ, 2)
    (m, n) == size(ΔA) || throw(DimensionMismatch("size of ΔA ($(size(ΔA))) does not match size of U*S*Vᴴ ($m, $n)"))
    minmn = min(m, n)
    S = diagview(Smat)
    ΔU, ΔS, ΔVᴴ = ΔUSVᴴ
    r = svd_rank(S; rank_atol)

    U₁ = view(U, :, 1:r)
    S₁ = view(S, 1:r)
    V₁ᴴ = view(Vᴴ, 1:r, :)

    # compact region
    V₁ = adjoint(V₁ᴴ)
    ΔAV₁ = ΔA * V₁
    UᴴΔAV₁ = U₁' * ΔAV₁
    if !iszerotangent(ΔS)
        zero!(ΔS) # make off-diagonal entries zero in case of full ΔS (svd_full!)
        ΔS₁ = view(diagview(ΔS), 1:r)
        ΔS₁ .= real.(diagview(UᴴΔAV₁))
    end
    if !iszerotangent(ΔU) || !iszerotangent(ΔVᴴ)
        hUᴴΔAV₁ = inv_safe.(transpose(S₁) .- S₁, degeneracy_atol) .* project_hermitian(UᴴΔAV₁)
        aUᴴΔAV₁ = inv_safe.(transpose(S₁) .+ S₁, degeneracy_atol) .* project_antihermitian(UᴴΔAV₁)
        if !iszerotangent(ΔU)
            ΔU₁ = view(ΔU, :, 1:r)
            K̇ = hUᴴΔAV₁ + aUᴴΔAV₁
            mul!(ΔU₁, U₁, K̇)
            if m > r
                ΔAV₁ = mul!(ΔAV₁, U₁, UᴴΔAV₁, -1, 1)
                ΔU₁ .+= ΔAV₁ ./ transpose(S₁)
            end
            if size(U, 2) > r # these columns of U are undetermined, but U' * U̇ should be antihermitian
                U₂ = view(U, :, (r + 1):size(U, 2))
                ΔU₁ᴴU₂ = ΔU₁' * U₂
                ΔU₂ = view(ΔU, :, (r + 1):size(U, 2))
                mul!(ΔU₂, U₁, ΔU₁ᴴU₂, -1, 0)
            end
        end
        if !iszerotangent(ΔVᴴ)
            ΔV₁ᴴ = view(ΔVᴴ, 1:r, :)
            Ṁ = hUᴴΔAV₁ - aUᴴΔAV₁
            mul!(ΔV₁ᴴ, Ṁ', V₁ᴴ)
            if n > r
                UᴴΔA₁ = U₁' * ΔA
                UᴴΔA₁ = mul!(UᴴΔA₁, UᴴΔAV₁, V₁ᴴ, -1, 1)
                ΔV₁ᴴ .+= S₁ .\ UᴴΔA₁
            end
            if size(Vᴴ, 1) > r # these rows of Vᴴ are undetermined, but V * V̇ should be antihermitian
                V₂ᴴ = view(Vᴴ, (r + 1):size(Vᴴ, 1), :)
                V₂ᴴΔV₁ = V₂ᴴ * ΔV₁ᴴ'
                ΔV₂ᴴ = view(ΔVᴴ, (r + 1):size(Vᴴ, 1), :)
                mul!(ΔV₂ᴴ, V₂ᴴΔV₁, V₁ᴴ, -1, 0)
            end
        end
        if eltype(U) <: Complex && !iszerotangent(ΔU) && !iszerotangent(ΔVᴴ) # fix gauge for `gaugefix!` compatibility
            ΔU₁ = view(ΔU, :, 1:r)
            ΔV₁ᴴ = view(ΔVᴴ, 1:r, :)
            _, I = findmax(abs, U₁; dims = 1)
            infinitesimal_phases = imag.(ΔU₁[I] .* inv_safe.(U₁[I]))
            ΔU₁ .-= im .* U₁ .* infinitesimal_phases
            ΔV₁ᴴ .+= im .* transpose(infinitesimal_phases) .* V₁ᴴ
        end
    end
    return (ΔU, ΔS, ΔVᴴ)
end

# TODO
#=function svd_trunc_pushforward!(ΔA, A, USVᴴ, ΔUSVᴴ, ind; rank_atol = default_pullback_rank_atol(A), kwargs...)
end=#

function svd_vals_pushforward!(
        ΔA, A, USVᴴ, ΔS, ind = Colon();
        rank_atol::Real = default_pullback_rank_atol(USVᴴ[2]),
        degeneracy_atol::Real = default_pullback_rank_atol(USVᴴ[2])
    )
    ΔUSVᴴ = (nothing, diagonal(ΔS), nothing)
    return svd_pushforward!(ΔA, A, USVᴴ, ΔUSVᴴ, ind; rank_atol, degeneracy_atol)
end
