using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using LinearAlgebra: LinearAlgebra, I, mul!
using MatrixAlgebraKit: TruncationKeepAbove, TruncationKeepBelow
using MatrixAlgebraKit: LAPACK_SVDAlgorithm, check_input, copy_input, default_svd_algorithm,
                        initialize_output

# Used to test non-AbstractMatrix codepaths.
struct LinearMap{P<:AbstractMatrix}
    parent::P
end
Base.parent(A::LinearMap) = getfield(A, :parent)
function Base.copy!(dest::LinearMap, src::LinearMap)
    copy!(parent(dest), parent(src))
    return dest
end
function LinearAlgebra.mul!(C::LinearMap, A::LinearMap, B::LinearMap)
    mul!(parent(C), parent(A), parent(B))
    return C
end

function MatrixAlgebraKit.copy_input(::typeof(qr_compact), A::LinearMap)
    return LinearMap(copy_input(qr_compact, parent(A)))
end
function MatrixAlgebraKit.copy_input(::typeof(lq_compact), A::LinearMap)
    return LinearMap(copy_input(lq_compact, parent(A)))
end
function MatrixAlgebraKit.initialize_output(::typeof(left_orth!), A::LinearMap)
    return LinearMap.(initialize_output(left_orth!, parent(A)))
end
function MatrixAlgebraKit.initialize_output(::typeof(right_orth!), A::LinearMap)
    return LinearMap.(initialize_output(right_orth!, parent(A)))
end
function MatrixAlgebraKit.check_input(::typeof(left_orth!), A::LinearMap, VC)
    return check_input(left_orth!, parent(A), parent.(VC))
end
function MatrixAlgebraKit.check_input(::typeof(right_orth!), A::LinearMap, VC)
    return check_input(right_orth!, parent(A), parent.(VC))
end
function MatrixAlgebraKit.default_svd_algorithm(::Type{LinearMap{A}}; kwargs...) where {A}
    return default_svd_algorithm(A; kwargs...)
end
function MatrixAlgebraKit.initialize_output(::typeof(svd_compact!), A::LinearMap,
                                            alg::LAPACK_SVDAlgorithm)
    return LinearMap.(initialize_output(svd_compact!, parent(A), alg))
end
function MatrixAlgebraKit.svd_compact!(A::LinearMap, USVᴴ, alg::LAPACK_SVDAlgorithm)
    return LinearMap.(svd_compact!(parent(A), parent.(USVᴴ), alg))
end

@testset "left_orth and left_null for T = $T" for T in (Float32, Float64, ComplexF32,
                                                        ComplexF64)
    rng = StableRNG(123)
    m = 54
    for n in (37, m, 63)
        minmn = min(m, n)
        A = randn(rng, T, m, n)
        V, C = @constinferred left_orth(A)
        N = @constinferred left_null(A)
        @test V isa Matrix{T} && size(V) == (m, minmn)
        @test C isa Matrix{T} && size(C) == (minmn, n)
        @test N isa Matrix{T} && size(N) == (m, m - minmn)
        @test V * C ≈ A
        @test V' * V ≈ I
        @test LinearAlgebra.norm(A' * N) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test N' * N ≈ I
        @test V * V' + N * N' ≈ I

        M = LinearMap(A)
        VM, CM = @constinferred left_orth(M; kind=:svd)
        @test parent(VM) * parent(CM) ≈ A

        if m > n
            nullity = 5
            V, C = @constinferred left_orth(A)
            N = @constinferred left_null(A; trunc=(; maxnullity=nullity))
            @test V isa Matrix{T} && size(V) == (m, minmn)
            @test C isa Matrix{T} && size(C) == (minmn, n)
            @test N isa Matrix{T} && size(N) == (m, nullity)
            @test V * C ≈ A
            @test V' * V ≈ I
            @test LinearAlgebra.norm(A' * N) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
            @test N' * N ≈ I
        end

        for alg_qr in ((; positive=true), (; positive=false), LAPACK_HouseholderQR())
            V, C = @constinferred left_orth(A; alg_qr)
            N = @constinferred left_null(A; alg_qr)
            @test V isa Matrix{T} && size(V) == (m, minmn)
            @test C isa Matrix{T} && size(C) == (minmn, n)
            @test N isa Matrix{T} && size(N) == (m, m - minmn)
            @test V * C ≈ A
            @test V' * V ≈ I
            @test LinearAlgebra.norm(A' * N) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
            @test N' * N ≈ I
            @test V * V' + N * N' ≈ I
        end

        Ac = similar(A)
        V2, C2 = @constinferred left_orth!(copy!(Ac, A), (V, C))
        N2 = @constinferred left_null!(copy!(Ac, A), N)
        @test V2 === V
        @test C2 === C
        @test N2 === N
        @test V2 * C2 ≈ A
        @test V2' * V2 ≈ I
        @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test N2' * N2 ≈ I
        @test V2 * V2' + N2 * N2' ≈ I

        atol = eps(real(T))
        V2, C2 = @constinferred left_orth!(copy!(Ac, A), (V, C); trunc=(; atol=atol))
        N2 = @constinferred left_null!(copy!(Ac, A), N; trunc=(; atol=atol))
        @test V2 !== V
        @test C2 !== C
        @test N2 !== C
        @test V2 * C2 ≈ A
        @test V2' * V2 ≈ I
        @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test N2' * N2 ≈ I
        @test V2 * V2' + N2 * N2' ≈ I

        rtol = eps(real(T))
        for (trunc_orth, trunc_null) in (((; rtol=rtol), (; rtol=rtol)),
                                         (TruncationKeepAbove(0, rtol), TruncationKeepBelow(0, rtol)))
            V2, C2 = @constinferred left_orth!(copy!(Ac, A), (V, C); trunc=trunc_orth)
            N2 = @constinferred left_null!(copy!(Ac, A), N; trunc=trunc_null)
            @test V2 !== V
            @test C2 !== C
            @test N2 !== C
            @test V2 * C2 ≈ A
            @test V2' * V2 ≈ I
            @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
            @test N2' * N2 ≈ I
            @test V2 * V2' + N2 * N2' ≈ I
        end

        for kind in (:qr, :polar, :svd) # explicit kind kwarg
            m < n && kind == :polar && continue
            V2, C2 = @constinferred left_orth!(copy!(Ac, A), (V, C); kind=kind)
            @test V2 === V
            @test C2 === C
            @test V2 * C2 ≈ A
            @test V2' * V2 ≈ I
            if kind != :polar
                N2 = @constinferred left_null!(copy!(Ac, A), N; kind=kind)
                @test N2 === N
                @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test N2' * N2 ≈ I
                @test V2 * V2' + N2 * N2' ≈ I
            end

            # with kind and tol kwargs
            if kind == :svd
                V2, C2 = @constinferred left_orth!(copy!(Ac, A), (V, C); kind=kind,
                                                   trunc=(; atol=atol))
                N2 = @constinferred left_null!(copy!(Ac, A), N; kind=kind,
                                               trunc=(; atol=atol))
                @test V2 !== V
                @test C2 !== C
                @test N2 !== C
                @test V2 * C2 ≈ A
                @test V2' * V2 ≈ I
                @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test N2' * N2 ≈ I
                @test V2 * V2' + N2 * N2' ≈ I

                V2, C2 = @constinferred left_orth!(copy!(Ac, A), (V, C); kind=kind,
                                                   trunc=(; rtol=rtol))
                N2 = @constinferred left_null!(copy!(Ac, A), N; kind=kind,
                                               trunc=(; rtol=rtol))
                @test V2 !== V
                @test C2 !== C
                @test N2 !== C
                @test V2 * C2 ≈ A
                @test V2' * V2 ≈ I
                @test LinearAlgebra.norm(A' * N2) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test N2' * N2 ≈ I
                @test V2 * V2' + N2 * N2' ≈ I
            else
                @test_throws ArgumentError left_orth!(copy!(Ac, A), (V, C); kind=kind,
                                                      trunc=(; atol=atol))
                @test_throws ArgumentError left_orth!(copy!(Ac, A), (V, C); kind=kind,
                                                      trunc=(; rtol=rtol))
                @test_throws ArgumentError left_null!(copy!(Ac, A), N; kind=kind,
                                                      trunc=(; atol=atol))
                @test_throws ArgumentError left_null!(copy!(Ac, A), N; kind=kind,
                                                      trunc=(; rtol=rtol))
            end
        end
    end
end

@testset "right_orth and right_null for T = $T" for T in (Float32, Float64, ComplexF32,
                                                          ComplexF64)
    rng = StableRNG(123)
    m = 54
    for n in (37, m, 63)
        minmn = min(m, n)
        A = randn(rng, T, m, n)
        C, Vᴴ = @constinferred right_orth(A)
        Nᴴ = @constinferred right_null(A)
        @test C isa Matrix{T} && size(C) == (m, minmn)
        @test Vᴴ isa Matrix{T} && size(Vᴴ) == (minmn, n)
        @test Nᴴ isa Matrix{T} && size(Nᴴ) == (n - minmn, n)
        @test C * Vᴴ ≈ A
        @test Vᴴ * Vᴴ' ≈ I
        @test LinearAlgebra.norm(A * adjoint(Nᴴ)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test Nᴴ * Nᴴ' ≈ I
        @test Vᴴ' * Vᴴ + Nᴴ' * Nᴴ ≈ I

        M = LinearMap(A)
        CM, VMᴴ = @constinferred right_orth(M; kind=:svd)
        @test parent(CM) * parent(VMᴴ) ≈ A

        Ac = similar(A)
        C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ))
        Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ)
        @test C2 === C
        @test Vᴴ2 === Vᴴ
        @test Nᴴ2 === Nᴴ
        @test C2 * Vᴴ2 ≈ A
        @test Vᴴ2 * Vᴴ2' ≈ I
        @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test Nᴴ2 * Nᴴ2' ≈ I
        @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I

        atol = eps(real(T))
        C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ); trunc=(; atol=atol))
        Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ; trunc=(; atol=atol))
        @test C2 !== C
        @test Vᴴ2 !== Vᴴ
        @test Nᴴ2 !== Nᴴ
        @test C2 * Vᴴ2 ≈ A
        @test Vᴴ2 * Vᴴ2' ≈ I
        @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test Nᴴ2 * Nᴴ2' ≈ I
        @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I

        rtol = eps(real(T))
        C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ); trunc=(; rtol=rtol))
        Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ; trunc=(; rtol=rtol))
        @test C2 !== C
        @test Vᴴ2 !== Vᴴ
        @test Nᴴ2 !== Nᴴ
        @test C2 * Vᴴ2 ≈ A
        @test Vᴴ2 * Vᴴ2' ≈ I
        @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test Nᴴ2 * Nᴴ2' ≈ I
        @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I

        for kind in (:lq, :polar, :svd)
            n < m && kind == :polar && continue
            C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ); kind=kind)
            @test C2 === C
            @test Vᴴ2 === Vᴴ
            @test C2 * Vᴴ2 ≈ A
            @test Vᴴ2 * Vᴴ2' ≈ I
            if kind != :polar
                Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ; kind=kind)
                @test Nᴴ2 === Nᴴ
                @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test Nᴴ2 * Nᴴ2' ≈ I
                @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I
            end

            if kind == :svd
                C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ); kind=kind,
                                                     trunc=(; atol=atol))
                Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ; kind=kind,
                                                 trunc=(; atol=atol))
                @test C2 !== C
                @test Vᴴ2 !== Vᴴ
                @test Nᴴ2 !== Nᴴ
                @test C2 * Vᴴ2 ≈ A
                @test Vᴴ2 * Vᴴ2' ≈ I
                @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test Nᴴ2 * Nᴴ2' ≈ I
                @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I

                C2, Vᴴ2 = @constinferred right_orth!(copy!(Ac, A), (C, Vᴴ); kind=kind,
                                                     trunc=(; rtol=rtol))
                Nᴴ2 = @constinferred right_null!(copy!(Ac, A), Nᴴ; kind=kind,
                                                 trunc=(; rtol=rtol))
                @test C2 !== C
                @test Vᴴ2 !== Vᴴ
                @test Nᴴ2 !== Nᴴ
                @test C2 * Vᴴ2 ≈ A
                @test Vᴴ2 * Vᴴ2' ≈ I
                @test LinearAlgebra.norm(A * adjoint(Nᴴ2)) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
                @test Nᴴ2 * Nᴴ2' ≈ I
                @test Vᴴ2' * Vᴴ2 + Nᴴ2' * Nᴴ2 ≈ I
            else
                @test_throws ArgumentError right_orth!(copy!(Ac, A), (C, Vᴴ); kind=kind,
                                                       trunc=(; atol=atol))
                @test_throws ArgumentError right_orth!(copy!(Ac, A), (C, Vᴴ); kind=kind,
                                                       trunc=(; rtol=rtol))
                @test_throws ArgumentError right_null!(copy!(Ac, A), Nᴴ; kind=kind,
                                                       trunc=(; atol=atol))
                @test_throws ArgumentError right_null!(copy!(Ac, A), Nᴴ; kind=kind,
                                                       trunc=(; rtol=rtol))
            end
        end
    end
end
