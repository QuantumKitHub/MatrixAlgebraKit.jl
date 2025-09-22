using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using ChainRulesCore
using Enzyme, EnzymeTestUtils
using MatrixAlgebraKit: diagview, TruncatedAlgorithm, PolarViaSVD
using LinearAlgebra: UpperTriangular, Diagonal, Hermitian, mul!

function remove_svdgauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ;
                                  degeneracy_atol=MatrixAlgebraKit.default_pullback_gaugetol(S))
    gaugepart  = U' * ΔU
    gaugepart += Vᴴ * ΔVᴴ'
    gaugepart  = (gaugepart - gaugepart') / 2
    gaugepart[abs.(transpose(diagview(S)) .- diagview(S)) .>= degeneracy_atol] .= 0
    mul!(ΔU, U, gaugepart, -1, 1)
    return ΔU, ΔVᴴ
end
function remove_eiggauge_dependence!(ΔV, D, V;
                                  degeneracy_atol=MatrixAlgebraKit.default_pullback_gaugetol(D))
    gaugepart = V' * ΔV
    gaugepart[abs.(transpose(D.diag) .- D.diag) .>= degeneracy_atol] .= 0
    mul!(ΔV, V / (V' * V), gaugepart, -1, 1)
    return ΔV
end
function remove_eighgauge_dependence!(ΔV, D, V;
                                   degeneracy_atol=MatrixAlgebraKit.default_pullback_gaugetol(D))
    gaugepart = V' * ΔV
    gaugepart = (gaugepart - gaugepart') / 2
    gaugepart[abs.(transpose(diagview(D)) .- diagview(D)) .>= degeneracy_atol] .= 0
    mul!(ΔV, V / (V' * V), gaugepart, -1, 1)
    return ΔV
end

precision(::Type{<:Union{Float32,Complex{Float32}}}) = 10*sqrt(eps(Float32))
precision(::Type{<:Union{Float64,Complex{Float64}}}) = sqrt(eps(Float64))
#=
@timedtestset "QR AD Rules with eltype $T" for T in (Float64, Float32,) #ComplexF64)
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        A     = randn(rng, T, m, n)
        atol  = rtol = m * n * precision(T)
        minmn = min(m, n)
        @testset for alg in (LAPACK_HouseholderQR(),
                             LAPACK_HouseholderQR(; positive=true),
                            )
            #=@testset "forward: RT $RT, TA $TA" for RT in (Const,Duplicated,DuplicatedNoNeed), TA in (Duplicated,)
                test_forward(qr_full, RT, (A, TA); fkwargs=(alg=alg,), atol=precision(T), rtol=precision(T))
                test_forward(qr_null, RT, (A, TA); fkwargs=(alg=alg,), atol=precision(T), rtol=precision(T))
                test_forward(qr_compact, RT, (A, TA); fkwargs=(alg=alg,), atol=precision(T), rtol=precision(T))
            end=# # TODO
            @testset "reverse: RT $RT, TA $TA" for RT  in (Duplicated,), TA in (Duplicated,)
                @testset "qr_compact" begin
                    test_reverse(qr_compact, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,))
                end
                @testset "qr_null" begin
                    Q, R  = qr_compact(A, alg)
                    ΔN    = Q * randn(rng, T, minmn, max(0, m - minmn))
                    test_reverse(qr_null, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,), output_tangent=ΔN)
                end
                @testset "qr_full" begin
                    Q, R = qr_full(A, alg)
                    Q1   = view(Q, 1:m, 1:minmn)
                    ΔQ   = randn(rng, T, m, m)
                    ΔQ2  = view(ΔQ, :, (minmn + 1):m)
                    mul!(ΔQ2, Q1, Q1' * ΔQ2)
                    ΔR   = randn(rng, T, m, n)
                    test_reverse(qr_full, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,), output_tangent=(ΔQ, ΔR))
                end
                @testset "qr_compact - rank-deficient A" begin
                    r = minmn - 5
                    Ard = randn(rng, T, m, r) * randn(rng, T, r, n)
                    Q, R = qr_compact(Ard, alg)
                    ΔQ = randn(rng, T, m, minmn)
                    Q1 = view(Q, 1:m, 1:r)
                    Q2 = view(Q, 1:m, (r + 1):minmn)
                    ΔQ2 = view(ΔQ, 1:m, (r + 1):minmn)
                    ΔQ2 .= 0
                    ΔR = randn(rng, T, minmn, n)
                    view(ΔR, (r + 1):minmn, :) .= 0
                    test_reverse(qr_compact, RT, (Ard, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,), output_tangent=(ΔQ, ΔR))
                end
            end
        end
    end
end


@timedtestset "LQ AD Rules with eltype $T" for T in (Float64, Float32,)# ComplexF64, )
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol  = rtol = m * n * precision(T)
        minmn = min(m, n)
        A     = randn(rng, T, m, n)
        @testset for alg in (LAPACK_HouseholderLQ(),
                             LAPACK_HouseholderLQ(; positive=true),
                            )
            #=@testset "forward: RT $RT, TA $TA" for RT in (Const,Duplicated,DuplicatedNoNeed), TA in (Duplicated,)
                test_forward(lq_full, RT, (A, TA); fkwargs=(alg=alg,), atol=precision(T), rtol=precision(T))
                test_forward(lq_null, RT, (A, TA); fkwargs=(alg=alg,), atol=precision(T), rtol=precision(T))
                test_forward(lq_compact, RT, (A, TA); fkwargs=(alg=alg,), atol=precision(T), rtol=precision(T))
            end=# # TODO
            @testset "reverse: RT $RT, TA $TA" for RT  in (Duplicated,), TA in (Duplicated,)
                @testset "lq_compact" begin
                    test_reverse(lq_compact, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,))
                end
                @testset "lq_null" begin
                    L, Q = lq_compact(A, alg)
                    ΔL   = randn(rng, T, m, minmn)
                    ΔQ   = randn(rng, T, minmn, n)
                    ΔNᴴ  = randn(rng, T, max(0, n - minmn), minmn) * Q
                    test_reverse(lq_null, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,), output_tangent=ΔNᴴ)
                end
                @testset "lq_full" begin
                    L, Q = lq_full(A, alg)
                    Q1 = view(Q, 1:minmn, 1:n)
                    ΔQ = randn(rng, T, n, n)
                    ΔQ2 = view(ΔQ, (minmn + 1):n, 1:n)
                    mul!(ΔQ2, ΔQ2 * Q1', Q1)
                    ΔL = randn(rng, T, m, n)
                    test_reverse(lq_full, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,), output_tangent = (ΔL, ΔQ))
                end
                @testset "lq_compact -- rank-deficient A" begin
                    r = minmn - 5
                    Ard = randn(rng, T, m, r) * randn(rng, T, r, n)
                    L, Q = lq_compact(Ard, alg)
                    ΔL = randn(rng, T, m, minmn)
                    ΔQ = randn(rng, T, minmn, n)
                    Q1 = view(Q, 1:r, 1:n)
                    Q2 = view(Q, (r + 1):minmn, 1:n)
                    ΔQ2 = view(ΔQ, (r + 1):minmn, 1:n)
                    ΔQ2 .= 0
                    view(ΔL, :, (r + 1):minmn) .= 0
                    test_reverse(lq_compact, RT, (Ard, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,), output_tangent = (ΔL, ΔQ))
                end
            end
        end
    end
end

@timedtestset "EIG AD Rules with eltype $T" for T in (Float64, Float32,) # ComplexF64)
    rng  = StableRNG(12345)
    m    = 19
    atol = rtol = m * m * precision(T)
    A    = randn(rng, T, m, m)
    D, V = eig_full(A)
    ΔV   = randn(rng, complex(T), m, m)
    ΔV   = remove_eiggauge_dependence!(ΔV, D, V; degeneracy_atol=atol)
    ΔD   = randn(rng, complex(T), m, m)
    ΔD2  = Diagonal(randn(rng, complex(T), m))
    @testset for alg in (LAPACK_Simple(), LAPACK_Expert())
        @testset for RT in (Const,Duplicated,DuplicatedNoNeed), TA in (Const,Duplicated,)
            test_forward(eig_full, RT, (copy(A), TA); fkwargs=(alg=alg,), atol=atol, rtol=rtol)
        end
        @testset for RT in (Duplicated,), TA in (Duplicated,)
            test_reverse(eig_full, RT, (copy(A), TA); fkwargs=(alg=alg,), atol=atol, rtol=rtol, output_tangent=(copy(ΔD2), copy(ΔV)))
        end
    end
end

#=
@timedtestset "EIGH AD Rules with eltype $T" for T in (Float64,)# ComplexF64, Float32)
    rng  = StableRNG(12345)
    m    = 19
    atol = rtol = m * m * precision(T)
    A    = randn(rng, T, m, m)
    A    = A + A'
    D, V = eigh_full(A)
    D2   = Diagonal(D)
    ΔV   = randn(rng, T, m, m)
    ΔV   = remove_eighgauge_dependence!(ΔV, D, V; degeneracy_atol=atol)
    ΔD   = randn(rng, real(T), m, m)
    ΔD2  = Diagonal(randn(rng, real(T), m))
    @testset for alg in (LAPACK_QRIteration(),
                         LAPACK_DivideAndConquer(),
                         LAPACK_Bisection(),
                         LAPACK_MultipleRelativelyRobustRepresentations())
        #=@testset "forward: RT $RT, TA $TA" for RT in (Const,Duplicated,DuplicatedNoNeed), TA in (Const,Duplicated,)
            RT <: Union{Duplicated, DuplicatedNoNeed} && TA == Duplicated && continue #broken :(
            test_forward(eigh_full, RT, (copy(A), TA); fkwargs=(alg=alg,))
        end=#
        @testset "reverse: RT $RT, TA $TA" for RT  in (Duplicated,), TA in (Duplicated,)
            test_reverse(eigh_full, RT, (copy(A), TA); fkwargs=(alg=alg,), atol=atol, rtol=rtol, output_tangent=(copy(ΔD2), copy(ΔV)))
        end
    end
end
=#
=#
@timedtestset "SVD AD Rules with eltype $T" for T in (Float64,)# Float32,) # ComplexF64)
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        minmn = min(m, n)
        @testset for alg in (LAPACK_QRIteration(),
                             LAPACK_DivideAndConquer(),
                            )
            isa(alg, LAPACK_Jacobi) && m < n && continue
            @testset "forward: RT $RT, TA $TA" for RT in (Const,Duplicated,DuplicatedNoNeed), TA in (Const,Duplicated,)
                test_forward(svd_compact, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,))
                #test_forward(svd_full, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,))
            end
            @testset "reverse: RT $RT, TA $TA" for RT  in (Duplicated,), TA in (Duplicated,)
                @testset "svd_compact" begin
                    U, S, Vᴴ = svd_compact(A)
                    ΔU  = randn(rng, T, m, minmn)
                    ΔS  = Diagonal(randn(rng, real(T), minmn))
                    ΔVᴴ = randn(rng, T, minmn, n)
                    ΔU, ΔVᴴ = remove_svdgauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ; degeneracy_atol=atol)
                    test_reverse(svd_compact, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,), output_tangent=(ΔU, ΔS, ΔVᴴ))
                end
                @testset "svd_full" begin
                    U, S, Vᴴ = svd_compact(A)
                    ΔU  = randn(rng, T, m, m)
                    ΔS  = randn(rng, real(T), m, n)
                    ΔVᴴ = randn(rng, T, n, n)
                    ΔU, ΔVᴴ = remove_svdgauge_dependence!(view(ΔU, :, 1:minmn), view(ΔVᴴ, 1:minmn, :), view(U, :, 1:minmn), view(S, 1:minmn, 1:minmn), view(Vᴴ, 1:minmn, :); degeneracy_atol=atol)
                    test_reverse(svd_full, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,), output_tangent=(ΔU, ΔS, ΔVᴴ))
                end
            end
        end
    end
end

@timedtestset "Polar AD Rules with eltype $T" for T in (Float64, Float32, ComplexF64)
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        @testset for alg in PolarViaSVD.((LAPACK_QRIteration(), LAPACK_DivideAndConquer()))
            @testset "reverse: RT $RT, TA $TA" for RT  in (Duplicated,), TA in (Duplicated,)
                m >= n &&
                    test_reverse(left_polar, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,))
                m <= n &&
                    test_reverse(right_polar, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(alg=alg,))
            end
        end
    end
end

@timedtestset "Orth and null with eltype $T" for T in (Float64,Float32,ComplexF64)
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        @testset "reverse: RT $RT, TA $TA" for RT  in (Duplicated,), TA in (Duplicated,)
            @testset "left_orth" begin
                @testset for kind in (:polar, :qr) 
                    n > m && kind == :polar && continue
                    test_reverse(left_orth, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(kind=kind,))
                end
            end
            @testset "right_orth" begin
                @testset for kind in (:polar, :lq) 
                    n < m && kind == :polar && continue
                    test_reverse(right_orth, RT, (A, TA); atol=atol, rtol=rtol, fkwargs=(kind=kind,))
                end
            end
            @testset "left_null" begin
                ΔN = left_orth(A; kind=:qr)[1] * randn(rng, T, min(m, n), m - min(m, n))
                test_reverse(left_null, RT, (A, TA); fkwargs=(; kind=:qr), output_tangent=ΔN, atol=atol, rtol=rtol)
            end
            @testset "right_null" begin
                ΔNᴴ = randn(rng, T, n - min(m, n), min(m, n)) * right_orth(A; kind=:lq)[2]
                test_reverse(right_null, RT, (A, TA); fkwargs=(; kind=:lq), output_tangent=ΔNᴴ, atol=atol, rtol=rtol)
            end
        end
    end
end
