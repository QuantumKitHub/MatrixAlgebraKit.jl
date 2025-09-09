using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using Mooncake, Mooncake.TestUtils, ChainRulesCore
using Mooncake: rrule!!
using MatrixAlgebraKit: diagview, TruncatedAlgorithm, PolarViaSVD
using CUDA
using LinearAlgebra: UpperTriangular, Diagonal, Hermitian, mul!

function remove_svdgauge_depence!(ΔU, ΔVᴴ, U, S, Vᴴ;
                                  degeneracy_atol=MatrixAlgebraKit.default_pullback_gaugetol(S))
    gaugepart = U' * ΔU + Vᴴ * ΔVᴴ'
    gaugepart = (gaugepart - gaugepart') / 2
    gaugepart[abs.(transpose(diagview(S)) .- diagview(S)) .>= degeneracy_atol] .= 0
    mul!(ΔU, U, gaugepart, -1, 1)
    return ΔU, ΔVᴴ
end
function remove_eiggauge_depence!(ΔV, D, V;
                                  degeneracy_atol=MatrixAlgebraKit.default_pullback_gaugetol(S))
    gaugepart = V' * ΔV
    gaugepart[abs.(transpose(diagview(D)) .- diagview(D)) .>= degeneracy_atol] .= 0
    mul!(ΔV, V / (V' * V), gaugepart, -1, 1)
    return ΔV
end
function remove_eighgauge_depence!(ΔV, D, V;
                                   degeneracy_atol=MatrixAlgebraKit.default_pullback_gaugetol(S))
    gaugepart = V' * ΔV
    gaugepart = (gaugepart - gaugepart') / 2
    gaugepart[abs.(transpose(diagview(D)) .- diagview(D)) .>= degeneracy_atol] .= 0
    mul!(ΔV, V / (V' * V), gaugepart, -1, 1)
    return ΔV
end

precision(::Type{<:Union{Float32,Complex{Float32}}}) = sqrt(eps(Float32))
precision(::Type{<:Union{Float64,Complex{Float64}}}) = sqrt(eps(Float64))

for f in
    (:qr_compact, :qr_full, :qr_null, :lq_compact, :lq_full, :lq_null, #:eig_full, :eigh_full, 
     :eigh_full, :svd_compact, :svd_trunc, :left_polar, :right_polar)
    copy_f = Symbol(:copy_, f)
    f! = Symbol(f, '!')
    @eval begin
        function $copy_f(input, alg)
            if $f === eigh_full
                input = (input + input') / 2
            end
            return $f(input, alg)
        end
        function ChainRulesCore.rrule(::typeof($copy_f), input, alg)
            output = MatrixAlgebraKit.initialize_output($f!, input, alg)
            if $f === eigh_full
                input = (input + input') / 2
            else
                input = copy(input)
            end
            output, pb = ChainRulesCore.rrule($f!, input, output, alg)
            return output, x -> (NoTangent(), pb(x)[2], NoTangent())
        end
        Mooncake.@from_chainrules Mooncake.DefaultCtx Tuple{typeof($copy_f), AbstractMatrix, MatrixAlgebraKit.AbstractAlgorithm} false Mooncake.ReverseMode
    end
end

for f in (:eig_full,)#:eigh_full)
    copy_f = Symbol(:copy_, f)
    f! = Symbol(f, '!')
    @eval begin
        function $copy_f(input, alg)
            if $f === eigh_full
                input = (input + input') / 2
            end
            return $f(input, alg)
        end
    end
end

@timedtestset "QR AD Rules with eltype $T" for T in (Float64, Float32) #, ComplexF64)
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        # qr_compact
        atol  = rtol = m * n * precision(T)
        A     = CUDA.randn(rng, T, m, n)
        minmn = min(m, n)
        alg   = CUSOLVER_HouseholderQR(; positive=true)
        @testset for f in (copy_qr_compact, copy_qr_null, copy_qr_full)
            Mooncake.TestUtils.test_rule(rng, f, A, alg; mode=Mooncake.ReverseMode)
        end
        
        # rank-deficient A
        r = minmn - 5
        A = CUDA.randn(rng, T, m, r) * CUDA.randn(rng, T, r, n)
        Q, R = qr_compact(A, alg)
        Mooncake.TestUtils.test_rule(rng, copy_qr_compact, A, alg; mode=Mooncake.ReverseMode)
    end
end

@timedtestset "LQ AD Rules with eltype $T" for T in (Float64, Float32) #, ComplexF64)
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        # lq_compact
        atol  = rtol = m * n * precision(T)
        A     = CUDA.randn(rng, T, m, n)
        minmn = min(m, n)
        alg   = CUSOLVER_HouseholderLQ(; positive=true)
        @testset for f in (copy_lq_compact, copy_lq_null, copy_lq_full)
            Mooncake.TestUtils.test_rule(rng, f, A, alg; mode=Mooncake.ReverseMode)
        end
        # rank-deficient A
        r = minmn - 5
        A = CUDA.randn(rng, T, m, r) * CUDA.randn(rng, T, r, n)
        Mooncake.TestUtils.test_rule(rng, copy_lq_compact, A, alg; mode=Mooncake.ReverseMode)
    end
end

@timedtestset "EIG AD Rules with eltype $T" for T in (Float64, Float32, ComplexF64)
    rng  = StableRNG(12345)
    m    = 19
    atol = rtol = m * m * precision(T)
    A    = CUDA.randn(rng, T, m, m)
    @testset for alg in (CUSOLVER_Simple(), CUSOLVER_Expert())
        Mooncake.TestUtils.test_rule(rng, copy_eig_full, A, alg; mode=Mooncake.ReverseMode)
    end
end
@timedtestset "EIGH AD Rules with eltype $T" for T in (Float64, Float32) #, ComplexF64)
    rng  = StableRNG(12345)
    m    = 19
    atol = rtol = m * m * precision(T)
    A    = CUDA.randn(rng, T, m, m)
    A    = A + A'
    @testset for alg in (CUSOLVER_QRIteration(), CUSOLVER_DivideAndConquer(), CUSOLVER_Bisection(),
                CUSOLVER_MultipleRelativelyRobustRepresentations())
        # copy_eigh_full includes a projector onto the Hermitian part of the matrix
        Mooncake.TestUtils.test_rule(rng, copy_eigh_full, A, alg; mode=Mooncake.ReverseMode)
    end
end

@timedtestset "SVD AD Rules with eltype $T" for T in (Float64, Float32) #, ComplexF64)
    rng = StableRNG(12345)
    m   = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol  = rtol = m * n * precision(T)
        A     = CUDA.randn(rng, T, m, n)
        minmn = min(m, n)
        U, S, Vᴴ = svd_compact(A)
        @testset for alg in (CUSOLVER_QRIteration(), CUSOLVER_DivideAndConquer())
            Mooncake.TestUtils.test_rule(rng, copy_svd_compact, A, alg; mode=Mooncake.ReverseMode)
            @testset for r in 1:4:minmn
                truncalg = TruncatedAlgorithm(alg, truncrank(r))
                Mooncake.TestUtils.test_rule(rng, copy_svd_trunc, A, truncalg; mode=Mooncake.ReverseMode)
            end
            truncalg = TruncatedAlgorithm(alg, trunctol(S[1, 1] / 2))
            r        = findlast(>=(S[1, 1] / 2), diagview(S))
            truncalg = TruncatedAlgorithm(alg, truncrank(r))
            Mooncake.TestUtils.test_rule(rng, copy_svd_trunc, A, truncalg; mode=Mooncake.ReverseMode)
        end
    end
end

@timedtestset "Polar AD Rules with eltype $T" for T in (Float64, Float32) #, ComplexF64)
    rng = StableRNG(12345)
    m   = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        A = CUDA.randn(rng, T, m, n)
        @testset for alg in PolarViaSVD.((CUSOLVER_QRIteration(), CUSOLVER_DivideAndConquer()))
            m >= n &&
                Mooncake.TestUtils.test_rule(rng, copy_left_polar, A, alg; mode=Mooncake.ReverseMode)

            m <= n &&
                Mooncake.TestUtils.test_rule(rng, copy_right_polar, A, alg; mode=Mooncake.ReverseMode)
        end
    end
end

#=
@timedtestset "Orth and null with eltype $T" for T in (Float64, ComplexF64, Float32)
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        config = Zygote.ZygoteRuleConfig()
        test_rrule(config, left_orth, A;
                   atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)
        test_rrule(config, left_orth, A; fkwargs=(; kind=:qr),
                   atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)
        m >= n &&
            test_rrule(config, left_orth, A; fkwargs=(; kind=:polar),
                       atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)

        ΔN = left_orth(A; kind=:qr)[1] * randn(rng, T, min(m, n), m - min(m, n))
        test_rrule(config, left_null, A; fkwargs=(; kind=:qr), output_tangent=ΔN,
                   atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)

        test_rrule(config, right_orth, A;
                   atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)
        test_rrule(config, right_orth, A; fkwargs=(; kind=:lq),
                   atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)
        m <= n &&
            test_rrule(config, right_orth, A; fkwargs=(; kind=:polar),
                       atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)

        ΔNᴴ = randn(rng, T, n - min(m, n), min(m, n)) * right_orth(A; kind=:lq)[2]
        test_rrule(config, right_null, A; fkwargs=(; kind=:lq), output_tangent=ΔNᴴ,
                   atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)
    end
end
=#
