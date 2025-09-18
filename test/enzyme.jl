using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using ChainRulesCore
using Enzyme, EnzymeTestUtils
using MatrixAlgebraKit: diagview, TruncatedAlgorithm, PolarViaSVD
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
    gaugepart[abs.(transpose(D.diag) .- D.diag) .>= degeneracy_atol] .= 0
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
    (:qr_compact, :qr_null, :lq_compact, :lq_full, :lq_null,
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
        Enzyme.@import_rrule(typeof($copy_f), Any, Any)
    end
end
#=
for f in
    (:eig_full,)
    copy_f = Symbol(:copy_, f)
    f! = Symbol(f, '!')
    @eval begin
        function $copy_f(input, alg)
            if $f === eigh_full
                input = (input + input') / 2
            end
            return $f(input, alg)
        end
        function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                              func::Const{typeof($copy_f)},
                                              ::Type{RT},
                                              input::Annotation{<:AbstractMatrix},
                                              alg,
                                             ) where {RT}
            output = MatrixAlgebraKit.initialize_output($f!, input.val, alg.val)
            if $f === eigh_full
                input.val .= (input.val .+ input.val') ./ 2
            end
            output = $f!(input.val, output, alg.val)
            primal = EnzymeRules.needs_primal(config) ? output : nothing
            d_output = (Diagonal(zeros(eltype(output[1]), size(output[1], 1))), zeros(eltype(output[2]), size(output[2])))
            shadow = EnzymeRules.needs_shadow(config) ? d_output : nothing
            # form cache if needed
            return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
        end
        function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
                                     func::Const{typeof($copy_f)},
                                     dret::Type{RT},
                                     cache,
                                     input::Annotation{<:AbstractMatrix},
                                     alg
                                    ) where {RT}
            output = MatrixAlgebraKit.initialize_output($f!, input.val, alg.val)
            new_input = $f === eigh_full ? (input.val + input.val') / 2 : copy(input.val)
            #reverse(::EnzymeCore.Const{typeof(MatrixAlgebraKit.eig_full!)}, ::EnzymeCore.Duplicated{Matrix{Float64}}, ::EnzymeCore.Duplicated{Tuple{LinearAlgebra.Diagonal{ComplexF64, Vector{ComplexF64}}, Matrix{ComplexF64}}}, ::EnzymeCore.Const{MatrixAlgebraKit.LAPACK_Simple{@NamedTuple{}}})
            input.val .= new_input
            output, pb = Enzyme.reverse(Const($f!), input, Enzyme.Duplicated(output, (Diagonal(zeros(eltype(output[1]), length(output[1].diag))), zeros(eltype(output[2]), size(output[2])))), alg)
            return output, x -> (nothing, pb(x)[2], nothing)
        end
    end
end
=#
#=
@timedtestset "QR AD Rules with eltype $T" for T in (Float64, ComplexF64, Float32)
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        # qr_compact
        atol  = rtol = m * n * precision(T)
        A     = randn(rng, T, m, n)
        minmn = min(m, n)
        alg   = LAPACK_HouseholderQR(; positive=true)
        #=Q, R  = copy_qr_compact(A, alg)
        ΔQ    = randn(rng, T, m, minmn)
        ΔR    = randn(rng, T, minmn, n)
        ΔR2   = UpperTriangular(randn(rng, T, minmn, minmn))
        ΔN    = Q * randn(rng, T, minmn, max(0, m - minmn))
        test_reverse(copy_qr_compact, A, alg)
        test_reverse(copy_qr_null, A, alg)=#
        # qr_full
        Q, R = qr_full(A, alg)
        Q1   = view(Q, 1:m, 1:minmn)
        ΔQ   = randn(rng, T, m, m)
        ΔQ2  = view(ΔQ, :, (minmn + 1):m)
        mul!(ΔQ2, Q1, Q1' * ΔQ2)
        ΔR   = randn(rng, T, m, n)
        @testset "reverse: RT $RT, TA $TA TQR $TQR" for RT  in (Active,), TA in (Duplicated,), TQR in (Duplicated,)
            test_reverse(qr_full!, RT, (A, TA), ((Q, R), TQR); atol=precision(T), rtol=precision(T), fkwargs=(alg=alg,), output_tangent=(ΔQ, ΔR))
        end

        # rank-deficient A
        #=r = minmn - 5
        A = randn(rng, T, m, r) * randn(rng, T, r, n)
        Q, R = qr_compact(A, alg)
        ΔQ = randn(rng, T, m, minmn)
        Q1 = view(Q, 1:m, 1:r)
        Q2 = view(Q, 1:m, (r + 1):minmn)
        ΔQ2 = view(ΔQ, 1:m, (r + 1):minmn)
        ΔQ2 .= 0
        ΔR = randn(rng, T, minmn, n)
        view(ΔR, (r + 1):minmn, :) .= 0
        test_reverse(copy_qr_compact, A, alg)=#
    end
end
=#
#=
@timedtestset "LQ AD Rules with eltype $T" for T in (Float64, ComplexF64, Float32)
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        # lq_compact
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        minmn = min(m, n)
        alg = LAPACK_HouseholderLQ(; positive=true)
        L, Q = copy_lq_compact(A, alg)
        ΔL = randn(rng, T, m, minmn)
        ΔQ = randn(rng, T, minmn, n)
        ΔNᴴ = randn(rng, T, max(0, n - minmn), minmn) * Q
        test_reverse(copy_lq_compact, A)
        test_reverse(copy_lq_null, A)
        #=config = Zygote.ZygoteRuleConfig()
        test_reverse(config, lq_compact, A;
                   fkwargs=(; positive=true), output_tangent=(ΔL, ΔQ),
                   atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)
        test_reverse(config, first ∘ lq_compact, A;
                   fkwargs=(; positive=true), output_tangent=ΔL,
                   atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)

        test_reverse(config, last ∘ lq_compact, A;
                   fkwargs=(; positive=true), output_tangent=ΔQ,
                   atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)
        test_reverse(config, lq_null, A;
                   fkwargs=(; positive=true), output_tangent=ΔNᴴ,
                   atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)=#
        # lq_full
        L, Q = copy_lq_full(A, alg)
        Q1 = view(Q, 1:minmn, 1:n)
        ΔQ = randn(rng, T, n, n)
        ΔQ2 = view(ΔQ, (minmn + 1):n, 1:n)
        mul!(ΔQ2, ΔQ2 * Q1', Q1)
        ΔL = randn(rng, T, m, n)
        test_reverse(copy_lq_full, A)
        #=config = Zygote.ZygoteRuleConfig()
        test_reverse(config, lq_full, A;
                   fkwargs=(; positive=true), output_tangent=(ΔL, ΔQ),
                   atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)
        if m < n
            Nᴴ, null_pb = Zygote.pullback(lq_null, A, alg)
            @test_logs (:warn,) null_pb(randn(rng, T, max(0, n - minmn), n))
            _, full_pb = Zygote.pullback(lq_full, A, alg)
            @test_logs (:warn,) full_pb((randn(rng, T, m, n), randn(rng, T, n, n)))
        end=#
        # rank-deficient A
        r = minmn - 5
        A = randn(rng, T, m, r) * randn(rng, T, r, n)
        L, Q = lq_compact(A, alg)
        ΔL = randn(rng, T, m, minmn)
        ΔQ = randn(rng, T, minmn, n)
        Q1 = view(Q, 1:r, 1:n)
        Q2 = view(Q, (r + 1):minmn, 1:n)
        ΔQ2 = view(ΔQ, (r + 1):minmn, 1:n)
        ΔQ2 .= 0
        view(ΔL, :, (r + 1):minmn) .= 0
        test_reverse(copy_lq_compact, A)
        #=config = Zygote.ZygoteRuleConfig()
        test_reverse(config, lq_compact, A;
                   fkwargs=(; positive=true), output_tangent=(ΔL, ΔQ),
                   atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)=#
    end
end
=#
@timedtestset "EIG AD Rules with eltype $T" for T in (Float64,)# ComplexF64, Float32)
    rng = StableRNG(12345)
    m = 19
    atol = rtol = m * m * precision(T)
    A = randn(rng, T, m, m)
    D, V = eig_full(A)
    ΔV = randn(rng, complex(T), m, m)
    ΔV = remove_eiggauge_depence!(ΔV, D, V; degeneracy_atol=atol)
    ΔD = randn(rng, complex(T), m, m)
    ΔD2 = Diagonal(randn(rng, complex(T), m))
    @testset for alg in (LAPACK_Simple(), LAPACK_Expert())
        @testset for RT in (Const,), TA in (Duplicated,), TDV in (Duplicated,)
            test_forward(eig_full!, RT, (copy(A), TA), ((copy(D), copy(V)), TDV); fkwargs=(alg=alg,))
        end
        @testset for RT in (Duplicated,), TA in (Duplicated,)
            test_reverse(eig_full, RT, (copy(A), TA); fkwargs=(alg=alg,), atol=precision(T), rtol=precision(T), output_tangent=(copy(ΔD2), copy(ΔV)))
            #test_reverse(copy_eig_full, RT, (copy(A), TA), ((copy(D), copy(V)), TDV); fkwargs=(alg=alg,), atol=precision(T), rtol=precision(T), output_tangent=(copy(ΔD2), copy(ΔV)))
        end
    end
end
#=@timedtestset "EIGH AD Rules with eltype $T" for T in (Float64,)# ComplexF64, Float32)
    rng  = StableRNG(12345)
    m    = 19
    atol = rtol = m * m * precision(T)
    A    = randn(rng, T, m, m)
    A    = A + A'
    D, V = eigh_full(A)
    @testset for alg in (LAPACK_QRIteration(),
                         LAPACK_DivideAndConquer(),
                         LAPACK_Bisection(),
                         LAPACK_MultipleRelativelyRobustRepresentations())
        @testset "forward: RT $RT, TA $TA TDV $TDV" for RT in (Const,), TA in (Duplicated,), TDV in (Duplicated,)
            test_forward(eigh_full!, RT, (A, TA), ((D, V), TDV); fkwargs=(alg=alg,))
        end
        @testset "reverse: RT $RT, TA $TA TDV $TDV" for RT  in (Active,), TA in (Duplicated,), TDV in (Duplicated,)
            test_reverse(eigh_full!, RT, (A, TA), ((D, V), TDV); atol=precision(T), rtol=precision(T), fkwargs=(alg=alg,))
        end
    end
end

@timedtestset "SVD AD Rules with eltype $T" for T in (Float64, ComplexF64, Float32)
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        minmn = min(m, n)
        U, S, Vᴴ = svd_full(A)
        @testset for alg in (LAPACK_QRIteration(),
                             LAPACK_DivideAndConquer(),
                             LAPACK_Bisection(),
                             LAPACK_Jacobi(),)
            @testset "forward: RT $RT, TA $TA TUSVᴴ $TUSVᴴ" for RT in (Const,), TA in (Duplicated,), TUSVᴴ in (Duplicated,)
                test_forward(svd_full!, RT, (A, TA), ((U, S, Vᴴ), TUSVᴴ); atol=precision(T), rtol=precision(T), fkwargs=(alg=alg,))
            end
            @testset "reverse: RT $RT, TA $TA TUSVᴴ $TUSVᴴ" for RT  in (Active,), TA in (Duplicated,), TUSVᴴ in (Duplicated,)
                test_reverse(svd_full!, RT, (A, TA), ((U, S, Vᴴ), TUSVᴴ); atol=precision(T), rtol=precision(T), fkwargs=(alg=alg,))
            end
        end
        #=ΔU = randn(rng, T, m, minmn)
        ΔS = randn(rng, real(T), minmn, minmn)
        ΔS2 = Diagonal(randn(rng, real(T), minmn))
        ΔVᴴ = randn(rng, T, minmn, n)
        ΔU, ΔVᴴ = remove_svdgauge_depence!(ΔU, ΔVᴴ, U, S, Vᴴ; degeneracy_atol=atol)
        for alg in (LAPACK_QRIteration(), LAPACK_DivideAndConquer())
            test_reverse(copy_svd_compact, A)
            test_reverse(copy_svd_compact, A)
            for r in 1:4:minmn
                truncalg = TruncatedAlgorithm(alg, truncrank(r))
                test_reverse(copy_svd_trunc, A, truncalg)
            end
            truncalg = TruncatedAlgorithm(alg, trunctol(S[1, 1] / 2))
            r = findlast(>=(S[1, 1] / 2), diagview(S))
            test_reverse(copy_svd_trunc, A, truncalg)
        end=#
    end
end
=#
#=@timedtestset "Polar AD Rules with eltype $T" for T in (Float64, ComplexF64, Float32)
    rng = StableRNG(12345)
    m = 19
    @testset "size ($m, $n)" for n in (17, m, 23)
        atol = rtol = m * n * precision(T)
        A = randn(rng, T, m, n)
        for alg in PolarViaSVD.((LAPACK_QRIteration(), LAPACK_DivideAndConquer()))
            m >= n &&
                test_reverse(copy_left_polar, A, alg)
            m <= n &&
                test_reverse(copy_right_polar, A, alg)
        end
        # Zygote part
        #=config = Zygote.ZygoteRuleConfig()
        m >= n && test_reverse(config, left_polar, A;
                             atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)
        m <= n && test_reverse(config, right_polar, A;
                             atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)
                             =#
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
        test_reverse(config, left_orth, A;
                   atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)
        test_reverse(config, left_orth, A; fkwargs=(; kind=:qr),
                   atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)
        m >= n &&
            test_reverse(config, left_orth, A; fkwargs=(; kind=:polar),
                       atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)

        ΔN = left_orth(A; kind=:qr)[1] * randn(rng, T, min(m, n), m - min(m, n))
        test_reverse(config, left_null, A; fkwargs=(; kind=:qr), output_tangent=ΔN,
                   atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)

        test_reverse(config, right_orth, A;
                   atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)
        test_reverse(config, right_orth, A; fkwargs=(; kind=:lq),
                   atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)
        m <= n &&
            test_reverse(config, right_orth, A; fkwargs=(; kind=:polar),
                       atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)

        ΔNᴴ = randn(rng, T, n - min(m, n), min(m, n)) * right_orth(A; kind=:lq)[2]
        test_reverse(config, right_null, A; fkwargs=(; kind=:lq), output_tangent=ΔNᴴ,
                   atol=atol, rtol=rtol, rrule_f=rrule_via_ad, check_inferred=false)
    end
end
=#
=#
