function test_mooncake_lq(
        T::Type, sz;
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "Mooncake lq $summary_str" begin
        test_mooncake_lq_compact(T, sz; kwargs...)
        test_mooncake_lq_full(T, sz; kwargs...)
        test_mooncake_lq_null(T, sz; kwargs...)
    end
end

function remove_lq_gauge_dependence!(ΔQ, A, L, Q)
    m, n = size(A)
    minmn = min(m, n)
    Q₁ = @view Q[1:minmn, :]
    ΔQ₂ = @view ΔQ[(minmn + 1):end, :]
    ΔQ₂Q₁ᴴ = ΔQ₂ * Q₁'
    mul!(ΔQ₂, ΔQ₂Q₁ᴴ, Q₁)
    MatrixAlgebraKit.check_lq_full_cotangents(Q₁, ΔQ₂, ΔQ₂Q₁ᴴ)
    return ΔQ
end

function remove_lq_null_gauge_dependence!(ΔNᴴ, A, Nᴴ)
    _, Q = lq_compact(A)
    ΔNᴴQᴴ = ΔNᴴ * Q'
    return mul!(ΔNᴴ, ΔNᴴQᴴ, Q)
end

function test_mooncake_lq_compact(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "lq_compact" begin
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(lq_compact, A; positive = true)
        LQ = lq_compact(A, alg)
        ΔLQ = Mooncake.randn_tangent(rng, LQ)
        remove_lq_gauge_dependence!(ΔLQ[2], A, LQ...)

        Mooncake.TestUtils.test_rule(
            rng, lq_compact, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔLQ, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, make_input_scratch!, lq_compact!, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔLQ, atol, rtol, is_primitive = false
        )
    end
end

function test_mooncake_lq_full(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "lq_full" begin
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(lq_full, A; positive = true)
        LQ = lq_full(A, alg)
        ΔLQ = Mooncake.randn_tangent(rng, LQ)
        remove_lq_gauge_dependence!(ΔLQ[2], A, LQ...)

        Mooncake.TestUtils.test_rule(
            rng, lq_full, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔLQ, atol, rtol
        )
        Mooncake.TestUtils.test_rule(
            rng, make_input_scratch!, lq_full!, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔLQ, atol, rtol, is_primitive = false
        )
    end
end

function test_mooncake_lq_null(
        T, sz;
        rng = Random.default_rng(), atol::Real = 0, rtol::Real = precision(T)
    )
    return @testset "lq_null" begin
        A = instantiate_matrix(T, sz)
        alg = MatrixAlgebraKit.select_algorithm(lq_null, A; positive = true)
        Nᴴ = lq_null(A, alg)
        ΔNᴴ = Mooncake.randn_tangent(rng, Nᴴ)
        remove_lq_null_gauge_dependence!(ΔNᴴ, A, Nᴴ)

        Mooncake.TestUtils.test_rule(
            rng, lq_null, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔNᴴ, atol, rtol
        )
        Nᴴ, ΔNᴴ = ad_lq_null_setup(A)
        Mooncake.TestUtils.test_rule(
            rng, make_input_scratch!, lq_null!, A, alg;
            mode = Mooncake.ReverseMode, output_tangent = ΔNᴴ, atol, rtol, is_primitive = false
        )
    end
end
