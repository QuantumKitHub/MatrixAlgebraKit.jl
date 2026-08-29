using TestExtras
using GenericSchur

# `gees!` and `geev!` agree on well-scaled matrices, but not on the ordering for `Diagonal`
sorted_vals(v) = sort!(collect(v); by = x -> (real(x), imag(x)))

function test_schur(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "schur $summary_str" begin
        test_schur_full(T, sz; kwargs...)
    end
end

function test_schur_algs(T::Type, sz, algs; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "schur algorithms $summary_str" begin
        test_schur_full_algs(T, sz, algs; kwargs...)
    end
end

function test_schur_full(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "schur_full! $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)
        Tc = complex(eltype(T))

        TA, Z, vals = @testinferred schur_full(A)
        @test eltype(TA) == eltype(Z) == eltype(T)
        @test eltype(vals) == Tc
        @test isisometric(Z)
        @test A * Z ≈ Z * TA
        @test sorted_vals(vals) ≈ sorted_vals(eig_vals(A))
        # a diagonal matrix is already in Schur form and is not reordered
        A isa Diagonal && @test TA ≈ A

        TA2, Z2, vals2 = @testinferred schur_full!(Ac, (TA, Z, vals))
        @test TA2 === TA
        @test Z2 === Z
        @test vals2 === vals
        @test A * Z ≈ Z * TA
    end
end

function test_schur_full_algs(
        T::Type, sz, algs;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "schur_full! algorithm $alg $summary_str" for alg in algs
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)
        Tc = complex(eltype(T))

        TA, Z, vals = @testinferred schur_full(A; alg)
        @test eltype(TA) == eltype(Z) == eltype(T)
        @test eltype(vals) == Tc
        @test isisometric(Z)
        @test A * Z ≈ Z * TA
        @test sorted_vals(vals) ≈ sorted_vals(eig_vals(A))

        TA2, Z2, vals2 = @testinferred schur_full!(Ac, (TA, Z, vals); alg)
        @test TA2 === TA
        @test Z2 === Z
        @test vals2 === vals
        @test A * Z ≈ Z * TA
    end
end
