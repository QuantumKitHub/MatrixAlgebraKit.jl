using TestExtras

function test_schur(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "schur $summary_str" begin
        test_schur_full(T, sz; kwargs...)
    end
end

function test_schur_full(
        T::Type, sz;
        test_blocksize = true,
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "eig_full! $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)
        Tc = isa(A, Diagonal) ? eltype(T) : complex(eltype(T))

        TA, Z, vals = @testinferred schur_full(A)
        @test eltype(TA) == eltype(Z) == eltype(T)
        @test eltype(vals) == Tc
        @test isisometric(Z)
        @test A * Z ≈ Z * TA

        TA2, Z2, vals2 = @testinferred schur_full!(Ac, (TA, Z, vals))
        @test TA2 === TA
        @test Z2 === Z
        @test vals2 === vals
        @test A * Z ≈ Z * TA

        valsc = @testinferred schur_vals(A)
        @test eltype(valsc) == Tc
        @test valsc ≈ eig_vals(A)
    end
end
