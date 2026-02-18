using TestExtras
using GenericSchur

function test_schur(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "schur $summary_str" begin
        test_schur_full(T, sz; kwargs...)
        test_schur_vals(T, sz; kwargs...)
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
    end
end

function test_schur_vals(
        T::Type, sz;
        atol::Real = 0, rtol::Real = precision(T),
        kwargs...
    )
    summary_str = testargs_summary(T, sz)
    return @testset "schur_vals! $summary_str" begin
        A = instantiate_matrix(T, sz)
        Ac = deepcopy(A)
        Tc = isa(A, Diagonal) ? eltype(T) : complex(eltype(T))

        valsc = @testinferred schur_vals(A)
        @test eltype(valsc) == Tc
        @test valsc ≈ eig_vals(A)

        valsc = similar(A, Tc, size(A, 1))
        valsc = @testinferred schur_vals!(Ac, valsc)
        @test eltype(valsc) == Tc
        @test valsc ≈ eig_vals(A)
    end
end
