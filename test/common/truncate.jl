using MatrixAlgebraKit
using Test
using TestExtras
using MatrixAlgebraKit: NoTruncation, TruncationIntersection, TruncationUnion,
    TruncationByOrder, TruncationByValue, TruncationMultiplets, TruncationStrategy,
    findtruncated, findtruncated_svd
using LinearAlgebra: Diagonal, norm

@testset "truncate" begin
    trunc = @constinferred TruncationStrategy()
    @test trunc isa NoTruncation

    atol = 1.0e-2
    rtol = 1.0e-3
    maxrank = 10

    trunc = @constinferred TruncationStrategy(; atol, rtol)
    @test trunc isa TruncationByValue
    @test trunc == trunctol(; atol, rtol)
    @test trunc.atol == atol
    @test trunc.rtol == rtol
    @test !trunc.keep_below

    trunc = @constinferred TruncationStrategy(; maxrank)
    @test trunc isa TruncationByOrder
    @test trunc == truncrank(maxrank)
    @test trunc.howmany == maxrank
    @test trunc.by == abs
    @test trunc.rev

    trunc = @constinferred TruncationStrategy(; atol, rtol, maxrank)
    @test trunc isa TruncationIntersection
    @test trunc == trunctol(; atol, rtol) & truncrank(maxrank)

    values = [1, 0.9, 0.5, -0.3, 0.01]
    @test values[@constinferred(findtruncated(values, truncrank(2)))] == values[1:2]
    @test values[@constinferred(findtruncated(values, truncrank(2; rev = false)))] == values[[5, 4]]
    @test values[@constinferred(findtruncated(values, truncrank(2; by = ((-) ∘ abs))))] == values[[5, 4]]
    @test values[@constinferred(findtruncated_svd(values, truncrank(2)))] == values[1:2]

    values = [1, 0.9, 0.5, -0.3, 0.01]
    strategy = trunctol(; atol = 0.4)
    @test values[@constinferred(findtruncated(values, strategy))] == values[1:3]
    @test values[@constinferred(findtruncated_svd(values, strategy))] == values[1:3]
    strategy = trunctol(; atol = 0.4, keep_below = true)
    @test values[@constinferred(findtruncated(values, strategy))] == values[4:5]
    @test values[@constinferred(findtruncated_svd(values, strategy))] == values[4:5]

    values = [0.01, 1, 0.9, -0.3, 0.5]
    for strategy in (trunctol(; atol = 0.4), trunctol(; atol = 0.2, by = identity))
        @test values[@constinferred(findtruncated(values, strategy))] == values[[2, 3, 5]]
    end
    strategy = trunctol(; atol = 0.2)
    @test values[@constinferred(findtruncated(values, strategy))] == values[[2, 3, 4, 5]]

    for strategy in
        (trunctol(; atol = 0.4, keep_below = true), trunctol(; atol = 0.2, by = identity, keep_below = true))
        @test values[@constinferred(findtruncated(values, strategy))] == values[[1, 4]]
    end
    strategy = trunctol(; atol = 0.2, keep_below = true)
    @test values[@constinferred(findtruncated(values, strategy))] == values[[1]]

    strategy = truncfilter(x -> 0.1 < x < 1)
    @test values[@constinferred(findtruncated(values, strategy))] == values[[3, 5]]

    strategy = truncerror(; atol = 0.2, rtol = 0)
    @test issetequal(values[@constinferred(findtruncated(values, strategy))], values[2:5])
    vals_sorted = sort(values; by = abs, rev = true)
    @test vals_sorted[@constinferred(findtruncated_svd(vals_sorted, strategy))] == vals_sorted[1:4]

    # TruncationUnion / minrank
    trunc = @constinferred TruncationStrategy(; minrank = 3)
    @test trunc isa TruncationByOrder
    @test trunc == truncrank(3)

    trunc = @constinferred TruncationStrategy(; atol, minrank = 3)
    @test trunc isa TruncationUnion
    @test trunc == trunctol(; atol) | truncrank(3)

    # | operator
    values2 = [1.0, 0.9, 0.5, 0.3, 0.01]
    # trunctol keeps 1:3 (above 0.4), truncrank(4) keeps 1:4, union keeps 1:4
    strategy = trunctol(; atol = 0.4) | truncrank(4)
    @test @constinferred(findtruncated_svd(values2, strategy)) == 1:4
    # trunctol keeps 1:3, truncrank(2) keeps 1:2, union keeps 1:3
    strategy = trunctol(; atol = 0.4) | truncrank(2)
    @test @constinferred(findtruncated_svd(values2, strategy)) == 1:3

    # notrunc is absorbing for |
    @test (notrunc() | truncrank(3)) isa NoTruncation
    @test (truncrank(3) | notrunc()) isa NoTruncation

    # TruncationUnion flattening
    union1 = truncrank(2) | trunctol(; atol = 0.4)
    union2 = union1 | truncrank(4)
    @test union2 isa TruncationUnion
    @test length(union2.components) == 3
end

@testset "truncmultiplets" begin
    trunc = @constinferred truncmultiplets(truncrank(2))
    @test trunc isa TruncationMultiplets
    @test trunc.trunc == truncrank(2)
    @test trunc.atol == 0 && trunc.rtol == 1.0e-10
    @test trunc.mode === :down
    @test trunc == truncmultiplets(truncrank(2))
    @test trunc != truncmultiplets(truncrank(2); mode = :up)
    @test_throws ArgumentError truncmultiplets(truncrank(2); mode = :sideways)

    # a degenerate group of three values around 0.5, split by truncrank(2) and truncrank(3)
    ϵ = 1.0e-14
    vals_sorted = [1.0, 0.5 + ϵ, 0.5, -0.5 + ϵ, 0.1]
    vals_unsorted = vals_sorted[[3, 5, 1, 4, 2]]
    for howmany in (2, 3)
        down = truncmultiplets(truncrank(howmany))
        up = truncmultiplets(truncrank(howmany); mode = :up)
        @test @constinferred(findtruncated_svd(vals_sorted, down)) == [1]
        @test @constinferred(findtruncated_svd(vals_sorted, up)) == 1:4
        @test sort(vals_unsorted[@constinferred(findtruncated(vals_unsorted, down))]) == [1.0]
        @test sort(vals_unsorted[@constinferred(findtruncated(vals_unsorted, up))]) ==
            sort(vals_sorted[1:4])
    end

    # nothing changes if the cut does not split a degenerate cluster
    for howmany in (0, 1, 4, 5, 6)
        for mode in (:down, :up)
            strategy = truncmultiplets(truncrank(howmany); mode)
            @test findtruncated_svd(vals_sorted, strategy) ==
                findtruncated_svd(vals_sorted, truncrank(howmany))
            @test sort(findtruncated(vals_unsorted, strategy)) ==
                sort(findtruncated(vals_unsorted, truncrank(howmany)))
        end
    end
    @test findtruncated_svd(vals_sorted, truncmultiplets(notrunc())) == 1:5
    @test isempty(findtruncated_svd(Float64[], truncmultiplets(truncrank(2))))

    # without tolerance, only exactly degenerate values are grouped
    @test findtruncated_svd(vals_sorted, truncmultiplets(truncrank(2); rtol = 0)) == 1:2
    @test findtruncated_svd([1.0, 0.5, 0.5, 0.1], truncmultiplets(truncrank(2); rtol = 0)) == [1]

    vals_chain = [1.0, 0.9, 0.8, 0.7, 0.1]
    @test findtruncated_svd(vals_chain, truncmultiplets(truncrank(2); atol = 0.15)) == Int[]
    @test findtruncated_svd(vals_chain, truncmultiplets(truncrank(2); atol = 0.15, mode = :up)) == 1:4
    @test findtruncated_svd(vals_chain, truncmultiplets(truncrank(2); rtol = 0.12)) == Int[]
    @test findtruncated_svd(vals_chain, truncmultiplets(truncrank(2); rtol = 0.12, mode = :up)) == 1:3

    strategy = truncmultiplets(truncrank(3) & trunctol(; atol = 0.2))
    @test findtruncated_svd(vals_sorted, strategy) == [1]
    strategy = truncmultiplets(trunctol(; atol = 0.5) | truncrank(1); mode = :up)
    @test findtruncated_svd(vals_sorted, strategy) == 1:4
    strategy = truncmultiplets(truncfilter(x -> x < 0.5 + ϵ / 2); mode = :up)
    @test sort(findtruncated(vals_unsorted, strategy)) == sort(findtruncated(vals_unsorted, truncfilter(x -> x < 0.9)))
    @test truncmultiplets(truncrank(2)) & trunctol(; atol = 0.2) isa TruncationIntersection

    # exactly degenerate singular values
    for T in (Float64, ComplexF64)
        n = 6
        Q₁ = qr_compact(randn(T, n, n))[1]
        Q₂ = qr_compact(randn(T, n, n))[1]
        Sref = [3.0, 2.0, 2.0, 2.0, 1.0, 0.5]
        A = Q₁ * Diagonal(Sref) * Q₂'
        for howmany in 2:3
            U, S, Vᴴ, ϵ = svd_trunc(A; trunc = truncmultiplets(truncrank(howmany); rtol = 1.0e-8))
            @test diagview(S) ≈ Sref[1:1]
            @test ϵ ≈ norm(Sref[2:end])
            @test U * S * Vᴴ ≈ Sref[1] * Q₁[:, 1] * Q₂[:, 1]'
            U, S, Vᴴ, ϵ = svd_trunc(A; trunc = truncmultiplets(truncrank(howmany); rtol = 1.0e-8, mode = :up))
            @test diagview(S) ≈ Sref[1:4]
            @test ϵ ≈ norm(Sref[5:end])
            @test U * S * Vᴴ ≈ Q₁[:, 1:4] * Diagonal(Sref[1:4]) * Q₂[:, 1:4]'
        end
    end
end
