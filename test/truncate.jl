using MatrixAlgebraKit
using Test
using TestExtras
using MatrixAlgebraKit: NoTruncation, TruncationIntersection, TruncationByOrder,
    TruncationByValue, TruncationStrategy, findtruncated, findtruncated_svd

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
end
