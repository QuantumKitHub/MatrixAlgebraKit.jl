using MatrixAlgebraKit
using Test
using TestExtras
using MatrixAlgebraKit: NoTruncation, TruncationIntersection, TruncationKeepAbove,
                        TruncationKeepBelow, TruncationStrategy, findtruncated

@testset "truncate" begin
    trunc = @constinferred TruncationStrategy()
    @test trunc isa NoTruncation

    trunc = @constinferred TruncationStrategy(; atol=1e-2, rtol=1e-3)
    @test trunc isa TruncationKeepAbove
    @test trunc == TruncationKeepAbove(1e-2, 1e-3)
    @test trunc.atol == 1e-2
    @test trunc.rtol == 1e-3

    trunc = @constinferred TruncationStrategy(; maxrank=10)
    @test trunc isa TruncationKeepSorted
    @test trunc == truncrank(10)
    @test trunc.howmany == 10
    @test trunc.sortby == abs
    @test trunc.rev == true

    trunc = @constinferred TruncationStrategy(; atol=1e-2, rtol=1e-3, maxrank=10)
    @test trunc isa TruncationIntersection
    @test trunc == truncrank(10) & TruncationKeepAbove(1e-2, 1e-3)
    @test trunc.components[1] == truncrank(10)
    @test trunc.components[2] == TruncationKeepAbove(1e-2, 1e-3)

    values = [1, 0.9, 0.5, 0.3, 0.01]
    @test @constinferred(findtruncated(values, truncrank(2))) == 1:2
    @test @constinferred(findtruncated(values, truncrank(2; rev=false))) == [5, 4]
    @test @constinferred(findtruncated(values, truncrank(2; by=-))) == [5, 4]

    values = [1, 0.9, 0.5, 0.3, 0.01]
    @test @constinferred(findtruncated(values, TruncationKeepAbove(0.4, 0.0))) == 1:3
    @test @constinferred(findtruncated(values, TruncationKeepBelow(0.4, 0.0))) == 4:5

    values = [0.01, 1, 0.9, 0.3, 0.5]
    @test @constinferred(findtruncated(values, TruncationKeepAbove(0.4, 0.0))) == [2, 3, 5]
    @test @constinferred(findtruncated(values, TruncationKeepBelow(0.4, 0.0))) == [1, 4]
end
