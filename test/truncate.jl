using MatrixAlgebraKit
using Test
using TestExtras
using MatrixAlgebraKit: NoTruncation, TruncationIntersection, TruncationKeepAbove,
                        TruncationKeepBelow, TruncationStrategy, findtruncated,
                        findtruncated_sorted

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
    @test trunc.by == abs
    @test trunc.rev == true

    trunc = @constinferred TruncationStrategy(; atol=1e-2, rtol=1e-3, maxrank=10)
    @test trunc isa TruncationIntersection
    @test trunc == truncrank(10) & TruncationKeepAbove(1e-2, 1e-3)
    @test trunc.components[1] == truncrank(10)
    @test trunc.components[2] == TruncationKeepAbove(1e-2, 1e-3)

    values = [1, 0.9, 0.5, -0.3, 0.01]
    @test @constinferred(findtruncated(values, truncrank(2))) == 1:2
    @test @constinferred(findtruncated(values, truncrank(2; rev=false))) == [5, 4]
    @test @constinferred(findtruncated(values, truncrank(2; by=((-) ∘ abs)))) == [5, 4]
    @test @constinferred(findtruncated_sorted(values, truncrank(2))) === 1:2

    values = [1, 0.9, 0.5, -0.3, 0.01]
    for strategy in (TruncationKeepAbove(; atol=0.4, rtol=0),
                     TruncationKeepAbove(0.4, 0))
        @test @constinferred(findtruncated(values, strategy)) == 1:3
        @test @constinferred(findtruncated_sorted(values, strategy)) === 1:3
    end
    for strategy in (TruncationKeepBelow(; atol=0.4, rtol=0),
                     TruncationKeepBelow(0.4, 0))
        @test @constinferred(findtruncated(values, strategy)) == 4:5
        @test @constinferred(findtruncated_sorted(values, strategy)) === 4:5
    end

    values = [0.01, 1, 0.9, -0.3, 0.5]
    for strategy in (TruncationKeepAbove(; atol=0.4, rtol=0),
                     TruncationKeepAbove(; atol=0.4, rtol=0, by=abs),
                     TruncationKeepAbove(0.4, 0),
                     TruncationKeepAbove(; atol=0.2, rtol=0.0, by=identity))
        @test @constinferred(findtruncated(values, strategy)) == [2, 3, 5]
    end
    for strategy in (TruncationKeepAbove(; atol=0.2, rtol=0),
                     TruncationKeepAbove(; atol=0.2, rtol=0, by=abs),
                     TruncationKeepAbove(0.2, 0))
        @test @constinferred(findtruncated(values, strategy)) == [2, 3, 4, 5]
    end
    for strategy in (TruncationKeepBelow(; atol=0.4, rtol=0),
                     TruncationKeepBelow(; atol=0.4, rtol=0, by=abs),
                     TruncationKeepBelow(0.4, 0),
                     TruncationKeepBelow(; atol=0.2, rtol=0.0, by=identity))
        @test @constinferred(findtruncated(values, strategy)) == [1, 4]
    end
    for strategy in (TruncationKeepBelow(; atol=0.2, rtol=0),
                     TruncationKeepBelow(; atol=0.2, rtol=0, by=abs),
                     TruncationKeepBelow(0.2, 0))
        @test @constinferred(findtruncated(values, strategy)) == [1]
    end
    for strategy in (truncerror(; atol=0.2, rtol=0),)
        @test issetequal(@constinferred(findtruncated(values, strategy)), 2:5)
        @test @constinferred(findtruncated_sorted(sort(values; by=abs, rev=true), strategy)) == 1:4
    end
end
