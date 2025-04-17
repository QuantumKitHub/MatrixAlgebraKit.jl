using MatrixAlgebraKit
using Test
using TestExtras
using MatrixAlgebraKit: NoTruncation, TruncationIntersection, TruncationKeepAbove,
                        TruncationStrategy

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
end
