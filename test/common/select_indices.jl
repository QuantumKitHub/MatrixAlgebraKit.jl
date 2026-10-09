using MatrixAlgebraKit
using MatrixAlgebraKit: select_indices
using Test

# `Bool <: Integer`, so logical masks used to hit the integer-index arithmetic in
# `select_indices` and silently produce wrong indices (e.g. `[1, 0, 1, ...]`).
@testset "select_indices" begin
    r = 1:6
    mask = [true, false, true, false, false, true]
    @test select_indices(r, mask) == [1, 3, 6]
    @test select_indices(r, view(mask, 1:6)) == [1, 3, 6]
    @test select_indices(r, falses(6)) == Int[]
    @test select_indices(r, [1, 3, 6]) == [1, 3, 6]
    @test select_indices(r, 2:4) == 2:4
    @test select_indices(r, :) == r
    @test select_indices(3:2:13, mask) == [3, 7, 13]
    @test_throws BoundsError select_indices(r, [true, false])
end
