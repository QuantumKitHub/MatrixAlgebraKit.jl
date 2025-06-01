using MatrixAlgebraKit
using Test
using TestExtras
using FillArrays

@testset "Eye" begin
    for f in [:eig_full!,
              :eigh_full!,
     # TODO: Reenable once https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/32
     # is merged.
     # :qr_compact!,
     # :qr_full!,
              :left_polar!,
              :lq_compact!,
              :lq_full!,
              :right_polar!,
              :svd_compact!,
              :svd_full!]
        @eval begin
            A = Eye(3)
            F = @constinferred $f(A)
            @test A === prod(F)
            @test all(x -> x === A, F)

            A = Zeros(3, 3)
            F = @constinferred $f(A)
            @test A === prod(F)
            @test all(x -> x === A, F)
        end
    end
end
