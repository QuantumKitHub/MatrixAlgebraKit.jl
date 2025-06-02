using MatrixAlgebraKit
using LinearAlgebra
using Test
using TestExtras
using FillArrays
using FillArrays: SquareEye

@testset "Zeros" begin
    for f in [:eig_full, :eigh_full]
        @eval begin
            A = Zeros(3, 3)
            D, V = @constinferred $f(A)
            @test A * V == V * D
            @test size(D) == size(A)
            @test size(V) == size(A)
            @test iszero(D)
            @test D isa Zeros
            @test V == I
            @test V isa Eye
        end
    end

    for f in [:eig_trunc, :eigh_trunc]
        @eval begin
            A = Zeros(3, 3)
            D, V = @constinferred $f(A; trunc=(; maxrank=2))
            @test A * V == V * D
            @test size(D) == (2, 2)
            @test size(V) == (3, 2)
            @test D == Zeros(2, 2)
            @test D isa Zeros
            @test V == Eye(3, 2)
            @test V isa Eye
        end
    end

    for f in [:eig_vals, :eigh_vals]
        @eval begin
            A = Zeros(3, 3)
            D = @constinferred $f(A)
            @test size(D) == (size(A, 1),)
            @test iszero(D)
            @test D isa Zeros
        end
    end

    # A = Zeros(4, 3)
    # Q, R = @constinferred qr_compact(A)
    # @test Q * R == A
    # @test size(Q) == (4, 3)
    # @test size(R) == (3, 3)
    # @test Q == Matrix(I, (4, 3))
    # @test Q isa Eye
    # @test iszero(R)
    # @test R isa Zeros

    # A = Zeros(4, 3)
    # Q, R = @constinferred qr_full(A)
    # @test Q * R == A
    # @test size(Q) == (4, 4)
    # @test size(R) == (4, 3)
    # @test Q == I
    # @test Q isa Eye
    # @test iszero(R)
    # @test R isa Zeros

    # A = Zeros(4, 3)
    # Q, R = @constinferred left_polar(A)
    # @test Q * R == A
    # @test size(Q) == (4, 3)
    # @test size(R) == (3, 3)
    # @test Q == Matrix(I, (4, 3))
    # @test Q isa Eye
    # @test iszero(R)
    # @test R isa Zeros

    A = Zeros(3, 4)
    L, Q = @constinferred lq_compact(A)
    @test L * Q == A
    @test size(L) == (3, 3)
    @test size(Q) == (3, 4)
    @test iszero(L)
    @test L isa Zeros
    @test Q == Matrix(I, (3, 4))
    @test Q isa Eye

    A = Zeros(3, 4)
    L, Q = @constinferred lq_compact(A)
    @test L * Q == A
    @test L == Zeros(3, 3)
    @test L isa Zeros
    @test Q == Eye(3, 4)
    @test Q isa Eye

    A = Zeros(3, 4)
    L, Q = @constinferred lq_full(A)
    @test L * Q == A
    @test size(L) == (3, 4)
    @test size(Q) == (4, 4)
    @test iszero(L)
    @test L isa Zeros
    @test Q == I
    @test Q isa Eye

    A = Zeros(3, 4)
    L, Q = @constinferred lq_full(A)
    @test L * Q == A
    @test L === A
    @test Q == Eye(4)
    @test Q isa Eye

    # A = Zeros(3, 4)
    # L, Q = @constinferred right_polar(A)
    # @test L * Q == A
    # @test size(L) == (3, 3)
    # @test size(Q) == (3, 4)
    # @test iszero(L)
    # @test L isa Zeros
    # @test Q == Matrix(I, (3, 4))
    # @test Q isa Eye

    A = Zeros(3, 4)
    U, S, V = @constinferred svd_compact(A)
    @test U * S * V == A
    @test size(U) == (3, 3)
    @test size(S) == (3, 3)
    @test size(V) == (3, 4)
    @test iszero(S)
    @test S isa Zeros
    @test U == I
    @test U isa Eye
    @test V == Matrix(I, (3, 4))
    @test V isa Eye

    A = Zeros(3, 4)
    U, S, V = @constinferred svd_full(A)
    @test U * S * V == A
    @test size(U) == (3, 3)
    @test size(S) == (3, 4)
    @test size(V) == (4, 4)
    @test iszero(S)
    @test S isa Zeros
    @test U == I
    @test U isa Eye
    @test V == I
    @test V isa Eye

    A = Zeros(3, 4)
    U, S, V = @constinferred svd_trunc(A; trunc=(; maxrank=2))
    @test U * S * V == Eye(3, 2) * Zeros(2, 2) * Eye(2, 4)
    @test size(U) == (3, 2)
    @test size(S) == (2, 2)
    @test size(V) == (2, 4)
    @test S == Zeros(2, 2)
    @test S isa Zeros
    @test U == Eye(3, 2)
    @test U isa Eye
    @test V == Eye(2, 4)
    @test V isa Eye

    A = Zeros(3, 4)
    D = @constinferred svd_vals(A)
    @test size(D) == (minimum(size(A)),)
    @test iszero(D)
    @test D isa Zeros
end

@testset "Eye" begin
    for f in [:eig_full, :eigh_full]
        @eval begin
            for A in (Eye(3), Eye(3, 3))
                local D, V = @constinferred $f(A)
                @test A * V == V * D
                @test size(D) == size(A)
                @test size(V) == size(A)
                @test V == I
                @test typeof(D) === typeof(A)
                @test V == I
                @test typeof(V) === typeof(A)
            end
        end
    end

    for f in [:eig_trunc, :eigh_trunc]
        @eval begin
            for A in (Eye(3), Eye(3, 3))
                local D, V = @constinferred $f(A; trunc=(; maxrank=2))
                @test A * V == V * D
                @test size(D) == (2, 2)
                @test size(V) == (3, 2)
                @test D == Eye(2, 2)
                @test D isa SquareEye
                @test V == Eye(3, 2)
                @test V isa Eye
            end
        end
    end

    for f in [:eig_vals, :eigh_vals]
        @eval begin
            for A in (Eye(3), Eye(3, 3))
                local D = @constinferred $f(A)
                @test size(D) == (size(A, 1),)
                @test all(isone, D)
                @test D isa Ones
            end
        end
    end

    # A = Eye(4, 3)
    # Q, R = @constinferred qr_compact(A)
    # @test Q * R == A
    # @test size(Q) == (4, 3)
    # @test size(R) == (3, 3)
    # @test Q == Matrix(I, (4, 3))
    # @test Q isa Eye
    # @test R == I
    # @test R isa Eye

    # A = Eye(3)
    # Q, R = @constinferred qr_compact(A)
    # @test Q * R == A
    # @test Q === A
    # @test R === A

    # A = Eye(4, 3)
    # Q, R = @constinferred qr_full(A)
    # @test Q * R == A
    # @test size(Q) == (4, 4)
    # @test size(R) == (4, 3)
    # @test Q == I
    # @test Q isa Eye
    # @test R == I
    # @test R isa Eye

    # A = Eye(3)
    # Q, R = @constinferred qr_full(A)
    # @test Q * R == A
    # @test Q === A
    # @test R === A

    # A = Eye(4, 3)
    # Q, R = @constinferred left_polar(A)
    # @test Q * R == A
    # @test size(Q) == (4, 3)
    # @test size(R) == (3, 3)
    # @test Q == Matrix(I, (4, 3))
    # @test Q isa Eye
    # @test R == I
    # @test R isa Eye

    # A = Eye(3)
    # Q, R = @constinferred left_polar(A)
    # @test Q * R == A
    # @test Q === A
    # @test R === A

    A = Eye(3, 4)
    L, Q = @constinferred lq_compact(A)
    @test L * Q == A
    @test size(L) == (3, 3)
    @test size(Q) == (3, 4)
    @test L == I
    @test L isa Eye
    @test Q == Matrix(I, (3, 4))
    @test Q isa Eye

    A = Eye(3)
    L, Q = @constinferred lq_compact(A)
    @test L * Q == A
    @test L === A
    @test Q === A

    A = Eye(3, 4)
    L, Q = @constinferred lq_full(A)
    @test L * Q == A
    @test size(L) == (3, 4)
    @test size(Q) == (4, 4)
    @test L == Matrix(I, (3, 4))
    @test L isa Eye
    @test Q == I
    @test Q isa Eye

    A = Eye(3)
    L, Q = @constinferred lq_full(A)
    @test L * Q == A
    @test L === A
    @test Q === A

    # A = Eye(3, 4)
    # L, Q = @constinferred right_polar(A)
    # @test L * Q == A
    # @test size(L) == (3, 3)
    # @test size(Q) == (3, 4)
    # @test L == I
    # @test L isa Eye
    # @test Q == Matrix(I, (3, 4))
    # @test Q isa Eye

    # A = Eye(3)
    # L, Q = @constinferred right_polar(A)
    # @test L * Q == A
    # @test L === A
    # @test Q === A

    A = Eye(3, 4)
    U, S, V = @constinferred svd_compact(A)
    @test U * S * V == A
    @test size(U) == (3, 3)
    @test size(S) == (3, 3)
    @test size(V) == (3, 4)
    @test S == I
    @test S isa Eye
    @test U == I
    @test U isa Eye
    @test V == Matrix(I, (3, 4))
    @test V isa Eye

    A = Eye(3)
    U, S, V = @constinferred svd_compact(A)
    @test U * S * V == A
    @test U === A
    @test S === A
    @test V === A

    A = Eye(3, 4)
    U, S, V = @constinferred svd_full(A)
    @test U * S * V == A
    @test size(U) == (3, 3)
    @test size(S) == (3, 4)
    @test size(V) == (4, 4)
    @test S == Matrix(I, (3, 4))
    @test S isa Eye
    @test U == I
    @test U isa Eye
    @test V == I
    @test V isa Eye

    A = Eye(3)
    U, S, V = @constinferred svd_full(A)
    @test U * S * V == A
    @test U === A
    @test S === A
    @test V === A

    A = Eye(3, 4)
    U, S, V = @constinferred svd_trunc(A; trunc=(; maxrank=2))
    @test U * S * V == Eye(3, 2) * Eye(2, 2) * Eye(2, 4)
    @test size(U) == (3, 2)
    @test size(S) == (2, 2)
    @test size(V) == (2, 4)
    @test S == Eye(2, 2)
    @test S isa Eye
    @test U == Eye(3, 2)
    @test U isa Eye
    @test V == Eye(2, 4)
    @test V isa Eye

    A = Eye(3, 4)
    D = @constinferred svd_vals(A)
    @test size(D) == (minimum(size(A)),)
    @test all(isone, D)
    @test D isa Ones
end
