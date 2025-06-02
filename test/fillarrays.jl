using MatrixAlgebraKit
using LinearAlgebra
using Test
using TestExtras
using FillArrays

@testset "Zeros" begin
    for f in [:eig_full, :eigh_full]
        @eval begin
            A = Zeros(3, 3)
            D, V = @constinferred $f(A)
            @test A * V == D * V
            @test size(D) == size(A)
            @test size(V) == size(A)
            @test iszero(D)
            @test D isa Zeros
            @test V == I
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

    # for f in [:qr_compact, :left_polar]
    #     @eval begin
    #         A = Zeros(4, 3)
    #         Q, R = @constinferred $f(A)
    #         @test A == Q * R
    #         @test size(Q) == (4, 3)
    #         @test size(R) == (3, 3)
    #         @test Q == Matrix(I, (4, 3))
    #         @test Q isa Eye
    #         @test iszero(R)
    #         @test R isa Zeros
    #     end
    # end

    # A = Zeros(4, 3)
    # Q, R = @constinferred qr_full(A)
    # @test A == Q * R
    # @test size(Q) == (4, 4)
    # @test size(R) == (4, 3)
    # @test Q == I
    # @test Q isa Eye
    # @test iszero(R)
    # @test R isa Zeros

    for f in [:lq_compact] # :right_polar]
        @eval begin
            A = Zeros(3, 4)
            L, Q = @constinferred $f(A)
            @test A == L * Q
            @test size(L) == (3, 3)
            @test size(Q) == (3, 4)
            @test iszero(L)
            @test L isa Zeros
            @test Q == Matrix(I, (3, 4))
            @test Q isa Eye
        end
    end

    A = Zeros(3, 4)
    L, Q = @constinferred lq_full(A)
    @test A == L * Q
    @test size(L) == (3, 4)
    @test size(Q) == (4, 4)
    @test iszero(L)
    @test L isa Zeros
    @test Q == I
    @test Q isa Eye

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
    D = @constinferred svd_vals(A)
    @test size(D) == (minimum(size(A)),)
    @test iszero(D)
    @test D isa Zeros
end

@testset "Eye" begin
    for f in [:eig_full, :eigh_full]
        @eval begin
            A = Eye(3, 3)
            D, V = @constinferred $f(A)
            @test A * V == D * V
            @test size(D) == size(A)
            @test size(V) == size(A)
            @test V == I
            @test D isa Eye
            @test V == I
            @test V isa Eye
        end
    end

    for f in [:eig_vals, :eigh_vals]
        @eval begin
            A = Eye(3, 3)
            D = @constinferred $f(A)
            @test size(D) == (size(A, 1),)
            @test all(isone, D)
            @test D isa Ones
        end
    end

    # for f in [:qr_compact, :left_polar]
    #     @eval begin
    #         A = Eye(4, 3)
    #         Q, R = @constinferred $f(A)
    #         @test A == Q * R
    #         @test size(Q) == (4, 3)
    #         @test size(R) == (3, 3)
    #         @test Q == Matrix(I, (4, 3))
    #         @test Q isa Eye
    #         @test R == I
    #         @test R isa Eye
    #     end
    # end

    # A = Eye(4, 3)
    # Q, R = @constinferred qr_full(A)
    # @test A == Q * R
    # @test size(Q) == (4, 4)
    # @test size(R) == (4, 3)
    # @test Q == I
    # @test Q isa Eye
    # @test R == I
    # @test R isa Eye

    for f in [:lq_compact] # :right_polar]
        @eval begin
            A = Eye(3, 4)
            L, Q = @constinferred $f(A)
            @test A == L * Q
            @test size(L) == (3, 3)
            @test size(Q) == (3, 4)
            @test L == I
            @test L isa Eye
            @test Q == Matrix(I, (3, 4))
            @test Q isa Eye
        end
    end

    A = Eye(3, 4)
    L, Q = @constinferred lq_full(A)
    @test A == L * Q
    @test size(L) == (3, 4)
    @test size(Q) == (4, 4)
    @test L == Matrix(I, (3, 4))
    @test L isa Eye
    @test Q == I
    @test Q isa Eye

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

    A = Eye(3, 4)
    D = @constinferred svd_vals(A)
    @test size(D) == (minimum(size(A)),)
    @test all(isone, D)
    @test D isa Ones
end
