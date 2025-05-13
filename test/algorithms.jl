using MatrixAlgebraKit
using Test
using TestExtras
using MatrixAlgebraKit: LAPACK_SVDAlgorithm, NoTruncation, PolarViaSVD, TruncatedAlgorithm,
                        default_algorithm, select_algorithm

@testset "default_algorithm" begin
    A = randn(3, 3)
    for f in (svd_compact!, svd_compact, svd_full!, svd_full)
        @test @constinferred(default_algorithm(f, A)) === LAPACK_DivideAndConquer()
    end
    for f in (eig_full!, eig_full, eig_vals!, eig_vals)
        @test @constinferred(default_algorithm(f, A)) === LAPACK_Expert()
    end
    for f in (eigh_full!, eigh_full, eigh_vals!, eigh_vals)
        @test @constinferred(default_algorithm(f, A)) ===
              LAPACK_MultipleRelativelyRobustRepresentations()
    end
    for f in (lq_full!, lq_full, lq_compact!, lq_compact, lq_null!, lq_null)
        @test @constinferred(default_algorithm(f, A)) == LAPACK_HouseholderLQ()
    end
    for f in (left_polar!, left_polar, right_polar!, right_polar)
        @test @constinferred(default_algorithm(f, A)) ==
              PolarViaSVD(LAPACK_DivideAndConquer())
    end
    for f in (qr_full!, qr_full, qr_compact!, qr_compact, qr_null!, qr_null)
        @test @constinferred(default_algorithm(f, A)) == LAPACK_HouseholderQR()
    end
    for f in (schur_full!, schur_full, schur_vals!, schur_vals)
        @test @constinferred(default_algorithm(f, A)) === LAPACK_Expert()
    end

    @test @constinferred(default_algorithm(qr_compact!, A; blocksize=2)) ===
          LAPACK_HouseholderQR(; blocksize=2)
end

@testset "select_algorithm" begin
    A = randn(3, 3)
    for f in (svd_trunc!, svd_trunc)
        @test @constinferred(select_algorithm(f, A)) ===
              TruncatedAlgorithm(LAPACK_DivideAndConquer(), NoTruncation())
    end
    for f in (eig_trunc!, eig_trunc)
        @test @constinferred(select_algorithm(f, A)) ===
              TruncatedAlgorithm(LAPACK_Expert(), NoTruncation())
    end
    for f in (eigh_trunc!, eigh_trunc)
        @test @constinferred(select_algorithm(f, A)) ===
              TruncatedAlgorithm(LAPACK_MultipleRelativelyRobustRepresentations(),
                                 NoTruncation())
    end

    @test @constinferred(select_algorithm(svd_compact!, A)) === LAPACK_DivideAndConquer()
    @test @constinferred(select_algorithm(svd_compact!, A, nothing)) ===
          LAPACK_DivideAndConquer()
    for alg in (:LAPACK_QRIteration, LAPACK_QRIteration, LAPACK_QRIteration())
        @test @constinferred(select_algorithm(svd_compact!, A, $alg)) ===
              LAPACK_QRIteration()
    end
end
