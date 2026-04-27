using PrecompileTools: @compile_workload

@compile_workload begin
    truncation_strategies = [
        truncrank(2), trunctol(; atol = 1.0e-2), truncrank(2) & trunctol(; atol = 1.0e-2), truncerror(; atol = 1.0e-2) | truncfilter(x -> abs(x) < 1.0e-2),
    ]
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = diagm(ones(T, 4))      # 4×4 Matrix{T}, Hermitian, nonsingular
        Atall = rand(T, 4, 2)
        Awide = rand(T, 2, 4)

        # decompositions
        # --------------
        qr_compact(A)
        qr_full(Atall)
        qr_null(Atall)

        lq_compact(A)
        lq_full(Awide)
        lq_null(Awide)

        svd_compact(A)
        svd_full(A)
        svd_vals(A)

        schur_full(A)
        schur_vals(A)

        eigh_full(A)
        eigh_vals(A)

        eig_full(A)
        eig_vals(A)

        gen_eig_full(A, A)
        gen_eig_vals(A, A)

        left_polar(A)
        right_polar(A)

        # derived decompositions
        left_orth(A)
        left_null(A)
        right_orth(A)
        right_null(A)

        # truncated decompositions
        for trunc in truncation_strategies
            svd_trunc(A; trunc)
            eigh_trunc(A; trunc)
            eig_trunc(A; trunc)
        end
        left_orth(A; trunc = truncrank(2))
        right_orth(A; trunc = truncrank(2))

        # projections
        project_hermitian(A)
        project_antihermitian(A)
        project_isometric(A)

        # properties
        isisometric(A)
        isunitary(A)
        ishermitian(A)
        isantihermitian(A)
    end
end
