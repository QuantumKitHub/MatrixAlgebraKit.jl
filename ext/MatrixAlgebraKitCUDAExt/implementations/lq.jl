function MatrixAlgebraKit.default_lq_algorithm(A::CuMatrix{<:BlasFloat}; kwargs...)
    qr_alg = CUSOLVER_HouseholderQR(; kwargs...)
    return LQViaTransposedQR(qr_alg)
end