function call_and_zero!(f!, A, alg)
    F′ = f!(A, alg)
    MatrixAlgebraKit.zero!(A)
    return F′
end

function test_enzyme(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "Enzyme AD $summary_str" begin
        test_enzyme_qr(T, sz; kwargs...)
        test_enzyme_lq(T, sz; kwargs...)
        if length(sz) == 1 || sz[1] == sz[2]
            test_enzyme_eig(T, sz; kwargs...)
            test_enzyme_eigh(T, sz; kwargs...)
        end
        test_enzyme_svd(T, sz; kwargs...)
        if eltype(T) <: BlasFloat # no Sylvester for BigFloat
            test_enzyme_polar(T, sz; kwargs...)
            test_enzyme_orthnull(T, sz; kwargs...)
        end
    end
end

is_cpu(A) = typeof(parent(A)) <: Array
