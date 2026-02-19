"""
    call_and_zero!(f!, A, alg)

Helper for testing in-place Mooncake rules.
Calls `f!(A, alg)`, followed by zeroing out `A` and returns the output of `f!`.
This allows `Mooncake.TestUtils.test_rule` to verify the reverse rule of `f!` through finite differences,
without counting the contributions of `A`, as this is used solely as scratch space.
"""
function call_and_zero!(f!, A, alg)
    F′ = f!(A, alg)
    MatrixAlgebraKit.zero!(A)
    return F′
end

"""
    test_mooncake(T, sz; kwargs...)

Run all Mooncake AD tests for element type `T` and size `sz`. Dispatches to per-decomposition
sub-suites. Square or vector sizes enable the eigendecomposition tests; element types that are
plain number types enable the orthnull tests.
"""
function test_mooncake(T::Type, sz; kwargs...)
    summary_str = testargs_summary(T, sz)
    return @testset "Mooncake AD $summary_str" verbose = true begin
        test_mooncake_qr(T, sz; kwargs...)
        test_mooncake_lq(T, sz; kwargs...)
        if length(sz) == 1 || sz[1] == sz[2]
            test_mooncake_eig(T, sz; kwargs...)
            test_mooncake_eigh(T, sz; kwargs...)
        end
        test_mooncake_svd(T, sz; kwargs...)
        test_mooncake_polar(T, sz; kwargs...)
        # doesn't work for Diagonals yet?
        if T <: Number
            test_mooncake_orthnull(T, sz; kwargs...)
        end
    end
end
