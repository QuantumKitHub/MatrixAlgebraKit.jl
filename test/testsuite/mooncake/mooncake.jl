using TestExtras
using MatrixAlgebraKit
using Mooncake, Mooncake.TestUtils
using Mooncake: rrule!!
using MatrixAlgebraKit: diagview, TruncatedAlgorithm, PolarViaSVD, eigh_trunc
using LinearAlgebra: BlasFloat
using GenericLinearAlgebra


make_mooncake_tangent(ΔAelem::T) where {T <: Number} = ΔAelem
make_mooncake_tangent(ΔA::AbstractMatrix) = ΔA
make_mooncake_tangent(ΔA::AbstractVector) = ΔA
make_mooncake_tangent(ΔD::Diagonal) = Mooncake.build_tangent(typeof(ΔD), diagview(ΔD))

make_mooncake_tangent(T::Tuple) = Mooncake.build_tangent(typeof(T), make_mooncake_tangent.(T)...)

make_mooncake_fdata(x) = make_mooncake_tangent(x)
make_mooncake_fdata(x::Diagonal) = Mooncake.FData((diag = make_mooncake_tangent(x.diag),))
make_mooncake_fdata(x::Tuple) = map(make_mooncake_fdata, x)

# copies a preset tangent into a Mooncake CoDual
# for use in the pullback.
function copy_tangent(var::Mooncake.CoDual, Δargs)
    dargs = make_mooncake_fdata(deepcopy(Δargs))
    copyto!(Mooncake.tangent(var), dargs)
    return
end

function copy_tangent(var::Mooncake.CoDual, Δargs::Tuple)
    dargs = make_mooncake_fdata.(deepcopy(Δargs))
    for (var_tangent, darg) in zip(Mooncake.tangent(var), dargs)
        if var_tangent isa Mooncake.FData
            for (var_f, darg_f) in zip(Mooncake._fields(var_tangent), Mooncake._fields(darg))
                copyto!(var_f, darg_f)
            end
        else
            copyto!(var_tangent, darg)
        end
    end
    return
end

# no `alg` argument
function _get_copying_derivative(f, rrule, A, ΔA, args, Δargs, ::Nothing, rdata)
    dA_copy = make_mooncake_fdata(copy(ΔA))
    A_copy = copy(A)
    A_dA = Mooncake.CoDual(A_copy, dA_copy)
    copy_out, copy_pb!! = rrule(Mooncake.CoDual(f, Mooncake.NoFData()), A_dA)
    # copy Δargs into tangent of the output variable for the pullback check
    copy_tangent(copy_out, Δargs)
    copy_pb!!(rdata)
    @test Mooncake.primal(A_dA) == A
    return dA_copy, Mooncake.tangent(copy_out)
end

# `alg` argument
function _get_copying_derivative(f, rrule, A, ΔA, args, Δargs, alg, rdata)
    dA_copy = make_mooncake_fdata(copy(ΔA))
    A_copy = copy(A)
    A_dA = Mooncake.CoDual(A_copy, dA_copy)
    copy_out, copy_pb!! = rrule(Mooncake.CoDual(f, Mooncake.NoFData()), A_dA, Mooncake.CoDual(alg, Mooncake.NoFData()))
    # copy Δargs into tangent of the output variable for the pullback check
    copy_tangent(copy_out, Δargs)
    copy_pb!!(rdata)
    @test Mooncake.primal(A_dA) == A
    return dA_copy, Mooncake.tangent(copy_out)
end

function _get_inplace_derivative(f!, A, ΔA, args, Δargs, ::Nothing, rdata; ȳ = Δargs)
    dA_inplace = make_mooncake_fdata(copy(ΔA))
    A_inplace = copy(A)
    args_copy = deepcopy(args)
    dargs_inplace = make_mooncake_fdata(deepcopy(Δargs))
    # not every f! has a handwritten rrule!!
    inplace_sig = Tuple{typeof(f!), typeof(A), typeof(args)}
    has_handwritten_rule = hasmethod(Mooncake.rrule!!, inplace_sig)
    A_dA = Mooncake.CoDual(A_inplace, dA_inplace)
    args_dargs = Mooncake.CoDual(args_copy, dargs_inplace)
    if has_handwritten_rule
        inplace_out, inplace_pb!! = Mooncake.rrule!!(Mooncake.CoDual(f!, Mooncake.NoFData()), A_dA, args_dargs)
    else
        inplace_sig = Tuple{typeof(f!), typeof(A), typeof(args)}
        rvs_interp = Mooncake.get_interpreter(Mooncake.ReverseMode)
        inplace_rrule = Mooncake.build_rrule(rvs_interp, inplace_sig)
        inplace_out, inplace_pb!! = inplace_rrule(Mooncake.CoDual(f!, Mooncake.NoFData()), A_dA, args_dargs)
    end
    # copy reference derivative of output ȳ into inplace_out
    # needed for inplace methods like svd_trunc! that generate
    # new output variables
    copy_tangent(inplace_out, ȳ)
    inplace_pb!!(rdata)
    @test Mooncake.primal(A_dA) == A
    @test Mooncake.primal(args_dargs) == args_copy
    return dA_inplace, Mooncake.tangent(inplace_out)
end

function _get_inplace_derivative(f!, A, ΔA, args, Δargs, alg, rdata; ȳ = Δargs)
    dA_inplace = make_mooncake_fdata(copy(ΔA))
    A_inplace = copy(A)
    args_copy = deepcopy(args)
    dargs_inplace = make_mooncake_fdata(deepcopy(Δargs))
    # not every f! has a handwritten rrule!!
    inplace_sig = Tuple{typeof(f!), typeof(A), typeof(args), typeof(alg)}
    has_handwritten_rule = hasmethod(Mooncake.rrule!!, inplace_sig)
    A_dA = Mooncake.CoDual(A_inplace, dA_inplace)
    args_dargs = Mooncake.CoDual(args_copy, dargs_inplace)
    if has_handwritten_rule
        inplace_out, inplace_pb!! = Mooncake.rrule!!(Mooncake.CoDual(f!, Mooncake.NoFData()), A_dA, args_dargs, Mooncake.CoDual(alg, Mooncake.NoFData()))
    else
        inplace_sig = Tuple{typeof(f!), typeof(A), typeof(args), typeof(alg)}
        rvs_interp = Mooncake.get_interpreter(Mooncake.ReverseMode)
        inplace_rrule = Mooncake.build_rrule(rvs_interp, inplace_sig)
        inplace_out, inplace_pb!! = inplace_rrule(Mooncake.CoDual(f!, Mooncake.NoFData()), A_dA, args_dargs, Mooncake.CoDual(alg, Mooncake.NoFData()))
    end
    # copy reference derivative of output ȳ into inplace_out
    # needed for inplace methods like svd_trunc! that generate
    # new output variables
    copy_tangent(inplace_out, ȳ)
    inplace_pb!!(rdata)
    @test Mooncake.primal(A_dA) == A
    @test Mooncake.primal(args_dargs) == args_copy
    return dA_inplace, Mooncake.tangent(inplace_out)
end

"""
    test_pullbacks_match(f!, f, A, args, Δargs, alg = nothing; rdata = Mooncake.NoRData())

Compare the result of running the *in-place, mutating* function `f!`'s reverse rule
with the result of running its *non-mutating* partner function `f`'s reverse rule.
We must compare directly because many of the mutating functions modify `A` as a
scratch workspace, making testing `f!` against finite differences infeasible.

The arguments to this function are:
  - `f!` the mutating, in-place version of the function (accepts `args` for the function result)
  - `f` the non-mutating version of the function (does not accept `args` for the function result)
  - `A` the input matrix to factorize
  - `args` preallocated output for `f!` (e.g. `Q` and `R` matrices for `qr_compact!`)
  - `Δargs` precomputed derivatives of `args` for pullbacks of `f` and `f!`, to ensure they receive the same input
  - `alg` optional algorithm keyword argument
  - `rdata` Mooncake reverse data to supply to the pullback, in case `f` and `f!` return scalar results (as truncating functions do)
"""
function test_pullbacks_match(f!, f, A, args, Δargs, alg = nothing; rdata = Mooncake.NoRData(), ȳ = deepcopy(Δargs))
    sig = isnothing(alg) ? Tuple{typeof(f), typeof(A)} : Tuple{typeof(f), typeof(A), typeof(alg)}
    rvs_interp = Mooncake.get_interpreter(Mooncake.ReverseMode)
    rrule = Mooncake.build_rrule(rvs_interp, sig)
    ΔA = randn(rng, eltype(A), size(A))

    copy_args = isa(args, Tuple) ? copy.(args) : copy(args)
    inplace_args = isa(args, Tuple) ? copy.(args) : copy(args)
    dA_copy, dargs_copy = _get_copying_derivative(f, rrule, A, ΔA, copy_args, ȳ, alg, rdata)
    dA_inplace, dargs_inplace = _get_inplace_derivative(f!, A, ΔA, inplace_args, Δargs, alg, rdata; ȳ)

    dA_inplace_ = Mooncake.arrayify(A, dA_inplace)[2]
    dA_copy_ = Mooncake.arrayify(A, dA_copy)[2]
    @test dA_inplace_ ≈ dA_copy_
    @test copy_args == inplace_args
    if dargs_copy isa Tuple
        for (darg_copy_, darg_inplace_) in zip(dargs_copy, dargs_inplace)
            if darg_copy_ isa Mooncake.FData
                for (c_f, i_f) in zip(Mooncake._fields(darg_copy_), Mooncake._fields(darg_inplace_))
                    @test c_f == i_f
                end
            else
                @test darg_copy_ == darg_inplace_
            end
        end
    else
        @test dargs_copy == dargs_inplace
    end
    return
end

function make_input_scratch!(f!, A, alg)
    F′ = f!(A, alg)
    MatrixAlgebraKit.zero!(A)
    return F′
end

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
