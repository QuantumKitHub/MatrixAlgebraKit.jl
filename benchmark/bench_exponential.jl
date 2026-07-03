# Benchmark and accuracy harness for `MatrixAlgebraKit.exponential`.
#
# For each element type it times every applicable exponential algorithm over a range of
# matrix sizes, and measures each algorithm's accuracy on matrices with a *known* spectrum.
#
# Accuracy inputs are built as `A = V · Diag(λ) · V⁻¹`, so the exact exponential
# `exp(A) = V · Diag(exp λ) · V⁻¹` is known analytically.  The eigenvalues `λ` are drawn
# from families chosen to help or hurt particular algorithms (small/wide/imaginary/stiff/
# ill-conditioned), and the eigenvectors `V` are either unitary (→ normal / Hermitian `A`)
# or a general basis with a prescribed condition number (→ non-normal `A`).  The analytic
# reference is formed at high precision (`Complex{BigFloat}` at `HIGH_BITS` bits).
#
# Usage:
#   julia --project=benchmark benchmark/bench_exponential.jl [--quick] [--csv path]
#
#   --quick     small sizes / short timing budget, for a fast smoke run
#   --csv PATH  write the flat results table to PATH (default: benchmark/results.csv)

using LinearAlgebra
using LinearAlgebra: BLAS
using Random
using Printf
using DelimitedFiles
using StableRNGs
using BenchmarkTools
using DoubleFloats
using MatrixAlgebraKit
using MatrixAlgebraKit: MatrixFunctionViaTaylor, MatrixFunctionViaLA,
    MatrixFunctionViaEig, MatrixFunctionViaEigh, QRIteration, GS, GLA
# Loading these makes the generic (BigFloat / Double64) eig / eigh drivers available.
using GenericSchur
using GenericLinearAlgebra

# ----------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------
const QUICK = "--quick" in ARGS
const CSV_PATH = let i = findfirst(==("--csv"), ARGS)
    i === nothing ? joinpath(@__DIR__, "results.csv") : ARGS[i + 1]
end

# Ordered by increasing precision: hardware BLAS floats, then double-double (~106 bits,
# software but fast), then arbitrary-precision BigFloat.
const ELTYPES = (
    Float64, ComplexF64, Double64, Complex{Double64}, BigFloat, Complex{BigFloat},
)

_isblas(::Type{T}) where {T} = T <: Union{Float32, Float64, ComplexF32, ComplexF64}
_isdouble(::Type{T}) where {T} = T <: Union{Double64, Complex{Double64}}

# Sizes for the timing sweep (generic types are far slower, so use fewer / smaller).
# Double64 is software but only ~10-30× slower than BLAS, so it gets an intermediate range;
# BigFloat is far slower still.
blas_sizes() = QUICK ? [8, 16, 32] : [8, 16, 32, 64, 128, 256]
double_sizes() = QUICK ? [8, 16] : [8, 16, 32, 64, 128]
generic_sizes() = QUICK ? [8, 16] : [8, 16, 32, 64]
perf_sizes(T) = _isblas(T) ? blas_sizes() : (_isdouble(T) ? double_sizes() : generic_sizes())

# Sizes for the accuracy sweep (kept small: high-precision reference is expensive).
acc_sizes() = QUICK ? [12] : [16, 48]

# Timing budget per (matrix, algorithm), in seconds.
perf_seconds(T) = QUICK ? 0.1 : (_isblas(T) ? 0.3 : 0.8)

# Precision of the analytic reference and its tolerance (must beat the tested precision).
const HIGH_BITS = QUICK ? 512 : 1024
const REF_TOL = QUICK ? big"1e-50" : big"1e-90"

# Designed condition number of the eigenvector basis `V` per family.
const GENCOND = 1.0e2
const ILLCOND = 1.0e8

# Column order for the printed tables.
const ALG_ORDER = ["Taylor", "LA", "Eig", "Eigh"]

# Accuracy cases: (eigenvalue family, eigenvector family).
const ACC_CASES = [
    (:hermitian, :unitary),       # real spectrum, unitary V  -> Hermitian / symmetric
    (:small, :unitary),           # tiny normal matrix        -> few squarings
    (:imaginary, :unitary),       # oscillatory normal matrix
    (:wide, :unitary),            # large-norm normal matrix  -> many squarings
    (:stiff, :general),           # widely varying real parts -> over/underflow
    (:wide, :general),            # non-normal, large norm
    (:small, :illconditioned),    # near-defective basis      -> eig-path cancellation
]

# Performance cases: (label, eigenvalue family, eigenvector family).
const PERF_CASES = [
    ("general", :wide, :general),   # non-normal, non-trivial norm
    ("hermitian", :hermitian, :unitary),  # exercises the Eigh path
]

# ----------------------------------------------------------------------------------------
# Random matrix helpers (support Complex{BigFloat}, which `randn` does not directly)
# ----------------------------------------------------------------------------------------
_randn(rng, ::Type{T}, dims...) where {T <: Real} = randn(rng, T, dims...)
function _randn(rng, ::Type{Complex{T}}, dims...) where {T <: Real}
    return randn(rng, T, dims...) .+ im .* randn(rng, T, dims...)
end

unitary_matrix(rng, ::Type{RT}, n) where {RT} = Matrix(qr(_randn(rng, RT, n, n)).Q)

# General basis with singular values geometrically spread from 1 to `targetcond`.
function general_matrix(rng, ::Type{RT}, n, targetcond) where {RT}
    U = unitary_matrix(rng, RT, n)
    W = unitary_matrix(rng, RT, n)
    s = exp.(range(0.0, log(targetcond); length = n))
    S = Diagonal(real(RT).(s))
    return U * S * W
end

# ----------------------------------------------------------------------------------------
# Spectra
# ----------------------------------------------------------------------------------------
function sample_eig(family, rng)::ComplexF64
    if family === :small
        return 0.15 * randn(rng) + 0.15im * (0.2 + abs(randn(rng)))
    elseif family === :wide
        r = exp(log(0.05) + (log(30.0) - log(0.05)) * rand(rng))
        θ = 0.1 + (π - 0.2) * rand(rng)
        return r * cis(θ)
    elseif family === :imaginary
        return im * (0.5 + 8.0 * rand(rng))
    elseif family === :stiff
        return (-40.0 + 44.0 * rand(rng)) + im * (0.3 + 0.5 * abs(randn(rng)))
    elseif family === :illconditioned
        return 0.5 * randn(rng) + im * (0.2 + 0.3 * abs(randn(rng)))
    else
        error("unknown eigenvalue family $family")
    end
end

# A conjugate-closed spectrum of length `n` (so a real matrix with this spectrum exists).
function make_spectrum(family, n, rng)::Vector{ComplexF64}
    if family === :hermitian
        return complex.(2 .* randn(rng, n))
    end
    half = fld(n, 2)
    μ = ComplexF64[]
    for _ in 1:half
        z = sample_eig(family, rng)
        push!(μ, z)
        push!(μ, conj(z))
    end
    if isodd(n)
        push!(μ, complex(real(sample_eig(family, rng))))
    end
    return μ
end

# Real block form of a conjugate-closed spectrum, together with its analytic exponential.
# Real eigenvalues sit on the diagonal; a complex pair a±bi becomes a 2×2 block [a b; -b a].
# Must be called inside a `setprecision` block so the BigFloat ops use `HIGH_BITS`.
function real_canonical(spectrum::Vector{ComplexF64})
    tol = 1.0e-9
    reals = BigFloat[]
    pairs = Tuple{BigFloat, BigFloat}[]
    for z in spectrum
        if abs(imag(z)) < tol
            push!(reals, BigFloat(real(z)))
        elseif imag(z) > 0
            push!(pairs, (BigFloat(real(z)), BigFloat(imag(z))))
        end
    end
    n = length(reals) + 2 * length(pairs)
    D = zeros(BigFloat, n, n)
    E = zeros(BigFloat, n, n)
    i = 1
    for a in reals
        D[i, i] = a
        E[i, i] = exp(a)
        i += 1
    end
    for (a, b) in pairs
        D[i, i] = a
        D[i, i + 1] = b
        D[i + 1, i] = -b
        D[i + 1, i + 1] = a
        ea, cb, sb = exp(a), cos(b), sin(b)
        E[i, i] = ea * cb
        E[i, i + 1] = ea * sb
        E[i + 1, i] = -ea * sb
        E[i + 1, i + 1] = ea * cb
        i += 2
    end
    return D, E
end

# ----------------------------------------------------------------------------------------
# Case construction: returns the input matrix at type `T` and the analytic reference.
# ----------------------------------------------------------------------------------------
function build_case(::Type{T}, n, eigfam, evecfam, rng) where {T}
    RT_hi = (T <: Complex) ? Complex{BigFloat} : BigFloat
    targetcond = evecfam === :illconditioned ? ILLCOND :
        (evecfam === :general ? GENCOND : 1.0)
    spectrum = make_spectrum(eigfam, n, rng)

    A_hi, expA_hi = setprecision(BigFloat, HIGH_BITS) do
        V = evecfam === :unitary ? unitary_matrix(rng, RT_hi, n) :
            general_matrix(rng, RT_hi, n, targetcond)
        Vi = inv(V)
        if T <: Complex
            λ = Complex{BigFloat}.(spectrum)
            A = V * Diagonal(λ) * Vi
            expA = V * Diagonal(exp.(λ)) * Vi
            return Complex{BigFloat}.(A), Complex{BigFloat}.(expA)
        else
            D, E = real_canonical(spectrum)
            A = V * D * Vi
            expA = V * E * Vi
            return A, Complex{BigFloat}.(expA)
        end
    end

    # Round the input down to the tested precision (default BigFloat precision for generic T).
    A_T = (T <: Complex) ? Complex{real(T)}.(A_hi) : real(T).(A_hi)
    herm = (eigfam === :hermitian && evecfam === :unitary)
    herm && (A_T = (A_T + A_T') / 2)  # enforce exact Hermiticity for the Eigh path
    return (A = Matrix(A_T), ref = expA_hi, condV = targetcond, hermitian = herm)
end

# ----------------------------------------------------------------------------------------
# Algorithm selection per input
# ----------------------------------------------------------------------------------------
_trydefault(f, A) = try
    f(typeof(A))
catch
    nothing
end

# The library only registers default eig / eigh algorithms for BLAS and BigFloat element
# types, but GenericSchur / GenericLinearAlgebra actually drive any `AbstractFloat`.  For
# types without a registered default (e.g. Double64) fall back to those drivers directly,
# using the same `QRIteration` the BigFloat default resolves to.
function eig_algorithm_for(A)
    eig = _trydefault(MatrixAlgebraKit.default_eig_algorithm, A)
    return eig !== nothing ? eig : QRIteration(; driver = GS())
end
function eigh_algorithm_for(A)
    eigh = _trydefault(MatrixAlgebraKit.default_eigh_algorithm, A)
    return eigh !== nothing ? eigh : QRIteration(; driver = GLA())
end

function algorithms_for(A, hermitian::Bool)
    T = eltype(A)
    algs = Tuple{String, Any}[("Taylor", MatrixFunctionViaTaylor())]
    if T <: LinearAlgebra.BlasFloat
        push!(algs, ("LA", MatrixFunctionViaLA()))
    end
    eig = eig_algorithm_for(A)
    eig !== nothing && push!(algs, ("Eig", MatrixFunctionViaEig(eig)))
    if hermitian
        eigh = eigh_algorithm_for(A)
        eigh !== nothing && push!(algs, ("Eigh", MatrixFunctionViaEigh(eigh)))
    end
    return algs
end

# ----------------------------------------------------------------------------------------
# Accuracy metrics
# ----------------------------------------------------------------------------------------
function relerr(X, ref)
    return setprecision(BigFloat, HIGH_BITS) do
        Xc = Complex{BigFloat}.(X)
        Float64(opnorm(Xc .- ref, 1) / opnorm(ref, 1))
    end
end

# Independent high-precision recomputation of exp of the *actual* input, so algorithms are
# also compared to a common reference for the exact matrix they were handed.
function highprec_exp(A_T)
    return setprecision(BigFloat, HIGH_BITS) do
        Ahi = Complex{BigFloat}.(A_T)
        exponential(Ahi, MatrixFunctionViaTaylor(; tol = REF_TOL, balance = true))
    end
end

# Reference-free consistency checks.
function expexp_err(A, alg)
    E = exponential(A, alg)
    Em = exponential(-A, alg)
    return Float64(opnorm(E * Em - I, 1))
end
function det_err(A, E)
    return try
        d = det(E)
        target = exp(tr(A))
        Float64(abs(d - target) / abs(target))
    catch
        NaN
    end
end

# ----------------------------------------------------------------------------------------
# Result rows
# ----------------------------------------------------------------------------------------
mutable struct Row
    pass::String
    size::Int
    eltype::String
    eigfam::String
    evecfam::String
    alg::String
    time_s::Float64
    allocs::Float64
    analytic_err::Float64
    recomp_err::Float64
    expexp_err::Float64
    det_err::Float64
    condV::Float64
    norm1::Float64
    status::String
end

const ROWS = Row[]

# ----------------------------------------------------------------------------------------
# Passes
# ----------------------------------------------------------------------------------------
function run_accuracy!()
    println("\n", "="^80)
    println("ACCURACY PASS  (analytic reference at $HIGH_BITS bits)")
    println("="^80)
    for T in ELTYPES
        for n in acc_sizes()
            for (eigfam, evecfam) in ACC_CASES
                rng = StableRNG(123)
                case = build_case(T, n, eigfam, evecfam, rng)
                A = case.A
                norm1 = Float64(opnorm(A, 1))
                recomp = highprec_exp(A)
                for (name, alg) in algorithms_for(A, case.hermitian)
                    row = Row(
                        "acc", n, string(T), string(eigfam), string(evecfam), name,
                        NaN, NaN, NaN, NaN, NaN, NaN, case.condV, norm1, "ok",
                    )
                    try
                        E = exponential(A, alg)
                        row.analytic_err = relerr(E, case.ref)
                        row.recomp_err = relerr(E, recomp)
                        row.expexp_err = expexp_err(A, alg)
                        row.det_err = det_err(A, E)
                    catch e
                        row.status = "error: $(sprint(showerror, e))"
                    end
                    push!(ROWS, row)
                end
            end
        end
    end
    return
end

function run_performance!()
    println("\n", "="^80)
    println("PERFORMANCE PASS")
    println("="^80)
    for T in ELTYPES
        sec = perf_seconds(T)
        BenchmarkTools.DEFAULT_PARAMETERS.seconds = sec
        for n in perf_sizes(T)
            for (label, eigfam, evecfam) in PERF_CASES
                rng = StableRNG(123)
                case = build_case(T, n, eigfam, evecfam, rng)
                A = case.A
                norm1 = Float64(opnorm(A, 1))
                for (name, alg) in algorithms_for(A, case.hermitian)
                    row = Row(
                        "perf", n, string(T), label, string(evecfam), name,
                        NaN, NaN, NaN, NaN, NaN, NaN, case.condV, norm1, "ok",
                    )
                    try
                        b = @benchmark exponential($A, $alg)
                        tr_ = minimum(b)
                        row.time_s = tr_.time / 1.0e9
                        row.allocs = Float64(tr_.allocs)
                    catch e
                        row.status = "error: $(sprint(showerror, e))"
                    end
                    push!(ROWS, row)
                end
            end
        end
    end
    return
end

# ----------------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------------
fmt_time(x) = isnan(x) ? "     -   " : (
        x < 1.0e-3 ? @sprintf("%7.1fµs", x * 1.0e6) :
        x < 1.0 ? @sprintf("%7.2fms", x * 1.0e3) : @sprintf("%7.3fs ", x)
    )
fmt_err(x) = isnan(x) ? "    -    " : @sprintf("%9.2e", x)

function print_perf_tables()
    for T in ELTYPES
        for (label, _, _) in PERF_CASES
            rows = filter(r -> r.pass == "perf" && r.eltype == string(T) && r.eigfam == label, ROWS)
            isempty(rows) && continue
            algs = filter(a -> any(r -> r.alg == a, rows), ALG_ORDER)
            println("\n[$T]  timing — $label matrix   (min time per call)")
            @printf("  %5s", "n")
            for a in algs
                @printf("  %10s", a)
            end
            println()
            for n in sort(unique(r.size for r in rows))
                @printf("  %5d", n)
                for a in algs
                    r = findfirst(x -> x.size == n && x.alg == a, rows)
                    @printf("  %10s", r === nothing ? "-" : strip(fmt_time(rows[r].time_s)))
                end
                println()
            end
        end
    end
    return
end

function print_acc_tables()
    for T in ELTYPES
        rows = filter(r -> r.pass == "acc" && r.eltype == string(T), ROWS)
        isempty(rows) && continue
        algs = filter(a -> any(r -> r.alg == a, rows), ALG_ORDER)
        println("\n[$T]  accuracy — relative error vs analytic reference")
        @printf("  %-26s %5s", "case (eig/evec)", "n")
        for a in algs
            @printf("  %9s", a)
        end
        println()
        for n in sort(unique(r.size for r in rows))
            for (eigfam, evecfam) in ACC_CASES
                sel = filter(r -> r.size == n && r.eigfam == string(eigfam) && r.evecfam == string(evecfam), rows)
                isempty(sel) && continue
                @printf("  %-26s %5d", "$eigfam/$evecfam", n)
                for a in algs
                    r = findfirst(x -> x.alg == a, sel)
                    @printf("  %9s", r === nothing ? "-" : strip(fmt_err(sel[r].analytic_err)))
                end
                println()
            end
        end
    end
    return
end

# The accuracy and timing passes produce disjoint columns, so they are written to two
# separate files rather than one flat table with half the columns empty.
function _write_csv(path, header, rows, cols)
    return open(path, "w") do io
        writedlm(io, permutedims(header), ',')
        for r in rows
            writedlm(io, permutedims(Any[getfield(r, c) for c in cols]), ',')
        end
    end
end

function write_csv(path)
    base, ext = splitext(path)
    isempty(ext) && (ext = ".csv")
    acc_path = base * "-accuracy" * ext
    perf_path = base * "-timing" * ext

    acc = filter(r -> r.pass == "acc", ROWS)
    perf = filter(r -> r.pass == "perf", ROWS)

    _write_csv(
        acc_path,
        [
            "size", "eltype", "eig_family", "evec_family", "algorithm",
            "analytic_err", "recomp_err", "expexp_err", "det_err", "condV", "norm1", "status",
        ],
        acc,
        (
            :size, :eltype, :eigfam, :evecfam, :alg,
            :analytic_err, :recomp_err, :expexp_err, :det_err, :condV, :norm1, :status,
        ),
    )
    _write_csv(
        perf_path,
        [
            "size", "eltype", "case", "evec_family", "algorithm",
            "time_s", "allocs", "condV", "norm1", "status",
        ],
        perf,
        (:size, :eltype, :eigfam, :evecfam, :alg, :time_s, :allocs, :condV, :norm1, :status),
    )
    println("\nWrote $(length(acc)) accuracy rows to $acc_path")
    return println("Wrote $(length(perf)) timing rows to $perf_path")
end

# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------
function main()
    # Single-threaded BLAS for a clean, reproducible algorithm-to-algorithm comparison.
    BLAS.set_num_threads(1)
    @info "matrix-exponential benchmark" QUICK HIGH_BITS blas_threads = BLAS.get_num_threads() types = ELTYPES
    run_accuracy!()
    run_performance!()
    print_acc_tables()
    print_perf_tables()
    write_csv(CSV_PATH)
    return nothing
end

main()
