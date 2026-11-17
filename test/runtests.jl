using ParallelTestRunner
using MatrixAlgebraKit

# Start with autodiscovered tests
testsuite = find_tests(@__DIR__)

# remove testsuite
filter!(!(startswith("testsuite") ∘ first), testsuite)

# remove utils
delete!(testsuite, "linearmap")

# Parse arguments
args = parse_args(ARGS; custom = ["fast"])

fast = !isnothing(args.custom["fast"])
fast && @info "Selected fast tests"

if filter_tests!(testsuite, args)
    # don't run all tests on GPU, only the GPU specific ones
    is_buildkite = get(ENV, "BUILDKITE", "false") == "true"
    if is_buildkite
        delete!(testsuite, "common/algorithms")
        delete!(testsuite, "common/codequality")
        delete!(testsuite, "common/jet")
        delete!(testsuite, "common/truncate")
        delete!(testsuite, "decompositions/gen_eig")
        filter!(p -> !startswith(first(p), "chainrules/"), testsuite)
    end
end

# JET tests: JET ≥ 0.12 (Julia 1.12+) signals through `JET_AVAILABLE` whether it is functional,
# and loads empty stubs when it is not. Older JET versions, still needed on Julia < 1.12, do not
# define it, hence the `isdefined` check instead of `using JET: JET_AVAILABLE`.
using JET: JET
const jet_new_generation = isdefined(JET, :JET_AVAILABLE)
const jet_available = !jet_new_generation || JET.JET_AVAILABLE
# whole-package analysis is pinned to the JET 0.12 generation, so its reports need not be curated for several JET/Julia combinations
(jet_new_generation && jet_available) || delete!(testsuite, "common/jet")

# Enzyme test workers peak at 5-7 GB (qr, lq, orthnull on Julia 1.10), so the default
# assumption of 2 GiB per worker oversubscribes memory on CI runners and makes them swap
memory_per_worker = any(startswith("enzyme/") ∘ first, testsuite) ? 6 * 2^30 : 2 * 2^30

runtests(MatrixAlgebraKit, args; testsuite, init_code = :(const fast_tests = $fast), memory_per_worker)
