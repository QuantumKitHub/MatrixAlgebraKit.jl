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
        delete!(testsuite, "common/truncate")
        delete!(testsuite, "decompositions/gen_eig")
        filter!(p -> !startswith(first(p), "chainrules/"), testsuite)
    else
        is_apple_ci = Sys.isapple() && get(ENV, "CI", "false") == "true"
        is_windows_ci = Sys.iswindows() && get(ENV, "CI", "false") == "true"
        if is_apple_ci
            filter!(p -> !startswith(first(p), "mooncake/"), testsuite)
            filter!(p -> !startswith(first(p), "chainrules/"), testsuite)
        end
        (is_windows_ci || is_apple_ci) && filter!(p -> !startswith(first(p), "enzyme/"), testsuite)
    end
end

# Enzyme test workers peak at 5-7 GB (qr, lq, orthnull on Julia 1.10), so the default
# assumption of 2 GiB per worker oversubscribes memory on CI runners and makes them swap
memory_per_worker = any(startswith("enzyme/") ∘ first, testsuite) ? 6 * 2^30 : 2 * 2^30

runtests(MatrixAlgebraKit, args; testsuite, init_code = :(const fast_tests = $fast), memory_per_worker)
