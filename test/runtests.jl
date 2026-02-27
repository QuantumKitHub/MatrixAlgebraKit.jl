using ParallelTestRunner
using MatrixAlgebraKit

# Start with autodiscovered tests
testsuite = find_tests(@__DIR__)

# remove testsuite
filter!(!(startswith("testsuite") ∘ first), testsuite)

# remove utils
delete!(testsuite, "utilities")
delete!(testsuite, "linearmap")

# Parse arguments
args = parse_args(ARGS)

if filter_tests!(testsuite, args)
    # don't run all tests on GPU, only the GPU specific ones
    is_buildkite = get(ENV, "BUILDKITE", "false") == "true"
    if is_buildkite
        delete!(testsuite, "algorithms")
        delete!(testsuite, "truncate")
        delete!(testsuite, "gen_eig")
        delete!(testsuite, "chainrules")
        delete!(testsuite, "codequality")
    else
        is_apple_ci = Sys.isapple() && get(ENV, "CI", "false") == "true"
        if is_apple_ci
            filter!(p -> !startswith(first(p), "mooncake/"), testsuite)
            delete!(testsuite, "chainrules")
        end
        (Sys.iswindows() || is_apple_ci) && filter!(p -> !startswith(first(p), "enzyme/"), testsuite)
    end
end

runtests(MatrixAlgebraKit, args; testsuite)
