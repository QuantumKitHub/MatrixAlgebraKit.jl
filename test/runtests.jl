using ParallelTestRunner
using MatrixAlgebraKit

# Start with autodiscovered tests
testsuite = find_tests(@__DIR__)

# remove testsuite
filter!(!(startswith("testsuite") ∘ first), testsuite)

# remove utils
delete!(testsuite, "utilities")
delete!(testsuite, "ad_utils")

# Parse arguments
args = parse_args(ARGS)

if filter_tests!(testsuite, args)
    # don't run all tests on GPU, only the GPU specific ones
    is_buildkite = get(ENV, "BUILDKITE", "false") == "true"
    if is_buildkite
        delete!(testsuite, "algorithms")
        delete!(testsuite, "truncate")
        delete!(testsuite, "gen_eig")
        delete!(testsuite, "mooncake")
        delete!(testsuite, "chainrules")
        delete!(testsuite, "codequality")
    end
end

runtests(MatrixAlgebraKit, args; testsuite)
