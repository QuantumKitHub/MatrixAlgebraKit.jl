"""
    isisometry(A; kwargs...) -> Bool

Test whether a linear map is an isometry, ie `A' * A ≈ I`. 
The `kwargs` are passed on to `isapprox` to control the tolerances.
"""
isisometry(A::AbstractMatrix; kwargs...) = isapprox(A' * A, LinearAlgebra.I; kwargs...)
