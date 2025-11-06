# Exponetial functions
# --------------

# TODO: docs
@functiondef exp

# Algorithm selection
# -------------------
default_exp_algorithm(A; kwargs...) = default_exp_algorithm(typeof(A); kwargs...)
function default_exp_algorithm(T::Type; kwargs...)
    return LA_exponential(; kwargs...)
end

for f in (:exp!,)
    @eval function default_algorithm(::typeof($f), ::Type{A}; kwargs...) where {A}
        return default_exp_algorithm(A; kwargs...)
    end
end
