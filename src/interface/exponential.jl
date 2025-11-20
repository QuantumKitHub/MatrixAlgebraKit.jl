# Exponential functions
# --------------
@functiondef exponential

# Algorithm selection
# -------------------
default_exponential_algorithm(A; kwargs...) = default_exponential_algorithm(typeof(A); kwargs...)
function default_exponential_algorithm(T::Type; kwargs...)
    return MatrixFunctionViaLA(; kwargs...)
end
function default_exponential_algorithm(::Type{T}; kwargs...) where {T <: Diagonal}
    return DiagonalAlgorithm(; kwargs...)
end

for f in (:exponential!,)
    @eval function default_algorithm(::typeof($f), ::Type{A}; kwargs...) where {A}
        return default_exponential_algorithm(A; kwargs...)
    end
end
