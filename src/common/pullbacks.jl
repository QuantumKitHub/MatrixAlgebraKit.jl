"""
    iszerotangent(x)

Return true if `x` is of a type that the different AD engines use to communicate
a (co)tangent that is identically zero. By overloading this method, and writing
pullback definitions in term of it, we will be able to hook into different AD
ecosystems
"""
function iszerotangent end

iszerotangent(::Any) = false
